from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.local_merger.dataset import LocalMergerPredictionDataset, collate_local_merger_batch
from baseline.local_merger.model import LocalMergeEdgeScorer


def _merge_components(num_nodes: int, edge_index: torch.Tensor, edge_scores: torch.Tensor, threshold: float = 0.5) -> list[int]:
    parent = list(range(int(num_nodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for (src, dst), score in zip(edge_index.t().tolist(), edge_scores.tolist()):
        if float(score) >= float(threshold):
            union(int(src), int(dst))
    return [find(index) for index in range(int(num_nodes))]


def _cluster_stats(num_nodes: int, edge_index: torch.Tensor, edge_scores: torch.Tensor) -> tuple[float, float]:
    if int(num_nodes) <= 0:
        return 0.0, 0.0
    roots = _merge_components(int(num_nodes), edge_index, edge_scores, threshold=0.5)
    counts: dict[int, int] = {}
    for root in roots:
        counts[int(root)] = counts.get(int(root), 0) + 1
    cluster_sizes = list(counts.values())
    singleton_rate = float(sum(1 for size in cluster_sizes if int(size) == 1)) / float(len(cluster_sizes))
    return singleton_rate, float(len(cluster_sizes))


def _summarize_local_merger(
    *,
    batch: dict,
    logits: torch.Tensor,
) -> dict[str, float]:
    invocation_count = 0
    fragments_total = 0.0
    singleton_total = 0.0
    clusters_total = 0.0
    same_pairs_total = 0
    same_pairs_covered = 0
    for sample_index, num_fragments in enumerate(batch["num_valid_fragments"]):
        start, end = batch["edge_sample_ranges"][sample_index]
        edge_logits = logits[start:end]
        edge_scores = torch.sigmoid(edge_logits.detach().cpu())
        num_nodes = int(num_fragments)
        if num_nodes > 1:
            invocation_count += 1
            fragments_total += float(num_nodes)
        singleton_rate, clusters_per_crop = _cluster_stats(
            num_nodes,
            batch["edge_index"][:, start:end] - int(batch["edge_index"][:, start:end].min().item()) if end > start else torch.zeros((2, 0), dtype=torch.long),
            edge_scores,
        )
        singleton_total += float(singleton_rate)
        clusters_total += float(clusters_per_crop)
        same_pairs_total += int(batch["same_instance_pairs_total"][sample_index])
        same_pairs_covered += int(batch["same_instance_pairs_covered"][sample_index])
    sample_count = float(max(len(batch["num_valid_fragments"]), 1))
    return {
        "local_graph_invocation_rate": float(invocation_count) / sample_count,
        "avg_fragments_per_invoked_crop": 0.0 if invocation_count <= 0 else float(fragments_total) / float(invocation_count),
        "same_instance_edge_recall": 0.0 if same_pairs_total <= 0 else float(same_pairs_covered) / float(same_pairs_total),
        "singleton_cluster_rate": float(singleton_total) / sample_count,
        "clusters_per_crop": float(clusters_total) / sample_count,
    }


def train_local_merger(
    *,
    prediction_root: str,
    output_dir: str,
    split: str,
    device: torch.device,
    val_split: str | None = None,
    epochs: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    max_train_steps: int = 0,
    hidden_dim: int = 32,
    learning_rate: float = 1.0e-3,
) -> None:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset = LocalMergerPredictionDataset(prediction_root=prediction_root, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_local_merger_batch,
    )
    probe = dataset[0]
    model = LocalMergeEdgeScorer(
        node_dim=int(probe["node_features"].shape[1]),
        edge_dim=int(probe["edge_features"].shape[1]),
        hidden_dim=int(hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    val_loader = None
    if val_split is not None:
        val_loader = DataLoader(
            LocalMergerPredictionDataset(prediction_root=prediction_root, split=str(val_split)),
            batch_size=max(int(batch_size), 1),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=collate_local_merger_batch,
        )

    start = time.time()
    step_count = 0
    loss_total = 0.0
    metric_rows: list[dict[str, float]] = []
    for _epoch in range(int(epochs)):
        model.train()
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            logits = model(
                node_features=batch["node_features"],
                edge_index=batch["edge_index"],
                edge_features=batch["edge_features"],
            )
            if logits.numel() > 0:
                targets = batch["edge_targets"].to(device)
                pos_count = max(float((targets > 0.5).sum().item()), 1.0)
                neg_count = max(float((targets <= 0.5).sum().item()), 1.0)
                pos_weight = torch.tensor(neg_count / pos_count, dtype=targets.dtype, device=device)
                loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            else:
                loss = batch["node_features"].sum() * 0.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step_count += 1
            loss_total += float(loss.item())
            metric_rows.append(_summarize_local_merger(batch=batch, logits=logits.detach().cpu()))
            if int(max_train_steps) > 0 and step_count >= int(max_train_steps):
                break
        if int(max_train_steps) > 0 and step_count >= int(max_train_steps):
            break

    torch.save(model.state_dict(), artifact_root / "model_final.pth")
    summary = {
        "prediction_root": str(Path(prediction_root).resolve()),
        "split": str(split),
        "epochs": int(epochs),
        "steps": int(step_count),
        "hidden_dim": int(hidden_dim),
        "wall_time_sec": int(time.time() - start),
        "loss_total": 0.0 if step_count == 0 else float(loss_total) / float(step_count),
    }
    if metric_rows:
        for key in metric_rows[0]:
            summary[key] = float(sum(float(row[key]) for row in metric_rows)) / float(len(metric_rows))
    (artifact_root / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if val_loader is not None:
        model.eval()
        val_rows: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                logits = model(
                    node_features=batch["node_features"],
                    edge_index=batch["edge_index"],
                    edge_features=batch["edge_features"],
                )
                val_rows.append(_summarize_local_merger(batch=batch, logits=logits.detach().cpu()))
        val_summary = {
            key: float(sum(float(row[key]) for row in val_rows)) / float(len(val_rows))
            for key in val_rows[0]
        } if val_rows else {}
        (artifact_root / "val_summary.json").write_text(json.dumps(val_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        torch.save(model.state_dict(), artifact_root / "model_best.pth")
