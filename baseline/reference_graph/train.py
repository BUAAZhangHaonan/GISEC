from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.reference_graph.dataset import FragmentGraphMergeDataset, collate_fragment_graph_batch
from baseline.reference_graph.eval import (
    DEFAULT_REFERENCE_GRAPH_THRESHOLDS,
    compute_edge_metrics,
    evaluate_reference_graph_loader,
)
from baseline.reference_graph.model import ReferenceGraphMergeModel


def _weighted_edge_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    positive_edge_weight: float,
    negative_edge_weight: float,
) -> torch.Tensor:
    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]
    per_edge = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction="none")
    weights = torch.where(
        valid_targets > 0.5,
        torch.full_like(valid_targets, float(positive_edge_weight)),
        torch.full_like(valid_targets, float(negative_edge_weight)),
    )
    return (per_edge * weights).mean()


def train_reference_graph_merge(
    *,
    cache_root: str,
    reference_root: str,
    output_dir: str,
    split: str,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    reference_image_size: int = 128,
    reference_max_views: int = 16,
    reference_view_sampler: str = "pose_farthest",
    hidden_dim: int = 64,
    reference_hidden_dim: int = 32,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    max_train_steps: int = 0,
    val_split: str | None = None,
    decision_threshold: float = 0.5,
    positive_edge_weight: float = 1.0,
    negative_edge_weight: float = 1.0,
    eval_thresholds: tuple[float, ...] = DEFAULT_REFERENCE_GRAPH_THRESHOLDS,
    conservative_f1_margin: float = 0.005,
) -> None:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset = FragmentGraphMergeDataset(
        cache_root=cache_root,
        reference_root=reference_root,
        split=split,
        reference_image_size=int(reference_image_size),
        reference_max_views=int(reference_max_views),
        reference_view_sampler=str(reference_view_sampler),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_fragment_graph_batch,
    )
    val_loader = None
    if val_split is not None:
        val_dataset = FragmentGraphMergeDataset(
            cache_root=cache_root,
            reference_root=reference_root,
            split=str(val_split),
            reference_image_size=int(reference_image_size),
            reference_max_views=int(reference_max_views),
            reference_view_sampler=str(reference_view_sampler),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(int(batch_size), 1),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=collate_fragment_graph_batch,
        )
    probe = dataset[0]
    model = ReferenceGraphMergeModel(
        node_dim=int(probe["node_features"].shape[1]),
        edge_dim=int(probe["edge_features"].shape[1]),
        reference_dim=int(probe["reference_features"].shape[0]),
        hidden_dim=int(hidden_dim),
        reference_hidden_dim=int(reference_hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    start = time.time()
    step_count = 0
    loss_total = 0.0
    accuracy_total = 0.0
    edge_positive_rate_total = 0.0
    pred_positive_rate_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0
    best_val_summary: dict[str, float] | None = None

    for _epoch in range(int(epochs)):
        model.train()
        for batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            logits = model(batch)
            valid_mask = ~batch["edge_ignore_mask"].to(torch.bool)
            if logits.numel() == 0 or not bool(valid_mask.any()):
                continue
            loss = _weighted_edge_loss(
                logits,
                batch["edge_targets"],
                valid_mask,
                positive_edge_weight=float(positive_edge_weight),
                negative_edge_weight=float(negative_edge_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            metrics = compute_edge_metrics(
                logits.detach()[valid_mask.detach()].cpu(),
                batch["edge_targets"].detach()[valid_mask.detach()].cpu(),
                threshold=float(decision_threshold),
            )
            step_count += 1
            loss_total += float(loss.item())
            accuracy_total += float(metrics["edge_accuracy"])
            edge_positive_rate_total += float(metrics["edge_positive_rate"])
            pred_positive_rate_total += float(metrics["pred_positive_rate"])
            precision_total += float(metrics["precision"])
            recall_total += float(metrics["recall"])
            f1_total += float(metrics["f1"])
            if max_train_steps > 0 and step_count >= int(max_train_steps):
                break
        if val_loader is not None:
            val_eval = evaluate_reference_graph_loader(
                model=model,
                loader=val_loader,
                device=device,
                thresholds=eval_thresholds,
                conservative_f1_margin=float(conservative_f1_margin),
            )
            val_summary = {
                **dict(val_eval["best"]),
                "loss_total": float(val_eval["loss_total"]),
                "best_threshold": float(dict(val_eval["best"])["threshold"]),
                "best_conservative_threshold": float(dict(val_eval["best_conservative"])["threshold"]),
                "best_conservative_f1": float(dict(val_eval["best_conservative"])["f1"]),
                "best_conservative_precision": float(dict(val_eval["best_conservative"])["precision"]),
                "best_conservative_pred_positive_rate": float(dict(val_eval["best_conservative"])["pred_positive_rate"]),
            }
            if (
                best_val_summary is None
                or float(val_summary["f1"]) > float(best_val_summary["f1"])
                or (
                    float(val_summary["f1"]) == float(best_val_summary["f1"])
                    and float(val_summary["precision"]) > float(best_val_summary["precision"])
                )
            ):
                best_val_summary = dict(val_summary)
                torch.save(model.state_dict(), artifact_root / "model_best.pth")
                (artifact_root / "val_summary.json").write_text(
                    json.dumps(best_val_summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                (artifact_root / "val_threshold_sweep.json").write_text(
                    json.dumps(
                        {
                            "loss_total": float(val_eval["loss_total"]),
                            "rows": list(val_eval["rows"]),
                            "best": dict(val_eval["best"]),
                            "best_conservative": dict(val_eval["best_conservative"]),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        if max_train_steps > 0 and step_count >= int(max_train_steps):
            break

    torch.save(model.state_dict(), artifact_root / "model_final.pth")
    summary = {
        "cache_root": str(Path(cache_root).resolve()),
        "reference_root": str(Path(reference_root).resolve()),
        "split": str(split),
        "epochs": int(epochs),
        "steps": int(step_count),
        "reference_mode": str(dataset.prototype_source.is_single_bank and "single_bank" or "multi_bank"),
        "decision_threshold": float(decision_threshold),
        "positive_edge_weight": float(positive_edge_weight),
        "negative_edge_weight": float(negative_edge_weight),
        "loss_total": 0.0 if step_count == 0 else loss_total / float(step_count),
        "edge_accuracy": 0.0 if step_count == 0 else accuracy_total / float(step_count),
        "edge_positive_rate": 0.0 if step_count == 0 else edge_positive_rate_total / float(step_count),
        "pred_positive_rate": 0.0 if step_count == 0 else pred_positive_rate_total / float(step_count),
        "precision": 0.0 if step_count == 0 else precision_total / float(step_count),
        "recall": 0.0 if step_count == 0 else recall_total / float(step_count),
        "f1": 0.0 if step_count == 0 else f1_total / float(step_count),
        "val_split": None if val_split is None else str(val_split),
        "best_val_f1": 0.0 if best_val_summary is None else float(best_val_summary["f1"]),
        "best_threshold": None if best_val_summary is None else float(best_val_summary["best_threshold"]),
        "best_conservative_threshold": None if best_val_summary is None else float(best_val_summary["best_conservative_threshold"]),
        "wall_time_sec": int(time.time() - start),
    }
    (artifact_root / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
