from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline.fragment_generator.dataset import FragmentGeneratorCacheDataset, collate_fragment_generator_batch
from baseline.fragment_generator.losses import fragment_generator_losses, match_fragment_slots
from baseline.fragment_generator.metrics import aggregate_fragment_quality_metrics, compute_fragment_quality_metrics
from baseline.fragment_generator.model import LocalFragmentGenerator


def _evaluate_fragment_generator(
    *,
    model: LocalFragmentGenerator,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metric_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(
                rgb_crop=batch["rgb_crop"],
                coarse_mask_logit_crop=batch["coarse_mask_logit_crop"],
                pixel_feature_crop=batch["pixel_feature_crop"],
            )
            metric_rows.append(
                compute_fragment_quality_metrics(
                    fragment_mask_logits=outputs["fragment_mask_logits"],
                    fragment_presence_logits=outputs["fragment_presence_logits"],
                    gt_fragment_masks=batch["gt_fragment_masks"],
                    gt_fragment_owner_ids=batch["gt_fragment_owner_ids"],
                    gt_instance_union_mask=batch["gt_instance_union_mask"],
                    overflow_crop=batch["overflow_crop"],
                )
            )
    return aggregate_fragment_quality_metrics(metric_rows)


def train_fragment_generator(
    *,
    cache_root: str,
    output_dir: str,
    split: str,
    device: torch.device,
    val_split: str | None = None,
    epochs: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    max_train_steps: int = 0,
    hidden_dim: int = 32,
    max_fragments: int = 6,
    learning_rate: float = 1.0e-3,
    compute_train_metrics: bool = True,
) -> None:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset = FragmentGeneratorCacheDataset(cache_root=cache_root, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_fragment_generator_batch,
    )
    probe = dataset[0]
    model = LocalFragmentGenerator(
        rgb_channels=int(probe["rgb_crop"].shape[0]),
        feature_channels=int(probe["pixel_feature_crop"].shape[0]),
        hidden_dim=int(hidden_dim),
        max_fragments=int(max_fragments),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    val_loader = None
    if val_split is not None:
        val_dataset = FragmentGeneratorCacheDataset(cache_root=cache_root, split=str(val_split))
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(int(batch_size), 1),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=collate_fragment_generator_batch,
        )

    start = time.time()
    step_count = 0
    loss_totals = {
        "loss_total": 0.0,
        "loss_mask": 0.0,
        "loss_presence": 0.0,
        "loss_coverage": 0.0,
        "loss_containment": 0.0,
        "loss_diversity": 0.0,
    }
    metric_rows: list[dict[str, float]] = []
    best_val_score: float | None = None

    for _epoch in range(int(epochs)):
        model.train()
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(
                rgb_crop=batch["rgb_crop"],
                coarse_mask_logit_crop=batch["coarse_mask_logit_crop"],
                pixel_feature_crop=batch["pixel_feature_crop"],
            )
            matches = match_fragment_slots(
                fragment_mask_logits=outputs["fragment_mask_logits"],
                gt_fragment_masks=batch["gt_fragment_masks"],
                gt_fragment_owner_ids=batch["gt_fragment_owner_ids"],
            )
            loss_rows = fragment_generator_losses(
                fragment_mask_logits=outputs["fragment_mask_logits"],
                fragment_presence_logits=outputs["fragment_presence_logits"],
                gt_fragment_masks=batch["gt_fragment_masks"],
                gt_fragment_owner_ids=batch["gt_fragment_owner_ids"],
                gt_instance_union_mask=batch["gt_instance_union_mask"],
                matches=matches,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_rows["loss_total"].backward()
            optimizer.step()
            step_count += 1
            for key in loss_totals:
                loss_totals[key] += float(loss_rows[key].item())
            if bool(compute_train_metrics):
                metric_rows.append(
                    compute_fragment_quality_metrics(
                        fragment_mask_logits=outputs["fragment_mask_logits"].detach(),
                        fragment_presence_logits=outputs["fragment_presence_logits"].detach(),
                        gt_fragment_masks=batch["gt_fragment_masks"].detach(),
                        gt_fragment_owner_ids=batch["gt_fragment_owner_ids"].detach(),
                        gt_instance_union_mask=batch["gt_instance_union_mask"].detach(),
                        overflow_crop=batch["overflow_crop"].detach(),
                    )
                )
            if int(max_train_steps) > 0 and step_count >= int(max_train_steps):
                break
        if int(max_train_steps) > 0 and step_count >= int(max_train_steps):
            break

    torch.save(model.state_dict(), artifact_root / "model_final.pth")
    train_metrics = aggregate_fragment_quality_metrics(metric_rows) if metric_rows else {}
    summary = {
        "cache_root": str(Path(cache_root).resolve()),
        "split": str(split),
        "epochs": int(epochs),
        "steps": int(step_count),
        "hidden_dim": int(hidden_dim),
        "max_fragments": int(max_fragments),
        "compute_train_metrics": bool(compute_train_metrics),
        "wall_time_sec": int(time.time() - start),
        **{key: 0.0 if step_count == 0 else float(value) / float(step_count) for key, value in loss_totals.items()},
        **train_metrics,
    }
    (artifact_root / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if val_loader is not None:
        val_summary = _evaluate_fragment_generator(model=model, loader=val_loader, device=device)
        (artifact_root / "val_summary.json").write_text(json.dumps(val_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        val_score = (
            float(val_summary["covered_gt_rate"])
            + float(val_summary["split_gt_rate"])
            - float(val_summary["impure_fragment_rate"])
            - float(val_summary["leakage_rate"])
            - float(val_summary["singleton_gt_rate"]) * 0.5
        )
        if best_val_score is None or val_score >= float(best_val_score):
            best_val_score = float(val_score)
            torch.save(model.state_dict(), artifact_root / "model_best.pth")
