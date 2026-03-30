from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from baseline.instance_fragment_generator.dataset import InstanceFragmentCacheDataset, collate_instance_fragment_batch
from baseline.instance_fragment_generator.eval import evaluate_instance_fragment_generator
from baseline.instance_fragment_generator.losses import instance_fragment_losses, match_instance_fragment_slots
from baseline.instance_fragment_generator.metrics import (
    accumulate_instance_fragment_metric_counts,
    aggregate_instance_fragment_metric_counts,
    finalize_instance_fragment_metric_counts,
)
from baseline.instance_fragment_generator.model import InstanceLocalFragmentGenerator


def train_instance_fragment_generator(
    *,
    cache_root: str,
    dataset_root: str,
    output_dir: str,
    split: str,
    device: torch.device,
    val_split: str | None = None,
    epochs: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    max_train_steps: int = 0,
    hidden_dim: int = 32,
    num_queries: int = 8,
    learning_rate: float = 1.0e-3,
) -> dict[str, Any]:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset = InstanceFragmentCacheDataset(cache_root=cache_root, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_instance_fragment_batch,
    )
    probe = dataset[0]
    model = InstanceLocalFragmentGenerator(
        rgb_channels=int(probe["anchor_rgb_crop"].shape[0]),
        feature_channels=int(probe["anchor_feature_crop"].shape[0]),
        neighbor_channels=int(probe["neighbor_union_mask_crop"].shape[0]),
        hidden_dim=int(hidden_dim),
        num_queries=int(num_queries),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))

    step_count = 0
    start = time.time()
    loss_totals = {
        "loss_total": 0.0,
        "loss_mask": 0.0,
        "loss_presence": 0.0,
        "loss_coverage": 0.0,
        "loss_containment": 0.0,
        "loss_diversity": 0.0,
    }
    metric_count_rows: list[dict[str, float]] = []

    for _epoch in range(int(epochs)):
        model.train()
        for batch in loader:
            batch_device = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(
                anchor_rgb_crop=batch_device["anchor_rgb_crop"],
                anchor_mask_logit_crop=batch_device["anchor_mask_logit_crop"],
                anchor_feature_crop=batch_device["anchor_feature_crop"],
                neighbor_union_mask_crop=batch_device["neighbor_union_mask_crop"],
            )
            matches = match_instance_fragment_slots(
                fragment_mask_logits=outputs["fragment_mask_logits"],
                gt_fragment_masks=batch_device["gt_fragment_masks"],
                fragment_count=batch_device["fragment_count"],
            )
            loss_rows = instance_fragment_losses(
                fragment_mask_logits=outputs["fragment_mask_logits"],
                fragment_presence_logits=outputs["fragment_presence_logits"],
                gt_fragment_masks=batch_device["gt_fragment_masks"],
                fragment_count=batch_device["fragment_count"],
                anchor_gt_mask=batch_device["anchor_gt_mask"],
                matches=matches,
                is_negative=batch_device["is_negative"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss_rows["loss_total"].backward()
            optimizer.step()
            step_count += 1
            for key in loss_totals:
                loss_totals[key] += float(loss_rows[key].item())
            metric_count_rows.append(
                accumulate_instance_fragment_metric_counts(
                    fragment_mask_logits=outputs["fragment_mask_logits"].detach().cpu(),
                    fragment_presence_logits=outputs["fragment_presence_logits"].detach().cpu(),
                    gt_fragment_masks=batch["gt_fragment_masks"].detach().cpu(),
                    fragment_count=batch["fragment_count"].detach().cpu(),
                    anchor_gt_mask=batch["anchor_gt_mask"].detach().cpu(),
                    is_negative=batch["is_negative"].detach().cpu(),
                )
            )
            if int(max_train_steps) > 0 and step_count >= int(max_train_steps):
                break
        if int(max_train_steps) > 0 and step_count >= int(max_train_steps):
            break

    torch.save(model.state_dict(), artifact_root / "model_final.pth")
    model_config = {
        "cache_root": str(Path(cache_root).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": str(split),
        "val_split": None if val_split is None else str(val_split),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "learning_rate": float(learning_rate),
        "hidden_dim": int(hidden_dim),
        "num_queries": int(num_queries),
    }
    (artifact_root / "model_config.json").write_text(json.dumps(model_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_summary = finalize_instance_fragment_metric_counts(aggregate_instance_fragment_metric_counts(metric_count_rows))
    train_summary.update(
        {
            "cache_root": str(Path(cache_root).resolve()),
            "dataset_root": str(Path(dataset_root).resolve()),
            "split": str(split),
            "epochs": int(epochs),
            "steps": int(step_count),
            "batch_size": int(batch_size),
            "num_queries": int(num_queries),
            "hidden_dim": int(hidden_dim),
            "wall_time_sec": int(time.time() - start),
            **{key: 0.0 if step_count <= 0 else float(value) / float(step_count) for key, value in loss_totals.items()},
            "owner_union_segm/AP": 0.0,
            "owner_union_boundary/IoU": 0.0,
            "owner_union_split_gt_count": 0,
            "owner_union_merge_pred_count": 0,
            "owner_union_segm/AP_truncated": 0.0,
        }
    )

    val_summary: dict[str, Any] = {}
    if val_split is not None:
        eval_root = artifact_root / f"eval_{val_split}"
        val_summary = evaluate_instance_fragment_generator(
            cache_root=cache_root,
            dataset_root=dataset_root,
            output_dir=str(eval_root),
            split=str(val_split),
            device=device,
            model=model,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
        )
        (artifact_root / "val_summary.json").write_text(json.dumps(val_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for key in [
            "owner_union_segm/AP",
            "owner_union_boundary/IoU",
            "owner_union_split_gt_count",
            "owner_union_merge_pred_count",
            "owner_union_segm/AP_truncated",
        ]:
            train_summary[key] = val_summary.get(key, train_summary[key])

    (artifact_root / "train_summary.json").write_text(json.dumps(train_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "model": model,
        "train_summary": train_summary,
        "val_summary": val_summary,
    }

