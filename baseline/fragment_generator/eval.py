from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.fragment_generator.dataset import FragmentGeneratorCacheDataset, collate_fragment_generator_batch
from baseline.fragment_generator.metrics import aggregate_fragment_quality_metrics, compute_fragment_quality_metrics


def _gate_fragment_metrics(summary: dict[str, float]) -> bool:
    return bool(
        float(summary["covered_gt_rate"]) >= 0.92
        and float(summary["split_gt_rate"]) >= 0.30
        and float(summary["impure_fragment_rate"]) <= 0.10
        and float(summary["leakage_rate"]) <= 0.05
        and float(summary["fragments_per_covered_gt"]) >= 1.5
        and float(summary["singleton_gt_rate"]) <= 0.70
        and float(summary["overflow_crop_rate"]) <= 0.05
    )


def evaluate_fragment_generator(
    *,
    cache_root: str,
    output_dir: str,
    split: str,
    device: torch.device,
    model: torch.nn.Module,
    batch_size: int = 1,
    num_workers: int = 0,
    export_predictions: bool = True,
) -> dict[str, Any]:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    export_root = artifact_root / "fragment_predictions"
    if export_predictions:
        export_root.mkdir(parents=True, exist_ok=True)
    dataset = FragmentGeneratorCacheDataset(cache_root=cache_root, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_fragment_generator_batch,
    )
    model = model.to(device)
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
            if export_predictions:
                mask_probs = torch.sigmoid(outputs["fragment_mask_logits"]).detach().cpu().numpy()
                mask_binaries = (mask_probs >= 0.5).astype(np.uint8)
                presence_scores = torch.sigmoid(outputs["fragment_presence_logits"]).detach().cpu().numpy()
                fragment_embeddings = outputs["fragment_embeddings"].detach().cpu().numpy()
                for row_index, sample_path in enumerate(batch["sample_path"]):
                    export_path = export_root / f"{Path(str(sample_path)).stem}.npz"
                    with export_path.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            fragment_mask_probs=mask_probs[row_index].astype(np.float16),
                            fragment_mask_binaries=mask_binaries[row_index].astype(np.uint8),
                            fragment_presence_scores=presence_scores[row_index].astype(np.float16),
                            fragment_embeddings=fragment_embeddings[row_index].astype(np.float16),
                            crop_bbox=batch["crop_bbox"][row_index].detach().cpu().numpy().astype(np.int32),
                            image_shape=batch["image_shape"][row_index].detach().cpu().numpy().astype(np.int32),
                            image_id=np.asarray(int(batch["image_id"][row_index].item()), dtype=np.int32),
                            pred_id=np.asarray(int(batch["pred_id"][row_index].item()), dtype=np.int32),
                            gt_fragment_masks=batch["gt_fragment_masks"][row_index].detach().cpu().numpy().astype(np.uint8),
                            gt_fragment_owner_ids=batch["gt_fragment_owner_ids"][row_index].detach().cpu().numpy().astype(np.int32),
                            gt_instance_union_mask=batch["gt_instance_union_mask"][row_index].detach().cpu().numpy().astype(np.uint8),
                            overflow_crop=np.asarray(int(batch["overflow_crop"][row_index].item()), dtype=np.uint8),
                        )

    summary = aggregate_fragment_quality_metrics(metric_rows)
    summary["gate_passed"] = bool(_gate_fragment_metrics(summary))
    (artifact_root / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
