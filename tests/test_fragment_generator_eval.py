from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _write_fragment_cache(root: Path, *, split: str = "val", count: int = 2) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_index in range(count):
        sample_path = split_dir / f"000001_{sample_index:04d}.npz"
        rgb_crop = np.zeros((3, 32, 32), dtype=np.float32)
        rgb_crop[:, 6:26, 4:14] = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)[:, None, None]
        rgb_crop[:, 6:26, 18:28] = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)[:, None, None]
        coarse_mask_logit_crop = np.full((1, 32, 32), -8.0, dtype=np.float32)
        coarse_mask_logit_crop[:, 4:28, 2:30] = 8.0
        pixel_feature_crop = np.zeros((4, 32, 32), dtype=np.float32)
        pixel_feature_crop[0, 6:26, 4:14] = 1.0
        pixel_feature_crop[1, 6:26, 18:28] = 1.0
        gt_union = np.zeros((1, 32, 32), dtype=np.uint8)
        gt_union[:, 6:26, 4:14] = 1
        gt_union[:, 6:26, 18:28] = 1
        gt_fragments = np.zeros((6, 32, 32), dtype=np.uint8)
        gt_fragments[0, 6:26, 4:14] = 1
        gt_fragments[1, 6:26, 18:28] = 1
        gt_fragment_owner_ids = np.asarray([1, 2, 0, 0, 0, 0], dtype=np.int32)
        with sample_path.open("wb") as handle:
            np.savez(
                handle,
                rgb_crop=rgb_crop,
                coarse_mask_logit_crop=coarse_mask_logit_crop,
                pixel_feature_crop=pixel_feature_crop,
                coarse_score=np.asarray(0.9, dtype=np.float32),
                crop_bbox=np.asarray([2, 4, 28, 24], dtype=np.int32),
                image_id=np.asarray(1, dtype=np.int32),
                pred_id=np.asarray(sample_index, dtype=np.int32),
                image_shape=np.asarray([64, 64], dtype=np.int32),
                gt_instance_union_mask=gt_union,
                gt_fragment_masks=gt_fragments,
                gt_fragment_owner_ids=gt_fragment_owner_ids,
                has_gt_overlap=np.asarray(1, dtype=np.uint8),
                overflow_crop=np.asarray(0, dtype=np.uint8),
            )
        rows.append({"path": str(sample_path), "pred_id": sample_index})
    (split_dir / "manifest.json").write_text(json.dumps({"num_samples": count}, ensure_ascii=False), encoding="utf-8")
    (split_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class _PerfectFragmentModel(torch.nn.Module):
    def forward(
        self,
        *,
        rgb_crop: torch.Tensor,
        coarse_mask_logit_crop: torch.Tensor,
        pixel_feature_crop: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = int(rgb_crop.shape[0])
        logits = torch.full((batch, 6, 32, 32), -8.0, dtype=rgb_crop.dtype, device=rgb_crop.device)
        logits[:, 0, 6:26, 4:14] = 8.0
        logits[:, 1, 6:26, 18:28] = 8.0
        presence = torch.full((batch, 6), -8.0, dtype=rgb_crop.dtype, device=rgb_crop.device)
        presence[:, :2] = 8.0
        crop_features = torch.zeros((batch, 8, 32, 32), dtype=rgb_crop.dtype, device=rgb_crop.device)
        crop_features[:, 0, 6:26, 4:14] = 1.0
        crop_features[:, 1, 6:26, 18:28] = 1.0
        embeddings = torch.zeros((batch, 6, 8), dtype=rgb_crop.dtype, device=rgb_crop.device)
        embeddings[:, 0, 0] = 1.0
        embeddings[:, 1, 1] = 1.0
        return {
            "fragment_mask_logits": logits,
            "fragment_presence_logits": presence,
            "crop_features": crop_features,
            "fragment_embeddings": embeddings,
        }


def test_evaluate_fragment_generator_writes_metrics_and_prediction_exports(tmp_path: Path) -> None:
    from baseline.fragment_generator.eval import evaluate_fragment_generator

    cache_root = tmp_path / "fragment_cache"
    output_root = tmp_path / "eval"
    _write_fragment_cache(cache_root, split="val", count=2)

    summary = evaluate_fragment_generator(
        cache_root=str(cache_root),
        output_dir=str(output_root),
        split="val",
        device=torch.device("cpu"),
        model=_PerfectFragmentModel(),
        batch_size=1,
        num_workers=0,
        export_predictions=True,
    )

    assert summary["covered_gt_rate"] == 1.0
    assert summary["split_gt_rate"] == 0.0
    assert summary["singleton_gt_rate"] == 1.0
    assert summary["impure_fragment_rate"] == 0.0
    assert summary["gate_passed"] is False
    assert "loss_single" not in summary
    exports = sorted((output_root / "fragment_predictions").glob("*.npz"))
    assert len(exports) == 2
    payload = np.load(exports[0], allow_pickle=False)
    assert set(payload.files) >= {
        "fragment_mask_probs",
        "fragment_mask_binaries",
        "fragment_presence_scores",
        "fragment_embeddings",
        "crop_bbox",
        "image_shape",
        "gt_fragment_masks",
        "gt_fragment_owner_ids",
    }
