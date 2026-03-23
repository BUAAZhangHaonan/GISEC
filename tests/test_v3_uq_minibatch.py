from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gisec.datasets.ecc_query_dataset import build_ownership_target as build_legacy_ownership_target
from gisec_v3.train.targets import build_ownership_target as build_v3_ownership_target
from gisec_v3.train.train_uq import run_uq_minibatch
from gisec_v3.train.train_uq import _build_alpha_targets_from_instance_maps, _classify_failure, _compute_alpha_losses


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[12:28, 12:28] = (60, 80, 120)
        image[36:52, 36:52] = (80, 120, 60)
        cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", np.full((64, 64), 0.9, dtype=np.float32))
        ann = {
            "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [12, 12, 16, 16],
                    "area": 256,
                    "iscrowd": 0,
                    "segmentation": [[12, 12, 28, 12, 28, 28, 12, 28]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [36, 36, 16, 16],
                    "area": 256,
                    "iscrowd": 0,
                    "segmentation": [[36, 36, 52, 36, 52, 52, 36, 52]],
                },
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def test_v3_uq_minibatch_runs_single_stage_training_and_eval(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    run_uq_minibatch(
        dataset_root=dataset_root,
        output_dir=output_root,
        model_id="UQ-s",
        device="cpu",
        image_size=64,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        min_area=8,
    )

    assert (output_root / "run_summary.json").exists()
    assert (output_root / "metrics_log.jsonl").exists()
    assert (output_root / "mask_calibration_summary.json").exists()
    assert (output_root / "object_pathology_summary.json").exists()
    assert (output_root / "match_diagnostics_summary.json").exists()
    assert (output_root / "failure_summary.json").exists()
    assert (output_root / "coco_instances_results.json").exists()
    assert (output_root / "metrics.cocoeval.json").exists()

    run_summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["model_id"] == "UQ-s"
    assert run_summary["variant"] == "UQ-s"
    assert run_summary["split_mode"] == "object_first"
    assert run_summary["use_reference"] is False
    assert run_summary["use_graph_rescue"] is False
    assert "metrics" in run_summary
    assert "segm/AP" in run_summary["metrics"]
    assert "bbox/AP" in run_summary["metrics"]

    metric_rows = [json.loads(line) for line in (output_root / "metrics_log.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows = [row for row in metric_rows if row.get("mode") == "train"]
    assert train_rows
    assert "object_count" in train_rows[0]
    assert "split_count" in train_rows[0]
    assert "avg_cores_per_object" in train_rows[0]

    failure_summary = json.loads((output_root / "failure_summary.json").read_text(encoding="utf-8"))
    assert failure_summary["total_images"] == 1
    assert set(failure_summary["counts"]).issuperset({"normal", "empty", "oversized_blob", "severe_under_count", "severe_over_split"})


def test_v3_alpha_targets_are_built_from_v3_semantics_not_legacy_scaled_offsets() -> None:
    instance_maps = torch.zeros((1, 64, 64), dtype=torch.long)
    instance_maps[0, 16:48, 20:44] = 1

    targets = _build_alpha_targets_from_instance_maps(instance_maps)
    expected_v3 = torch.from_numpy(build_v3_ownership_target(instance_maps[0].numpy())).float().unsqueeze(0)
    legacy = torch.from_numpy(build_legacy_ownership_target(instance_maps[0].numpy())).float().unsqueeze(0)

    assert torch.allclose(targets["ownership"], expected_v3)
    assert not torch.allclose(targets["ownership"], legacy)


def test_v3_alpha_losses_reward_better_predictions() -> None:
    instance_maps = torch.zeros((1, 32, 32), dtype=torch.long)
    instance_maps[0, 8:24, 8:24] = 1
    targets = _build_alpha_targets_from_instance_maps(instance_maps)

    good_outputs = {
        "fg_logits": torch.where(targets["fg"] > 0.5, torch.full_like(targets["fg"], 4.0), torch.full_like(targets["fg"], -4.0)),
        "boundary_logits": torch.where(
            targets["boundary"] > 0.5,
            torch.full_like(targets["boundary"], 4.0),
            torch.full_like(targets["boundary"], -4.0),
        ),
        "core_heatmap": torch.where(targets["core"] > 0.1, torch.full_like(targets["core"], 4.0), torch.full_like(targets["core"], -4.0)),
        "ownership_offsets": targets["ownership"].clone(),
    }
    bad_outputs = {
        "fg_logits": -good_outputs["fg_logits"],
        "boundary_logits": -good_outputs["boundary_logits"],
        "core_heatmap": -good_outputs["core_heatmap"],
        "ownership_offsets": torch.zeros_like(targets["ownership"]),
    }

    good_losses = _compute_alpha_losses(good_outputs, targets)
    bad_losses = _compute_alpha_losses(bad_outputs, targets)

    assert good_losses["fg"] < bad_losses["fg"]
    assert good_losses["boundary"] < bad_losses["boundary"]
    assert good_losses["core"] < bad_losses["core"]
    assert good_losses["ownership"] <= bad_losses["ownership"]


def test_v3_alpha_fg_loss_matches_plain_bce_plus_dice_contract() -> None:
    instance_maps = torch.zeros((1, 32, 32), dtype=torch.long)
    instance_maps[0, 8:24, 8:24] = 1
    targets = _build_alpha_targets_from_instance_maps(instance_maps)
    outputs = {
        "fg_logits": torch.where(targets["fg"] > 0.5, torch.full_like(targets["fg"], 1.5), torch.full_like(targets["fg"], -0.75)),
        "boundary_logits": torch.zeros_like(targets["boundary"]),
        "core_heatmap": torch.zeros_like(targets["core"]),
        "ownership_offsets": torch.zeros_like(targets["ownership"]),
    }

    losses = _compute_alpha_losses(outputs, targets)
    expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(outputs["fg_logits"], targets["fg"])
    probs = torch.sigmoid(outputs["fg_logits"])
    intersection = (probs * targets["fg"]).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets["fg"].sum(dim=(1, 2, 3))
    expected_dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()

    assert torch.isclose(losses["fg"], expected_bce + expected_dice, atol=1e-6)


def test_v3_failure_classifier_keeps_oversized_blob_separate_from_under_count() -> None:
    gt_map = torch.zeros((16, 16), dtype=torch.long)
    gt_map[2:6, 2:6] = 1
    gt_map[10:14, 10:14] = 2
    pred_map = torch.zeros((16, 16), dtype=torch.long)
    pred_map[1:15, 1:15] = 1

    assert _classify_failure(gt_map, pred_map) == "oversized_blob"
