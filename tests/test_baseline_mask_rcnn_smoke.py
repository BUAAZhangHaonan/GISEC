from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torchvision.models import ResNet50_Weights

from baseline.mask_rcnn.train import _build_mask_rcnn_model, train_mask_rcnn_baseline


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = (60, 80, 120)
        cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", np.full((64, 64), 0.9, dtype=np.float32))
        ann = {
            "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [16, 16, 32, 32],
                    "area": 1024,
                    "iscrowd": 0,
                    "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
                }
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def test_mask_rcnn_rgb_baseline_smoke_exports_shared_artifacts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    train_mask_rcnn_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        score_threshold=0.05,
        variant="rgb_phasea_test",
        backbone_name="resnet50_fpn",
        pretrained_backbone=False,
        amp=False,
        eval_every_epochs=1,
        render_overlay_limit=2,
    )

    assert (output_root / "run_summary.json").exists()
    assert (output_root / "metrics.cocoeval.json").exists()
    assert (output_root / "inference_speed.json").exists()
    assert (output_root / "coco_instances_results.json").exists()
    assert (output_root / "params_trainable.txt").exists()
    assert (output_root / "wall_time_sec.txt").exists()
    assert (output_root / "peak_memory_mb.txt").exists()
    assert (output_root / "model_best.pth").exists()
    assert (output_root / "model_final.pth").exists()
    assert (output_root / "visualizations" / "progress" / "training_curves.png").exists()
    assert list((output_root / "visualizations" / "overlay").glob("*.png"))

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "mask_rcnn"
    assert summary["variant"] == "rgb_phasea_test"
    assert summary["modality"] == "rgb"
    assert summary["benchmark"]["model_family"] == "mask_rcnn"
    assert summary["benchmark"]["backbone_name"] == "resnet50_fpn"
    assert summary["decode_config"]["score_threshold"] == 0.05
    assert "boundary/IoU" in summary["metrics"]


def test_mask_rcnn_rgbd_baseline_smoke_exports_shared_artifacts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out_rgbd"
    _write_dataset(dataset_root)

    train_mask_rcnn_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        score_threshold=0.05,
        variant="rgbd_phaseb_test",
        backbone_name="resnet50_fpn",
        input_mode="rgbd",
        pretrained_backbone=False,
        amp=False,
        eval_every_epochs=1,
        render_overlay_limit=2,
    )

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "mask_rcnn"
    assert summary["variant"] == "rgbd_phaseb_test"
    assert summary["modality"] == "rgbd"
    assert summary["benchmark"]["input_mode"] == "rgbd"
    assert summary["benchmark"]["fusion_mode"] == "rgbd"
    assert "boundary/IoU" in summary["metrics"]


def test_mask_rcnn_builder_switches_to_four_channel_stem_for_rgbd() -> None:
    model = _build_mask_rcnn_model(
        backbone_name="resnet50_fpn",
        pretrained_backbone=False,
        input_channels=4,
    )

    assert int(model.backbone.body.conv1.in_channels) == 4


def test_mask_rcnn_builder_does_not_bypass_backbone_hash_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fake_maskrcnn_resnet50_fpn(*, weights=None, weights_backbone=None):
        calls.append(weights_backbone)
        if weights_backbone is ResNet50_Weights.DEFAULT:
            raise RuntimeError("invalid hash value")
        raise AssertionError("hash failure fallback should not be attempted")

    monkeypatch.setattr("baseline.mask_rcnn.train.maskrcnn_resnet50_fpn", fake_maskrcnn_resnet50_fpn)

    with pytest.raises(RuntimeError, match="invalid hash value"):
        _build_mask_rcnn_model(
            backbone_name="resnet50_fpn",
            pretrained_backbone=True,
        )

    assert calls == [ResNet50_Weights.DEFAULT]
