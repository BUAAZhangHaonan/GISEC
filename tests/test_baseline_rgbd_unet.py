from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.common.dataset import BaselineInstanceDataset
from baseline.rgbd.fusion import prepare_unet_inputs
from baseline.unet.eval import decode_instance_predictions
from baseline.unet.model import build_unet_family_model
from baseline.unet.train import train_unet_baseline


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    yy, xx = np.indices((64, 64), dtype=np.float32)
    depth = 0.2 + 0.5 * (xx / 63.0) + 0.1 * (yy / 63.0)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[12:52, 10:44] = (60, 80, 120)
        cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", depth.astype(np.float32))
        ann = {
            "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 12, 34, 40],
                    "area": 1360,
                    "iscrowd": 0,
                    "segmentation": [[10, 12, 44, 12, 44, 52, 10, 52]],
                }
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def test_rgbd_fusion_prepares_expected_channel_counts() -> None:
    sample = {
        "image": torch.zeros((3, 64, 64), dtype=torch.float32),
        "depth": torch.linspace(0.1, 1.0, steps=64 * 64, dtype=torch.float32).reshape(1, 64, 64),
    }

    rgbd = prepare_unet_inputs(sample, input_mode="rgbd")
    depth_geometry = prepare_unet_inputs(sample, input_mode="depth_geometry")
    depth_geometry_dense = prepare_unet_inputs(sample, input_mode="depth_geometry_dense")

    assert tuple(rgbd.shape) == (4, 64, 64)
    assert tuple(depth_geometry.shape) == (6, 64, 64)
    assert tuple(depth_geometry_dense.shape) == (8, 64, 64)
    assert not torch.allclose(depth_geometry[3], depth_geometry[4])
    assert not torch.allclose(depth_geometry_dense[4], depth_geometry_dense[5])


def test_unet_family_models_accept_rgbd_channel_counts() -> None:
    for name, channels in [("unet", 4), ("unetpp", 6), ("attention_unet", 8)]:
        model = build_unet_family_model(
            name,
            in_channels=channels,
            encoder_name="resnet34",
            pretrained_backbone=False,
            decoder_channels=64,
        )
        outputs = model(torch.randn(1, channels, 64, 64))
        assert outputs["fg_logits"].shape == (1, 1, 64, 64)
        assert outputs["center_heatmap"].shape == (1, 1, 64, 64)
        assert outputs["offsets"].shape == (1, 2, 64, 64)
        assert outputs["boundary_logits"].shape == (1, 1, 64, 64)


def test_depth_geometry_unet_smoke_exports_rgbd_summary(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    train_unet_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        threshold=0.5,
        model_name="unet",
        input_mode="depth_geometry",
        encoder_name="resnet34",
        pretrained_backbone=False,
        task_mode="semantic_smoke",
    )

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "unet"
    assert summary["variant"] == "depth_geometry_smoke"
    assert summary["modality"] == "rgbd"


def test_depth_geometry_dense_cache_script_and_dataset_loading(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/precompute_baseline_depth_cache.py",
            "--dataset-root",
            str(dataset_root),
            "--image-size",
            "64",
            "--feature-mode",
            "depth_geometry_dense",
            "--split",
            "train",
            "--workers",
            "1",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=64,
        include_depth=True,
        depth_feature_mode="depth_geometry_dense",
    )
    sample = dataset[0]

    assert sample["depth_features"] is not None
    assert tuple(sample["depth_features"].shape) == (5, 64, 64)


def test_depth_geometry_dense_unet_smoke_uses_cached_features(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/precompute_baseline_depth_cache.py",
            "--dataset-root",
            str(dataset_root),
            "--image-size",
            "64",
            "--feature-mode",
            "depth_geometry_dense",
            "--split",
            "train",
            "--split",
            "val",
            "--workers",
            "1",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    train_unet_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        threshold=0.5,
        model_name="unet",
        input_mode="depth_geometry_dense",
        encoder_name="resnet34",
        pretrained_backbone=False,
        task_mode="semantic_smoke",
    )

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["variant"] == "depth_geometry_dense_smoke"
    assert summary["modality"] == "rgbd"


def test_instance_decoder_separates_two_components_from_centers_and_offsets() -> None:
    fg_logits = torch.full((1, 1, 32, 32), -8.0)
    fg_logits[:, :, 8:24, 6:26] = 8.0
    center_heatmap = torch.full((1, 1, 32, 32), -8.0)
    center_heatmap[:, :, 16, 10] = 8.0
    center_heatmap[:, :, 16, 21] = 8.0
    boundary_logits = torch.full((1, 1, 32, 32), -8.0)
    boundary_logits[:, :, 8:24, 15:17] = 8.0
    offsets = torch.zeros((1, 2, 32, 32), dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    offsets[0, 0, 8:24, 6:16] = 10.0 - xx[8:24, 6:16]
    offsets[0, 1, 8:24, 6:16] = 16.0 - yy[8:24, 6:16]
    offsets[0, 0, 8:24, 16:26] = 21.0 - xx[8:24, 16:26]
    offsets[0, 1, 8:24, 16:26] = 16.0 - yy[8:24, 16:26]

    label_map, stats = decode_instance_predictions(
        fg_logits=fg_logits[0],
        center_heatmap=center_heatmap[0],
        offsets=offsets[0],
        boundary_logits=boundary_logits[0],
        fg_threshold=0.5,
        center_threshold=0.5,
        min_area=8,
    )

    labels = sorted(int(x) for x in torch.unique(label_map).tolist() if int(x) > 0)
    assert labels == [1, 2]
    assert stats["num_instances"] == 2.0
