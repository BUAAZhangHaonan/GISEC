from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.unet.model import build_unet_family_model
from baseline.unet.export import export_unet_fragment_cache


def _write_dataset(root: Path) -> None:
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = (180, 120, 60)
        cv2.imwrite(str(root / "images" / split / "000001.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        ann = {
            "images": [{"id": 1, "file_name": "000001.png", "width": 32, "height": 32}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [8, 8, 16, 16],
                    "area": 256,
                    "iscrowd": 0,
                    "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
                }
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


class _ToySplitModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _channels, height, width = x.shape
        fg = torch.full((batch_size, 1, height, width), -8.0, dtype=x.dtype, device=x.device)
        fg[:, :, 8:24, 8:24] = 8.0
        center = torch.full_like(fg, -8.0)
        center[:, :, 15, 12] = 8.0
        center[:, :, 15, 19] = 8.0
        boundary = torch.full_like(fg, -8.0)
        offsets = torch.zeros((batch_size, 2, height, width), dtype=x.dtype, device=x.device)
        return {
            "fg_logits": fg,
            "center_heatmap": center,
            "boundary_logits": boundary,
            "offsets": offsets,
        }


def test_export_unet_fragment_cache_writes_fragment_artifacts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "fragment_cache"
    _write_dataset(dataset_root)

    export_unet_fragment_cache(
        model=_ToySplitModel(),
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=32,
        device=torch.device("cpu"),
        split="train",
        input_mode="rgb",
        threshold=0.5,
        center_threshold=0.5,
        min_area=4,
        watershed_enabled=True,
        use_depth_split_walls=False,
        depth_wall_threshold=0.1,
    )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "fragment_quality_summary.json").read_text(encoding="utf-8"))
    fragment_cache = np.load(output_root / "fragments" / "000001_000001.npz")

    assert manifest["split"] == "train"
    assert manifest["num_images"] == 1
    assert summary["fragment_count"] == 2
    assert summary["same_instance_recall"] == 1.0
    assert tuple(fragment_cache["label_map"].shape) == (32, 32)
    assert tuple(fragment_cache["offsets"].shape) == (2, 32, 32)
    assert tuple(fragment_cache["boundary_prob"].shape) == (32, 32)


def test_export_unet_fragment_cache_script_builds_offline_cache(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    output_root = tmp_path / "offline_cache"
    _write_dataset(dataset_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "unet_rgb_fragment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "common:",
                "  image_size: 32",
                "train:",
                "  epochs: 1",
                "model:",
                "  model_name: unet",
                "  input_mode: rgb",
                "  encoder_name: resnet18",
                "  pretrained_backbone: false",
                "  task_mode: instance",
                "  threshold: 0.18",
                "  center_threshold: 0.03",
                "  fragment_min_area: 4",
                "  watershed_enabled: true",
                "  use_depth_split_walls: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    model = build_unet_family_model(
        "unet",
        in_channels=3,
        encoder_name="resnet18",
        pretrained_backbone=False,
        decoder_channels=16,
    )
    checkpoint_path = run_dir / "model_best.pth"
    torch.save(model.state_dict(), checkpoint_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/export_unet_fragment_cache.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_root),
            "--split",
            "train",
            "--device",
            "cpu",
            "--decoder-channels",
            "16",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "train"
    assert manifest["input_mode"] == "rgb"
    assert (output_root / "fragment_quality_summary.json").exists()
