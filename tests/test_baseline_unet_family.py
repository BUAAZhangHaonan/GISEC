from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.unet.model import build_unet_family_model
from baseline.unet.train import train_unet_baseline


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


def test_unet_family_model_factory_builds_expected_variants() -> None:
    for name in ["unet", "unetpp", "attention_unet"]:
        model = build_unet_family_model(name, in_channels=3, base_channels=8)
        logits = model(torch.randn(1, 3, 64, 64))
        assert logits.shape == (1, 1, 64, 64)


def test_attention_unet_smoke_uses_shared_export_contract(tmp_path: Path) -> None:
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
        model_name="attention_unet",
    )

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "attention_unet"
    assert summary["variant"] == "rgb_smoke"
    assert summary["modality"] == "rgb"
