from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Mask2FormerImageProcessor

from baseline.mask2former.adapter import sample_to_mask2former_inputs
from baseline.mask2former.train import train_mask2former_baseline


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[10:54, 10:30] = (40, 100, 180)
        image[20:52, 34:56] = (180, 80, 40)
        cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", np.full((64, 64), 0.9, dtype=np.float32))
        ann = {
            "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 44],
                    "area": 880,
                    "iscrowd": 0,
                    "segmentation": [[10, 10, 30, 10, 30, 54, 10, 54]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [34, 20, 22, 32],
                    "area": 704,
                    "iscrowd": 0,
                    "segmentation": [[34, 20, 56, 20, 56, 52, 34, 52]],
                },
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def test_mask2former_adapter_encodes_instance_targets() -> None:
    sample = {
        "image_id": 1,
        "file_name": "000001.png",
        "image": torch.zeros((3, 64, 64), dtype=torch.float32),
        "masks": torch.from_numpy(
            np.stack(
                [
                    np.pad(np.ones((20, 16), dtype=np.uint8), ((8, 36), (8, 40))),
                    np.pad(np.ones((16, 20), dtype=np.uint8), ((24, 24), (28, 16))),
                ],
                axis=0,
            )
        ),
        "boxes": torch.tensor([[8.0, 8.0, 16.0, 20.0], [28.0, 24.0, 20.0, 16.0]], dtype=torch.float32),
        "labels": torch.tensor([1, 1], dtype=torch.int64),
    }
    processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)

    encoded = sample_to_mask2former_inputs(sample, processor=processor)

    assert tuple(encoded["pixel_values"].shape) == (1, 3, 64, 64)
    assert tuple(encoded["pixel_mask"].shape) == (1, 64, 64)
    assert len(encoded["mask_labels"]) == 1
    assert len(encoded["class_labels"]) == 1
    assert tuple(encoded["mask_labels"][0].shape) == (2, 64, 64)
    assert encoded["class_labels"][0].tolist() == [1, 1]


def test_mask2former_rgb_baseline_smoke_exports_shared_artifacts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    train_mask2former_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        score_threshold=0.0,
        mask_threshold=0.5,
        pretrained_model_name=None,
        hidden_dim=32,
        feature_size=32,
        mask_feature_size=32,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=4,
        num_queries=8,
        train_num_points=256,
    )

    assert (output_root / "run_summary.json").exists()
    assert (output_root / "metrics.cocoeval.json").exists()
    assert (output_root / "inference_speed.json").exists()
    assert (output_root / "coco_instances_results.json").exists()
    assert (output_root / "params_trainable.txt").exists()
    assert (output_root / "wall_time_sec.txt").exists()
    assert (output_root / "peak_memory_mb.txt").exists()
    assert list((output_root / "visualizations" / "overlay").glob("*.png"))

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "mask2former"
    assert summary["variant"] == "rgb_smoke"
    assert summary["modality"] == "rgb"
