from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.reference_splitter.train import train_reference_splitter_alpha


def _write_reference_root(root: Path, *, part_key: str = "partA", num_views: int = 2) -> None:
    bank = root / part_key
    for name in ["rgb", "depth", "mask", "meta"]:
        (bank / name).mkdir(parents=True, exist_ok=True)
    for index in range(num_views):
        rgb = np.zeros((24, 24, 3), dtype=np.uint8)
        rgb[4:20, 4:20] = (60 + index * 20, 80, 120)
        cv2.imwrite(str(bank / "rgb" / f"view_{index:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(bank / "depth" / f"view_{index:03d}.npy", np.full((24, 24), 0.7 + 0.1 * index, dtype=np.float32))
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[4:20, 4:20] = 255
        cv2.imwrite(str(bank / "mask" / f"view_{index:03d}.png"), mask)


def _write_split_cache(root: Path, *, split: str = "train", part_key: str = "partA", count: int = 3) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_index in range(count):
        sample_path = split_dir / f"000001_{sample_index:04d}.npz"
        rgb = np.zeros((3, 32, 40), dtype=np.uint8)
        rgb[:, 8:24, 6:18] = np.array([80, 90, 120], dtype=np.uint8)[:, None, None]
        rgb[:, 8:24, 22:34] = np.array([80, 90, 120], dtype=np.uint8)[:, None, None]
        depth = np.ones((1, 32, 40), dtype=np.float32)
        depth[:, :, 20:] = 1.4
        blob_mask = np.zeros((1, 32, 40), dtype=np.uint8)
        blob_mask[:, 8:24, 6:34] = 1
        center_heatmap = np.zeros((1, 32, 40), dtype=np.float32)
        center_heatmap[:, 16, 12] = 1.0
        center_heatmap[:, 16, 28] = 1.0
        with sample_path.open("wb") as handle:
            np.savez(
                handle,
                rgb=rgb,
                depth=depth,
                blob_mask=blob_mask,
                center_heatmap=center_heatmap,
                instance_count=np.asarray(2, dtype=np.int32),
                part_key=np.asarray(part_key),
            )
        rows.append(
            {
                "image_id": 1,
                "file_name": "partA_scene_0001.png",
                "sample_index": sample_index,
                "instance_count": 2,
                "part_key": part_key,
                "path": str(sample_path),
            }
        )
    (split_dir / "manifest.json").write_text(
        json.dumps({"split": split, "num_samples": count}, ensure_ascii=False),
        encoding="utf-8",
    )
    (split_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_train_reference_splitter_alpha_writes_summary_and_checkpoint(tmp_path: Path) -> None:
    cache_root = tmp_path / "split_cache"
    reference_root = tmp_path / "references"
    output_root = tmp_path / "out"
    _write_split_cache(cache_root)
    _write_reference_root(reference_root)

    train_reference_splitter_alpha(
        cache_root=str(cache_root),
        reference_root=str(reference_root),
        output_dir=str(output_root),
        split="train",
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        roi_size=32,
        reference_image_size=32,
        max_train_steps=2,
    )

    summary = json.loads((output_root / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs"] == 1
    assert summary["steps"] == 2
    assert summary["loss_total"] >= 0.0
    assert (output_root / "model_final.pth").exists()
