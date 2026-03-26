from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.reference_splitter.dataset import ReferenceSplitCacheDataset, collate_reference_splitter_batch
from baseline.reference_splitter.model import ReferenceLocalSplitter, build_query_depth_features


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


def _write_split_cache(root: Path, *, split: str = "train", part_key: str = "partA") -> Path:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    sample_path = split_dir / "000001_0000.npz"
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
    (split_dir / "manifest.json").write_text(
        json.dumps({"split": split, "num_samples": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (split_dir / "metadata.jsonl").write_text(
        json.dumps(
            {
                "image_id": 1,
                "file_name": "partA_scene_0001.png",
                "sample_index": 0,
                "instance_count": 2,
                "part_key": part_key,
                "path": str(sample_path),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return split_dir


def test_build_query_depth_features_returns_expected_channels() -> None:
    depth = torch.tensor([[[[0.2, 0.3], [0.7, 0.9]]]], dtype=torch.float32)
    blob_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32)

    features = build_query_depth_features(depth, blob_mask)

    assert features.shape == (1, 4, 2, 2)
    assert torch.isfinite(features).all()


def test_reference_split_cache_dataset_loads_query_and_reference_tensors(tmp_path: Path) -> None:
    cache_root = tmp_path / "split_cache"
    reference_root = tmp_path / "references"
    _write_split_cache(cache_root)
    _write_reference_root(reference_root)

    dataset = ReferenceSplitCacheDataset(
        cache_root=str(cache_root),
        reference_root=str(reference_root),
        split="train",
        roi_size=32,
        reference_image_size=32,
        slot_count=6,
    )
    sample = dataset[0]

    assert sample["query_rgb"].shape == (3, 32, 32)
    assert sample["query_depth"].shape == (1, 32, 32)
    assert sample["blob_mask"].shape == (1, 32, 32)
    assert sample["center_heatmap"].shape == (1, 32, 32)
    assert sample["reference_rgb"].shape[0] == 2
    assert sample["reference_depth"].shape[0] == 2
    assert sample["reference_mask"].shape[0] == 2
    assert sample["instance_count"].item() == 2
    assert sample["part_key"] == "partA"


def test_reference_splitter_forward_returns_expected_heads(tmp_path: Path) -> None:
    cache_root = tmp_path / "split_cache"
    reference_root = tmp_path / "references"
    _write_split_cache(cache_root)
    _write_reference_root(reference_root, num_views=3)
    dataset = ReferenceSplitCacheDataset(
        cache_root=str(cache_root),
        reference_root=str(reference_root),
        split="train",
        roi_size=32,
        reference_image_size=32,
        slot_count=6,
    )
    batch = collate_reference_splitter_batch([dataset[0]])
    model = ReferenceLocalSplitter(base_channels=16, max_count=4, reference_skip_margin=0.15)

    outputs = model(
        query_rgb=batch["query_rgb"],
        query_depth=batch["query_depth"],
        blob_mask=batch["blob_mask"],
        reference_rgb=batch["reference_rgb"],
        reference_depth=batch["reference_depth"],
        reference_mask=batch["reference_mask"],
        reference_view_ids=batch["reference_view_ids"],
    )

    assert outputs["single_object_logit"].shape == (1, 1)
    assert outputs["count_logits"].shape == (1, 4)
    assert outputs["center_heatmap"].shape == (1, 1, 32, 32)
    assert len(outputs["reference_routing"]) == 1
    assert outputs["reference_routing"][0]["reference_routing_mode"] == "hard_top1"
    assert len(outputs["reference_routing"][0]["selected_view_ids"]) == 1
