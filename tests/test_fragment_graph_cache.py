from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from baseline.common.fragment_graph_cache import build_fragment_graph_cache, summarize_fragment_graph_sample
from gisec.models.graph_utils import GraphBatch


def _write_dataset(root: Path, *, split: str = "train", file_name: str = "partA_scene_0001.png") -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "depth" / split).mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 12:52] = (80, 120, 160)
    cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    depth = np.ones((64, 64), dtype=np.float32)
    depth[:, :32] = 1.0
    depth[:, 32:] = 1.1
    np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", depth)
    ann = {
        "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [12, 16, 40, 32],
                "area": 1280,
                "iscrowd": 0,
                "segmentation": [[12, 16, 52, 16, 52, 48, 12, 48]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


class _DummySplitFirstModel(nn.Module):
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _channels, height, width = image.shape
        fg_logits = torch.full((batch, 1, height, width), -8.0, dtype=image.dtype, device=image.device)
        fg_logits[:, :, 16:48, 12:52] = 8.0
        center_heatmap = torch.full((batch, 1, height, width), -8.0, dtype=image.dtype, device=image.device)
        center_heatmap[:, :, 32, 22] = 8.0
        center_heatmap[:, :, 32, 42] = 8.0
        boundary_logits = torch.full((batch, 1, height, width), -8.0, dtype=image.dtype, device=image.device)
        boundary_logits[:, :, 16:48, 32] = 8.0
        offsets = torch.zeros((batch, 2, height, width), dtype=image.dtype, device=image.device)
        feature_map = torch.zeros((batch, 8, height, width), dtype=image.dtype, device=image.device)
        feature_map[:, 0, 16:48, 12:32] = 1.0
        feature_map[:, 1, 16:48, 32:52] = 1.0
        return {
            "fg_logits": fg_logits,
            "center_heatmap": center_heatmap,
            "boundary_logits": boundary_logits,
            "offsets": offsets,
            "feature_map": feature_map,
        }


def test_summarize_fragment_graph_sample_reports_purity_and_same_instance_recall() -> None:
    graph_batch = GraphBatch(
        node_features=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_features=torch.zeros((2, 6), dtype=torch.float32),
        edge_targets=torch.tensor([1.0, 0.0], dtype=torch.float32),
        fragments=np.zeros((8, 8), dtype=np.int32),
        diagnostics={"num_fragments": 3, "num_edges": 2},
        edge_ignore_mask=torch.tensor([False, False], dtype=torch.bool),
        fragment_stats=[
            {"gt_instance": 1, "purity": 1.0, "area_ratio": 0.02},
            {"gt_instance": 1, "purity": 0.75, "area_ratio": 0.03},
            {"gt_instance": 2, "purity": 0.9, "area_ratio": 0.01},
        ],
    )

    summary = summarize_fragment_graph_sample(graph_batch)

    assert summary["num_fragments"] == 3
    assert summary["fragment_purity_mean"] == 0.8833333333333333
    assert summary["same_instance_pairs_total"] == 1
    assert summary["same_instance_pairs_covered"] == 1
    assert summary["same_instance_recall"] == 1.0
    assert summary["positive_edge_ratio"] == 0.5


def test_build_fragment_graph_cache_exports_graph_ready_samples(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "cache"
    reference_root = tmp_path / "references"
    (reference_root / "partA").mkdir(parents=True, exist_ok=True)
    _write_dataset(dataset_root)

    manifest = build_fragment_graph_cache(
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        split="train",
        image_size=64,
        device=torch.device("cpu"),
        model=_DummySplitFirstModel(),
        model_name="unet",
        input_mode="rgb",
        encoder_name="resnet34",
        decoder_channels=64,
        fg_threshold=0.18,
        center_threshold=0.03,
        min_area=8,
        boundary_threshold=0.5,
        variant="B0",
        reference_root=str(reference_root),
        max_images=1,
    )

    assert manifest["num_samples"] == 1
    assert manifest["same_instance_recall"] == 1.0
    assert manifest["fragment_purity_mean"] == 1.0
    assert manifest["avg_fragments"] == 2.0

    sample_path = output_root / "train" / "000001.pt"
    assert sample_path.exists()
    payload = torch.load(sample_path, map_location="cpu")
    assert payload["part_key"] == "partA"
    assert payload["summary"]["num_fragments"] == 2
    assert payload["summary"]["same_instance_recall"] == 1.0
    assert payload["edge_targets"].tolist() == [1.0]


def test_build_fragment_graph_cache_forwards_graph_purity_threshold(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "cache"
    reference_root = tmp_path / "references"
    (reference_root / "partA").mkdir(parents=True, exist_ok=True)
    _write_dataset(dataset_root)

    captured: dict[str, float] = {}

    def _fake_build_graph_batch_from_fragments(**kwargs):
        captured["purity_threshold"] = float(kwargs["purity_threshold"])
        return GraphBatch(
            node_features=torch.zeros((2, 4), dtype=torch.float32),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_features=torch.zeros((1, 6), dtype=torch.float32),
            edge_targets=torch.tensor([1.0], dtype=torch.float32),
            fragments=np.zeros((64, 64), dtype=np.int32),
            diagnostics={"num_fragments": 2, "num_edges": 1},
            edge_ignore_mask=torch.tensor([False], dtype=torch.bool),
            fragment_stats=[
                {"gt_instance": 1, "purity": 1.0, "area_ratio": 0.02},
                {"gt_instance": 1, "purity": 1.0, "area_ratio": 0.02},
            ],
            shape_stats={},
        )

    monkeypatch.setattr("baseline.common.fragment_graph_cache.build_graph_batch_from_fragments", _fake_build_graph_batch_from_fragments)

    manifest = build_fragment_graph_cache(
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        split="train",
        image_size=64,
        device=torch.device("cpu"),
        model=_DummySplitFirstModel(),
        model_name="unet",
        input_mode="rgb",
        encoder_name="resnet34",
        decoder_channels=64,
        fg_threshold=0.18,
        center_threshold=0.03,
        min_area=8,
        boundary_threshold=0.5,
        purity_threshold=0.6,
        variant="B0",
        reference_root=str(reference_root),
        max_images=1,
    )

    assert captured["purity_threshold"] == 0.6
    assert manifest["purity_threshold"] == 0.6
