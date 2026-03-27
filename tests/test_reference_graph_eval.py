from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

from baseline.reference_graph.dataset import FragmentGraphMergeDataset
from baseline.reference_graph.eval_pipeline import evaluate_reference_graph_merge
from baseline.reference_graph.eval_pipeline import render_reference_graph_preview_sheet
from baseline.reference_graph.model import ReferenceGraphMergeModel


def _write_dataset(
    root: Path,
    *,
    split: str = "val",
    file_name: str = "partA_scene_0001.png",
    extra_uncovered_image: bool = False,
) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = (80, 120, 160)
    cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    images = [{"id": 1, "file_name": file_name, "width": 32, "height": 32}]
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [8, 8, 16, 16],
            "area": 256,
            "iscrowd": 0,
            "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
        }
    ]
    if extra_uncovered_image:
        extra_name = "partA_scene_0002.png"
        cv2.imwrite(str(root / "images" / split / extra_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        images.append({"id": 2, "file_name": extra_name, "width": 32, "height": 32})
        annotations.append(
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "area": 256,
                "iscrowd": 0,
                "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
            }
        )
    ann = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def _write_reference_root(root: Path, *, part_key: str = "partA") -> None:
    bank = root / part_key
    for name in ["rgb", "depth", "mask", "meta"]:
        (bank / name).mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[4:20, 4:20] = (60, 80, 120)
    cv2.imwrite(str(bank / "rgb" / "view_000.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(bank / "depth" / "view_000.npy", np.full((24, 24), 0.7, dtype=np.float32))
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 4:20] = 255
    cv2.imwrite(str(bank / "mask" / "view_000.png"), mask)


def _write_graph_cache(root: Path, *, split: str = "val") -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    fragments = torch.zeros((32, 32), dtype=torch.int16)
    fragments[8:24, 8:16] = 1
    fragments[8:24, 16:24] = 2
    payload = {
        "image_id": 1,
        "file_name": "partA_scene_0001.png",
        "part_key": "partA",
        "fragments": fragments,
        "node_features": torch.tensor(
            [
                [1.0, 0.0, 0.2, 0.1],
                [0.9, 0.1, 0.2, 0.1],
            ],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_features": torch.tensor([[0.1, 0.8, 0.05, 0.02, 0.01, 0.9]], dtype=torch.float32),
        "edge_targets": torch.tensor([1.0], dtype=torch.float32),
        "edge_ignore_mask": torch.tensor([False], dtype=torch.bool),
        "fragment_stats": [
            {"gt_instance": 1, "purity": 1.0, "area_ratio": 0.08, "bbox": (1, 1, 2, 4)},
            {"gt_instance": 1, "purity": 1.0, "area_ratio": 0.08, "bbox": (3, 1, 2, 4)},
        ],
        "diagnostics": {"num_fragments": 2, "num_edges": 1},
        "shape_stats": {"mean_area_ratio": 0.16, "mean_aspect_ratio": 1.0},
        "summary": {"same_instance_recall": 1.0, "fragment_purity_mean": 1.0},
    }
    torch.save(payload, split_dir / "000001.pt")
    (split_dir / "manifest.json").write_text(json.dumps({"split": split, "num_samples": 1}), encoding="utf-8")


class _DummyMergeModel(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.full((batch["edge_features"].shape[0],), 5.0, dtype=torch.float32)


def test_evaluate_reference_graph_merge_writes_metrics_and_results(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    reference_root = tmp_path / "refs"
    output_root = tmp_path / "eval"
    _write_dataset(dataset_root)
    _write_graph_cache(cache_root)
    _write_reference_root(reference_root)

    metrics, summary = evaluate_reference_graph_merge(
        cache_root=str(cache_root),
        reference_root=str(reference_root),
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        split="val",
        device=torch.device("cpu"),
        threshold=0.5,
        model=_DummyMergeModel(),
    )

    assert metrics["segm/AP50"] > 0.9
    assert summary["threshold"] == 0.5
    assert summary["num_images"] == 1
    assert (output_root / "metrics.cocoeval.json").exists()
    assert (output_root / "coco_instances_results.json").exists()


def test_render_reference_graph_preview_sheet_writes_progress_png(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    reference_root = tmp_path / "refs"
    output_root = tmp_path / "preview"
    _write_dataset(dataset_root)
    _write_graph_cache(cache_root)
    _write_reference_root(reference_root)

    output_path = output_root / "latest.png"
    rendered = render_reference_graph_preview_sheet(
        cache_root=str(cache_root),
        reference_root=str(reference_root),
        dataset_root=str(dataset_root),
        split="val",
        output_path=output_path,
        device=torch.device("cpu"),
        threshold=0.5,
        model=_DummyMergeModel(),
        batch_size=1,
        num_workers=0,
        limit=1,
    )

    assert rendered == output_path
    assert output_path.exists()


def test_evaluate_reference_graph_merge_filters_gt_to_cache_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    reference_root = tmp_path / "refs"
    output_root = tmp_path / "eval"
    _write_dataset(dataset_root, extra_uncovered_image=True)
    _write_graph_cache(cache_root)
    _write_reference_root(reference_root)

    metrics, summary = evaluate_reference_graph_merge(
        cache_root=str(cache_root),
        reference_root=str(reference_root),
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        split="val",
        device=torch.device("cpu"),
        threshold=0.5,
        model=_DummyMergeModel(),
    )

    assert metrics["segm/AP50"] > 0.9
    assert summary["num_images"] == 1
    assert summary["eval_image_count"] == 1


def test_eval_reference_graph_merge_script_cli_uses_best_threshold(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    reference_root = tmp_path / "refs"
    model_dir = tmp_path / "model"
    output_root = tmp_path / "eval_cli"
    config_path = tmp_path / "merge_eval.yaml"
    _write_dataset(dataset_root)
    _write_graph_cache(cache_root)
    _write_reference_root(reference_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "common:",
                "  device: cpu",
                "train:",
                "  batch_size: 1",
                "  num_workers: 0",
                "  decision_threshold: 0.5",
                "model:",
                "  reference_image_size: 128",
                "  reference_max_views: 16",
                "  reference_view_sampler: pose_farthest",
                "  hidden_dim: 64",
                "  reference_hidden_dim: 32",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = FragmentGraphMergeDataset(
        cache_root=str(cache_root),
        reference_root=str(reference_root),
        split="val",
        reference_image_size=128,
        reference_max_views=16,
        reference_view_sampler="pose_farthest",
    )
    probe = dataset[0]
    model = ReferenceGraphMergeModel(
        node_dim=int(probe["node_features"].shape[1]),
        edge_dim=int(probe["edge_features"].shape[1]),
        reference_dim=int(probe["reference_features"].shape[0]),
        hidden_dim=64,
        reference_hidden_dim=32,
    )
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    torch.save(model.state_dict(), model_dir / "model_best.pth")
    (model_dir / "train_summary.json").write_text(
        json.dumps({"best_threshold": 0.42}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/eval_reference_graph_merge.py",
            "--config",
            str(config_path),
            "--cache-root",
            str(cache_root),
            "--reference-root",
            str(reference_root),
            "--dataset-root",
            str(dataset_root),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_root),
            "--split",
            "val",
            "--device",
            "cpu",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["summary"]["threshold"] == 0.42
    assert payload["metrics"]["segm/AP50"] > 0.9
    assert (output_root / "metrics.cocoeval.json").exists()
    assert (output_root / "coco_instances_results.json").exists()
