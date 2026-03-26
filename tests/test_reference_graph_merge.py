from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

from baseline.reference_graph.eval import DEFAULT_REFERENCE_GRAPH_THRESHOLDS
from baseline.reference_graph.eval import summarize_threshold_sweep
from baseline.reference_graph.train import build_edge_training_mask
from baseline.reference_graph.train import pairwise_ranking_loss
from baseline.reference_graph.train import train_reference_graph_merge
from gisec.models.graph_head import GraphEdgeScorer


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


def _write_graph_cache(root: Path, *, split: str = "train", part_key: str | None = "partA", count: int = 2) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for sample_index in range(count):
        payload = {
            "image_id": int(sample_index + 1),
            "file_name": f"{part_key or 'query'}_scene_{sample_index:04d}.png",
            "part_key": part_key,
            "fragments": torch.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 2, 2, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int16,
            ),
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
                {"gt_instance": 1, "purity": 1.0, "area_ratio": 0.02},
                {"gt_instance": 1, "purity": 1.0, "area_ratio": 0.02},
            ],
            "diagnostics": {"num_fragments": 2, "num_edges": 1},
            "shape_stats": {"mean_area_ratio": 0.02, "mean_aspect_ratio": 1.0},
            "summary": {"same_instance_recall": 1.0, "fragment_purity_mean": 1.0},
        }
        torch.save(payload, split_dir / f"{sample_index + 1:06d}.pt")
    (split_dir / "manifest.json").write_text(
        json.dumps({"split": split, "num_samples": count}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_train_reference_graph_merge_writes_summary_and_checkpoint(tmp_path: Path) -> None:
    cache_root = tmp_path / "graph_cache"
    reference_root = tmp_path / "references"
    output_root = tmp_path / "out"
    _write_graph_cache(cache_root, split="train", part_key="partA", count=2)
    _write_reference_root(reference_root, part_key="partA")

    train_reference_graph_merge(
        cache_root=str(cache_root),
        reference_root=str(reference_root),
        output_dir=str(output_root),
        split="train",
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        max_train_steps=1,
    )

    summary = json.loads((output_root / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs"] == 1
    assert summary["steps"] == 1
    assert summary["loss_total"] >= 0.0
    assert summary["edge_positive_rate"] == 1.0
    assert (output_root / "model_final.pth").exists()


def test_build_edge_training_mask_keeps_hardest_negatives() -> None:
    logits = torch.tensor([4.0, 3.0, 1.0, -1.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    valid_mask = torch.tensor([True, True, True, True], dtype=torch.bool)

    selected = build_edge_training_mask(
        logits=logits,
        targets=targets,
        valid_mask=valid_mask,
        hard_negative_ratio=1.0,
    )

    assert selected.tolist() == [True, True, False, False]


def test_pairwise_ranking_loss_rewards_margin_between_positive_and_negative_edges() -> None:
    logits = torch.tensor([0.6, 0.5, 0.1, 0.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    valid_mask = torch.tensor([True, True, True, True], dtype=torch.bool)

    low_loss = pairwise_ranking_loss(
        logits=logits,
        targets=targets,
        valid_mask=valid_mask,
        margin=0.2,
        max_samples_per_class=8,
    )
    high_loss = pairwise_ranking_loss(
        logits=torch.tensor([0.15, 0.1, 0.14, 0.12], dtype=torch.float32),
        targets=targets,
        valid_mask=valid_mask,
        margin=0.2,
        max_samples_per_class=8,
    )

    assert float(low_loss.item()) == 0.0
    assert float(high_loss.item()) > 0.0


def test_reference_graph_default_threshold_sweep_captures_fine_margin_region() -> None:
    logits = torch.tensor([0.024, 0.022, 0.016, 0.015, 0.005, -0.01], dtype=torch.float32)
    targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    sweep = summarize_threshold_sweep(
        logits,
        targets,
        thresholds=DEFAULT_REFERENCE_GRAPH_THRESHOLDS,
        conservative_f1_margin=0.01,
    )

    assert 0.505 in DEFAULT_REFERENCE_GRAPH_THRESHOLDS
    assert float(sweep["best"]["threshold"]) == 0.505


def test_graph_edge_scorer_is_symmetric_for_undirected_edge_order() -> None:
    scorer = GraphEdgeScorer(node_dim=3, edge_dim=2, hidden_dim=4)
    with torch.no_grad():
        for param_index, param in enumerate(scorer.parameters()):
            values = torch.arange(1, param.numel() + 1, dtype=param.dtype).reshape(param.shape)
            param.copy_(values / float(param.numel() + param_index + 1))

    node_features = torch.tensor(
        [
            [0.2, 0.5, -0.3],
            [1.1, -0.7, 0.4],
        ],
        dtype=torch.float32,
    )
    edge_features = torch.tensor([[0.6, -0.2]], dtype=torch.float32)
    edge_ab = torch.tensor([[0], [1]], dtype=torch.long)
    edge_ba = torch.tensor([[1], [0]], dtype=torch.long)

    logit_ab = scorer(node_features=node_features, edge_index=edge_ab, edge_features=edge_features)
    logit_ba = scorer(node_features=node_features, edge_index=edge_ba, edge_features=edge_features)

    assert torch.allclose(logit_ab, logit_ba, atol=1e-6)


def test_train_reference_graph_merge_supports_single_bank_reference_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "graph_cache"
    reference_root = tmp_path / "references"
    output_root = tmp_path / "out"
    _write_graph_cache(cache_root, split="train", part_key=None, count=1)
    _write_reference_root(reference_root, part_key="partA")

    train_reference_graph_merge(
        cache_root=str(cache_root),
        reference_root=str(reference_root / "partA"),
        output_dir=str(output_root),
        split="train",
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
    )

    summary = json.loads((output_root / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] == 1
    assert summary["reference_mode"] == "single_bank"


def test_train_reference_graph_merge_writes_val_summary_and_best_checkpoint(tmp_path: Path) -> None:
    cache_root = tmp_path / "graph_cache"
    reference_root = tmp_path / "references"
    output_root = tmp_path / "out"
    _write_graph_cache(cache_root, split="train", part_key="partA", count=2)
    _write_graph_cache(cache_root, split="val", part_key="partA", count=1)
    _write_reference_root(reference_root, part_key="partA")

    train_reference_graph_merge(
        cache_root=str(cache_root),
        reference_root=str(reference_root),
        output_dir=str(output_root),
        split="train",
        device=torch.device("cpu"),
        epochs=2,
        batch_size=2,
        num_workers=0,
        max_train_steps=0,
        val_split="val",
        negative_edge_weight=3.0,
    )

    train_summary = json.loads((output_root / "train_summary.json").read_text(encoding="utf-8"))
    val_summary = json.loads((output_root / "val_summary.json").read_text(encoding="utf-8"))
    val_sweep = json.loads((output_root / "val_threshold_sweep.json").read_text(encoding="utf-8"))
    assert train_summary["best_val_f1"] >= 0.0
    assert train_summary["negative_edge_weight"] == 3.0
    assert "best_threshold" in train_summary
    assert "best_conservative_threshold" in train_summary
    assert "best_threshold" in val_summary
    assert "best_conservative_threshold" in val_summary
    assert "precision" in val_summary
    assert "recall" in val_summary
    assert "f1" in val_summary
    assert len(val_sweep["rows"]) >= 2
    assert (output_root / "model_best.pth").exists()


def test_train_reference_graph_merge_script_cli_overrides_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = tmp_path / "graph_cache"
    reference_root = tmp_path / "references"
    output_root = tmp_path / "out"
    config_path = tmp_path / "merge.yaml"
    _write_graph_cache(cache_root, split="train", part_key="partA", count=2)
    _write_reference_root(reference_root, part_key="partA")
    config_path.write_text(
        "\n".join(
            [
                "train:",
                "  epochs: 5",
                "  batch_size: 8",
                "  num_workers: 0",
                "  max_train_steps: 0",
                "model:",
                "  reference_image_size: 64",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/train_reference_graph_merge.py",
            "--config",
            str(config_path),
            "--cache-root",
            str(cache_root),
            "--reference-root",
            str(reference_root),
            "--output-dir",
            str(output_root),
            "--split",
            "train",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--max-train-steps",
            "1",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_root / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs"] == 1
    assert summary["steps"] == 1
