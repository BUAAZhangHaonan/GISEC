from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


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


def _write_prototype_bank(root: Path) -> None:
    for name in ["rgb", "depth", "mask", "meta"]:
        (root / name).mkdir(parents=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = (60, 80, 120)
    cv2.imwrite(str(root / "rgb" / "view0.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / "view0.npy", np.full((64, 64), 0.9, dtype=np.float32))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    cv2.imwrite(str(root / "mask" / "view0.png"), mask)


def _write_multi_part_prototype_root(root: Path, *, part_name: str) -> Path:
    part_root = root / part_name
    _write_prototype_bank(part_root)
    return root


def test_train_gisec_minibatch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    prototype_root = tmp_path / "prototype_bank"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)
    _write_prototype_bank(prototype_root)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.train",
            "--dataset-root",
            str(dataset_root),
            "--prototype-root",
            str(prototype_root),
            "--output-dir",
            str(output_root),
            "--variant",
            "G5",
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--epochs",
            "1",
            "--batch",
            "1",
            "--num-workers",
            "0",
            "--max-train-steps",
            "1",
            "--graph-warmup-steps",
            "16",
            "--max-val-images",
            "1",
            "--min-area",
            "4",
            "--contract-mode",
            "compat",
            "--save-overlays",
            "--overlay-limit",
            "1",
            "--save-graph-diagnostics",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_root / "metrics.cocoeval.json").exists()
    assert (output_root / "metrics_log.jsonl").exists()
    assert (output_root / "metrics_log.csv").exists()
    assert (output_root / "params_trainable.txt").exists()
    assert (output_root / "peak_memory_mb.txt").exists()
    assert (output_root / "run.log").exists()
    assert (output_root / "wall_time_sec.txt").exists()
    assert (output_root / "graph_diagnostics.jsonl").exists()
    epoch_artifact_root = output_root / "epoch_0001_artifacts"
    assert (epoch_artifact_root / "graph_diagnostics.jsonl").exists()
    assert (epoch_artifact_root / "graph_readiness_summary.json").exists()
    assert (epoch_artifact_root / "match_diagnostics_summary.json").exists()
    assert (epoch_artifact_root / "failure_summary.json").exists()
    assert list((output_root / "visualizations" / "overlay").glob("*.png"))
    run_summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "prototype_bank_manifest.json").read_text(encoding="utf-8"))
    assert run_summary["variant"] == "G5"
    assert run_summary["dataset_root"] == str(dataset_root.resolve())
    assert run_summary["prototype_root"] == str(prototype_root.resolve())
    assert run_summary["split"] == "val"
    assert run_summary["image_size"] == 64
    assert run_summary["batch"] == 1
    assert run_summary["num_workers"] == 0
    assert run_summary["min_area"] == 4
    assert run_summary["edge_threshold"] == 0.5
    assert manifest["max_views"] == 0
    assert manifest["view_sampler"] == "all"
    assert manifest["prototype_slot_count"] == 6
    assert manifest["prototype_topk"] == 2
    metric_rows = [json.loads(line) for line in (output_root / "metrics_log.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows = [row for row in metric_rows if row.get("mode") == "train"]
    epoch_eval_rows = [row for row in metric_rows if row.get("mode") == "epoch_eval"]
    assert train_rows
    assert epoch_eval_rows
    assert "graph_has_edges" in train_rows[0]
    assert "graph_edge_count" in train_rows[0]
    assert "graph_positive_edge_targets" in train_rows[0]
    assert train_rows[0]["graph_loss"] == 0.0
    assert "pred_fg_rate" in train_rows[0]
    assert "pred_boundary_rate" in train_rows[0]
    assert "target_fg_rate" in train_rows[0]
    assert "target_boundary_rate" in train_rows[0]
    assert "fg_prob_p50" in train_rows[0]
    assert "fg_prob_p95" in train_rows[0]
    assert "boundary_prob_p50" in train_rows[0]
    assert "boundary_prob_p95" in train_rows[0]
    assert epoch_eval_rows[0]["epoch"] == 1
    assert "graph_loss_mean" in epoch_eval_rows[0]
    assert "graph_loss_max" in epoch_eval_rows[0]
    assert "num_merged_mean" in epoch_eval_rows[0]
    assert "num_merged_std" in epoch_eval_rows[0]
    assert "num_merged_min" in epoch_eval_rows[0]
    assert "num_merged_max" in epoch_eval_rows[0]
    assert "gt_count_mean" in epoch_eval_rows[0]
    assert "pred_count_mean" in epoch_eval_rows[0]
    assert "non_finite_event_count" in epoch_eval_rows[0]
    assert epoch_eval_rows[0]["non_finite_event_count"] == 0


def test_train_gisec_minibatch_accepts_multi_part_reference_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    prototype_root = tmp_path / "prototype_bank"
    output_root = tmp_path / "out"
    file_name = "A-DF15A_KG-T2S_1_100_scene_000003_000968_v0.png"
    _write_dataset(dataset_root, file_name=file_name)
    _write_multi_part_prototype_root(prototype_root, part_name="A-DF15A_KG-T2S_1")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.train",
            "--dataset-root",
            str(dataset_root),
            "--prototype-root",
            str(prototype_root),
            "--output-dir",
            str(output_root),
            "--variant",
            "G5",
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--epochs",
            "1",
            "--batch",
            "1",
            "--num-workers",
            "0",
            "--max-train-steps",
            "1",
            "--max-val-images",
            "1",
            "--min-area",
            "4",
            "--contract-mode",
            "compat",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    run_summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "prototype_bank_manifest.json").read_text(encoding="utf-8"))
    assert run_summary["prototype_root"] == str(prototype_root.resolve())
    assert manifest["mode"] == "per_part"
    assert manifest["resolved_roots"] == [str((prototype_root / "A-DF15A_KG-T2S_1").resolve())]


def test_eval_can_reuse_model_config_saved_by_train_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    prototype_root = tmp_path / "prototype_bank"
    output_root = tmp_path / "out"
    eval_root = tmp_path / "eval"
    _write_dataset(dataset_root)
    _write_prototype_bank(prototype_root)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.train",
            "--dataset-root",
            str(dataset_root),
            "--prototype-root",
            str(prototype_root),
            "--output-dir",
            str(output_root),
            "--variant",
            "A1",
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--epochs",
            "1",
            "--batch",
            "1",
            "--num-workers",
            "0",
            "--max-train-steps",
            "1",
            "--max-val-images",
            "1",
            "--min-area",
            "4",
            "--contract-mode",
            "compat",
            "--base-channels",
            "8",
            "--graph-hidden-dim",
            "32",
            "--norm-layer",
            "group",
            "--prototype-slot-count",
            "4",
            "--prototype-topk",
            "1",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.eval",
            "--dataset-root",
            str(dataset_root),
            "--prototype-root",
            str(prototype_root),
            "--output-dir",
            str(output_root),
            "--checkpoint",
            str(output_root / "model_final.pth"),
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--max-images",
            "1",
            "--results-json",
            str(eval_root / "results.json"),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    model_config = json.loads((output_root / "model_config.json").read_text(encoding="utf-8"))
    assert model_config["base_channels"] == 8
    assert model_config["graph_hidden_dim"] == 32
    assert model_config["norm_layer"] == "group"
    assert model_config["prototype_slot_count"] == 4
    assert model_config["prototype_topk"] == 1
    assert (output_root / "metrics.cocoeval.json").exists()
