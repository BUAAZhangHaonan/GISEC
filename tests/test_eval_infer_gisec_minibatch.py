from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_dataset(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = (60, 80, 120)
        cv2.imwrite(str(root / "images" / split / "000001.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / "000001.npy", np.full((64, 64), 0.9, dtype=np.float32))
        ann = {
            "images": [{"id": 1, "file_name": "000001.png", "width": 64, "height": 64}],
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


def _run_train(repo_root: Path, dataset_root: Path, prototype_root: Path, output_root: Path) -> None:
    subprocess.run(
            [
                sys.executable,
                "-m",
                "gisec.cli.train_legacy",
                "--dataset-root",
                str(dataset_root),
                "--prototype-root",
                str(prototype_root),
            "--output-dir",
            str(output_root),
            "--variant",
            "legacy_prototype_unet_with_rgbd_similarity_shape_stats",
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


def test_eval_and_infer_gisec_minibatch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    prototype_root = tmp_path / "prototype_bank"
    train_output = tmp_path / "train_out"
    eval_output = tmp_path / "eval_out"
    infer_output = tmp_path / "infer_out"
    _write_dataset(dataset_root)
    _write_prototype_bank(prototype_root)
    _run_train(repo_root, dataset_root, prototype_root, train_output)

    checkpoint = train_output / "model_best.pth"

    subprocess.run(
            [
                sys.executable,
                "-m",
                "gisec.cli.eval_legacy",
                "--dataset-root",
                str(dataset_root),
                "--prototype-root",
                str(prototype_root),
            "--output-dir",
            str(eval_output),
            "--checkpoint-dir",
            str(train_output),
            "--variant",
            "legacy_prototype_unet_with_rgbd_similarity_shape_stats",
            "--checkpoint",
            "model_best.pth",
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--num-workers",
            "0",
            "--max-images",
            "1",
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

    subprocess.run(
            [
                sys.executable,
                "-m",
                "gisec.cli.infer_legacy",
                "--dataset-root",
                str(dataset_root),
                "--prototype-root",
                str(prototype_root),
            "--output-dir",
            str(infer_output),
            "--checkpoint-dir",
            str(train_output),
            "--variant",
            "legacy_prototype_unet_with_rgbd_similarity_shape_stats",
            "--checkpoint",
            "model_best.pth",
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--num-workers",
            "0",
            "--max-images",
            "1",
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

    assert (eval_output / "metrics.cocoeval.json").exists()
    assert (eval_output / "run_summary.json").exists()
    assert (eval_output / "graph_diagnostics.jsonl").exists()
    assert (eval_output / "failure_summary.json").exists()
    assert (eval_output / "mask_calibration_summary.json").exists()
    assert (eval_output / "component_pathology_summary.json").exists()
    assert (eval_output / "graph_readiness_summary.json").exists()
    assert (eval_output / "routing_confidence_summary.json").exists()
    assert (eval_output / "reference_routing_summary.json").exists()
    assert (infer_output / "coco_instances_results.json").exists()
    assert (infer_output / "coco_instances_results.raw.json").exists()
    assert (infer_output / "run_summary.json").exists()
    assert (infer_output / "graph_diagnostics.jsonl").exists()
    assert (infer_output / "failure_summary.json").exists()
    assert (infer_output / "mask_calibration_summary.json").exists()
    assert (infer_output / "component_pathology_summary.json").exists()
    assert (infer_output / "graph_readiness_summary.json").exists()
    assert (infer_output / "routing_confidence_summary.json").exists()
    assert (infer_output / "reference_routing_summary.json").exists()
    eval_overlays = list((eval_output / "visualizations" / "overlay").glob("*.png"))
    infer_overlays = list((infer_output / "visualizations" / "overlay").glob("*.png"))
    assert eval_overlays
    assert infer_overlays
    labels = ["_normal_", "_tiny_island_", "_full_frame_", "_empty_", "_border_strip_", "_oversized_blob_", "_mixed_"]
    assert any(any(label in path.name for label in labels) for path in eval_overlays)
    assert any(any(label in path.name for label in labels) for path in infer_overlays)

    eval_summary = json.loads((eval_output / "run_summary.json").read_text(encoding="utf-8"))
    infer_summary = json.loads((infer_output / "run_summary.json").read_text(encoding="utf-8"))
    eval_failure_summary = json.loads((eval_output / "failure_summary.json").read_text(encoding="utf-8"))
    infer_failure_summary = json.loads((infer_output / "failure_summary.json").read_text(encoding="utf-8"))
    eval_mask_summary = json.loads((eval_output / "mask_calibration_summary.json").read_text(encoding="utf-8"))
    infer_mask_summary = json.loads((infer_output / "mask_calibration_summary.json").read_text(encoding="utf-8"))
    eval_component_summary = json.loads((eval_output / "component_pathology_summary.json").read_text(encoding="utf-8"))
    infer_component_summary = json.loads((infer_output / "component_pathology_summary.json").read_text(encoding="utf-8"))
    eval_graph_readiness = json.loads((eval_output / "graph_readiness_summary.json").read_text(encoding="utf-8"))
    infer_graph_readiness = json.loads((infer_output / "graph_readiness_summary.json").read_text(encoding="utf-8"))
    eval_routing_confidence = json.loads((eval_output / "routing_confidence_summary.json").read_text(encoding="utf-8"))
    infer_routing_confidence = json.loads((infer_output / "routing_confidence_summary.json").read_text(encoding="utf-8"))
    eval_routing_summary = json.loads((eval_output / "reference_routing_summary.json").read_text(encoding="utf-8"))
    infer_routing_summary = json.loads((infer_output / "reference_routing_summary.json").read_text(encoding="utf-8"))
    assert eval_summary["dataset_root"] == str(dataset_root.resolve())
    assert eval_summary["prototype_root"] == str(prototype_root.resolve())
    assert eval_summary["split"] == "val"
    assert eval_summary["image_size"] == 64
    assert eval_summary["checkpoint"] == str(checkpoint.resolve())
    assert (eval_output / "model_config.json").exists()
    assert infer_summary["dataset_root"] == str(dataset_root.resolve())
    assert infer_summary["prototype_root"] == str(prototype_root.resolve())
    assert infer_summary["split"] == "val"
    assert infer_summary["image_size"] == 64
    assert infer_summary["checkpoint"] == str(checkpoint.resolve())
    assert (infer_output / "model_config.json").exists()
    assert eval_failure_summary["total_images"] == 1
    assert infer_failure_summary["total_images"] == 1
    assert set(eval_failure_summary["counts"]).issuperset({"normal", "tiny_island", "full_frame", "empty"})
    assert set(infer_failure_summary["counts"]).issuperset({"normal", "tiny_island", "full_frame", "empty"})
    assert eval_mask_summary["total_images"] == 1
    assert infer_mask_summary["total_images"] == 1
    assert "pred_fg_rate_mean" in eval_mask_summary
    assert "pred_boundary_rate_mean" in eval_mask_summary
    assert eval_component_summary["total_images"] == 1
    assert infer_component_summary["total_images"] == 1
    assert "largest_component_ratio_mean" in eval_component_summary
    assert eval_graph_readiness["total_images"] == 1
    assert infer_graph_readiness["total_images"] == 1
    assert "zero_edge_ratio" in eval_graph_readiness
    assert "num_merged_std" in eval_graph_readiness
    assert "num_merged_min" in eval_graph_readiness
    assert "num_merged_max" in eval_graph_readiness
    assert eval_routing_confidence["total_images"] == 1
    assert infer_routing_confidence["total_images"] == 1
    assert "top1_top2_margin_mean" in eval_routing_confidence
    assert eval_routing_summary["total_images"] == 1
    assert infer_routing_summary["total_images"] == 1
    assert "prototype_slot_count" in eval_routing_summary
    assert "prototype_topk" in eval_routing_summary
    assert "selected_view_histogram" in eval_routing_summary
    eval_graph_rows = [json.loads(line) for line in (eval_output / "graph_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    infer_graph_rows = [json.loads(line) for line in (infer_output / "graph_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert eval_graph_rows
    assert infer_graph_rows
    assert "num_fragments" in eval_graph_rows[0]
    assert "num_edges" in eval_graph_rows[0]
    assert "num_merged" in eval_graph_rows[0]
    for key in [
        "ownership_offset_prediction_error",
        "boundary_miss_rate",
        "fragment_overflow_rate",
        "fragment_impurity_rate",
        "over_merge_count",
        "under_merge_count",
    ]:
        assert key in eval_graph_rows[0]
        assert key in infer_graph_rows[0]


def test_eval_and_infer_reject_shared_checkpoint_and_output_roots(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    prototype_root = tmp_path / "prototype_bank"
    train_output = tmp_path / "train_out"
    _write_dataset(dataset_root)
    _write_prototype_bank(prototype_root)
    _run_train(repo_root, dataset_root, prototype_root, train_output)

    for command in ["eval", "infer"]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                f"gisec.cli.{command}_legacy",
                "--dataset-root",
                str(dataset_root),
                "--prototype-root",
                str(prototype_root),
                "--output-dir",
                str(train_output),
                "--checkpoint-dir",
                str(train_output),
                "--variant",
                "legacy_prototype_unet_with_rgbd_similarity_shape_stats",
                "--checkpoint",
                "model_best.pth",
                "--device",
                "cpu",
                "--image-size",
                "64",
                "--num-workers",
                "0",
                "--max-images",
                "1",
                "--contract-mode",
                "compat",
            ],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "checkpoint-dir" in (result.stderr + result.stdout)
