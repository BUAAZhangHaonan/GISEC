from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from gisec.train.train_active import _query_instances_from_outputs


def _write_dataset(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[10:30, 10:26] = (40, 100, 180)
        image[24:52, 34:54] = (180, 80, 40)
        cv2.imwrite(str(root / "images" / split / "000001.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        depth = np.full((64, 64), 0.9, dtype=np.float32)
        depth[24:52, 34:54] = 0.6
        np.save(root / "depth" / split / "000001.npy", depth)
        ann = {
            "images": [{"id": 1, "file_name": "000001.png", "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 16, 20],
                    "area": 320,
                    "iscrowd": 0,
                    "segmentation": [[10, 10, 26, 10, 26, 30, 10, 30]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [34, 24, 20, 28],
                    "area": 560,
                    "iscrowd": 0,
                    "segmentation": [[34, 24, 54, 24, 54, 52, 34, 52]],
                },
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def _write_split_like_dataset(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[10:24, 10:22] = (40, 100, 180)
        image[30:44, 30:42] = (40, 100, 180)
        cv2.imwrite(str(root / "images" / split / "000001.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        depth = np.full((64, 64), 0.9, dtype=np.float32)
        depth[10:24, 10:22] = 0.7
        depth[30:44, 30:42] = 0.72
        np.save(root / "depth" / split / "000001.npy", depth)
        ann = {
            "images": [{"id": 1, "file_name": "000001.png", "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 32, 34],
                    "area": 336,
                    "iscrowd": 0,
                    "segmentation": [
                        [10, 10, 22, 10, 22, 24, 10, 24],
                        [30, 30, 42, 30, 42, 44, 30, 44],
                    ],
                }
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def _write_prototype_bank(root: Path) -> None:
    for name in ["rgb", "depth", "mask", "meta"]:
        (root / name).mkdir(parents=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[10:24, 10:22] = (40, 100, 180)
    image[30:44, 30:42] = (40, 100, 180)
    cv2.imwrite(str(root / "rgb" / "view0.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    depth = np.full((64, 64), 0.8, dtype=np.float32)
    np.save(root / "depth" / "view0.npy", depth)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:24, 10:22] = 255
    mask[30:44, 30:42] = 255
    cv2.imwrite(str(root / "mask" / "view0.png"), mask)


def _active_args(dataset_root: Path, output_root: Path, *, variant: str) -> list[str]:
    return [
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(output_root),
        "--variant",
        variant,
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
        "--score-threshold",
        "0.0",
        "--mask-threshold",
        "0.5",
        "--pretrained-model-name",
        "none",
        "--hidden-dim",
        "32",
        "--feature-size",
        "32",
        "--mask-feature-size",
        "32",
        "--encoder-layers",
        "1",
        "--decoder-layers",
        "1",
        "--num-attention-heads",
        "4",
        "--num-queries",
        "8",
        "--train-num-points",
        "64",
    ]


def test_active_query_export_keeps_foreground_queries_even_when_label_index_is_one() -> None:
    class_logits = torch.tensor(
        [
            [-8.0, 4.0, -2.0],
            [-7.5, 3.5, -1.0],
        ],
        dtype=torch.float32,
    )
    mask_logits = torch.full((2, 8, 8), 8.0, dtype=torch.float32)

    rows = _query_instances_from_outputs(
        class_logits=class_logits,
        mask_logits=mask_logits,
        image_shape=(8, 8),
        score_threshold=0.5,
        mask_threshold=0.5,
    )

    assert len(rows) == 2
    assert all(int(row["binary_mask"].sum().item()) == 64 for row in rows)


def test_active_cli_minibatch_runs_train_eval_infer_for_base_rgb(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    train_root = tmp_path / "train_out"
    eval_root = tmp_path / "eval_out"
    infer_root = tmp_path / "infer_out"
    _write_dataset(dataset_root)

    subprocess.run(
        [sys.executable, "-m", "gisec.cli.train", *_active_args(dataset_root, train_root, variant="base_rgb_1024")],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    checkpoint = train_root / "model_best.pth"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.eval",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(eval_root),
            "--variant",
            "base_rgb_1024",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--num-workers",
            "0",
            "--max-images",
            "1",
            "--score-threshold",
            "0.0",
            "--mask-threshold",
            "0.5",
            "--pretrained-model-name",
            "none",
            "--hidden-dim",
            "32",
            "--feature-size",
            "32",
            "--mask-feature-size",
            "32",
            "--encoder-layers",
            "1",
            "--decoder-layers",
            "1",
            "--num-attention-heads",
            "4",
            "--num-queries",
            "8",
            "--train-num-points",
            "64",
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
            "gisec.cli.infer",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(infer_root),
            "--variant",
            "base_rgb_1024",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--num-workers",
            "0",
            "--max-images",
            "1",
            "--score-threshold",
            "0.0",
            "--mask-threshold",
            "0.5",
            "--pretrained-model-name",
            "none",
            "--hidden-dim",
            "32",
            "--feature-size",
            "32",
            "--mask-feature-size",
            "32",
            "--encoder-layers",
            "1",
            "--decoder-layers",
            "1",
            "--num-attention-heads",
            "4",
            "--num-queries",
            "8",
            "--train-num-points",
            "64",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    train_summary = json.loads((train_root / "run_summary.json").read_text(encoding="utf-8"))
    eval_summary = json.loads((eval_root / "run_summary.json").read_text(encoding="utf-8"))
    infer_summary = json.loads((infer_root / "run_summary.json").read_text(encoding="utf-8"))
    assert train_summary["variant"] == "base_rgb_1024"
    assert eval_summary["variant"] == "base_rgb_1024"
    assert infer_summary["variant"] == "base_rgb_1024"
    assert "split_gt_count" in train_summary["metrics"]
    assert "merge_pred_count" in train_summary["metrics"]
    assert "refinement_invocation_rate" in train_summary["metrics"]
    assert (infer_root / "coco_instances_results.raw.json").exists()


def test_active_cli_minibatch_runs_base_rgbd_concat_without_prototype_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    subprocess.run(
        [sys.executable, "-m", "gisec.cli.train", *_active_args(dataset_root, output_root, variant="base_rgbd_1024")],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["variant"] == "base_rgbd_1024"
    assert summary["modality"] == "rgbd_concat"
    assert summary["benchmark"]["input_mode"] == "rgbd_concat"
    assert summary["metrics"]["refinement_invocation_rate"] == 0.0


def test_active_cli_minibatch_runs_refine_reference_graph_variant_with_prototype_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    prototype_root = tmp_path / "prototype_bank"
    base_root = tmp_path / "base_out"
    output_root = tmp_path / "out"
    _write_split_like_dataset(dataset_root)
    _write_prototype_bank(prototype_root)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.train",
            *_active_args(dataset_root, base_root, variant="base_rgbd_1024"),
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
            "gisec.cli.train",
            *_active_args(dataset_root, output_root, variant="base_rgbd_1024_refine_ref_graph"),
            "--prototype-root",
            str(prototype_root),
            "--init-checkpoint",
            str(base_root / "model_final.pth"),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["variant"] == "base_rgbd_1024_refine_ref_graph"
    assert summary["modality"] == "rgbd_concat"
    assert "refinement_invocation_rate" in summary["metrics"]
    assert "local_graph_invocation_rate" in summary["metrics"]
