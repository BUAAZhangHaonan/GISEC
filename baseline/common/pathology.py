from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from baseline.rgbd.depth_cache import build_depth_feature_pack
from gisec.datasets.ecc_query_dataset import _load_depth_array, ann_to_mask


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _annotation_path(dataset_root: str, split: str) -> Path:
    return Path(dataset_root).resolve() / "annotations" / f"instances_{split}.json"


def _depth_path(dataset_root: str, split: str, file_name: str) -> Path:
    root = Path(dataset_root).resolve()
    candidates = [
        root / "depth" / split / f"{Path(file_name).stem}.npy",
        root / "depth" / "depth_npy" / split / f"{Path(file_name).stem}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(file_name)


def _bbox_diag_from_mask(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0.0
    width = float(int(xs.max()) - int(xs.min()) + 1)
    height = float(int(ys.max()) - int(ys.min()) + 1)
    return float(np.sqrt(width * width + height * height))


def _decode_prediction_mask(pred: dict[str, Any]) -> np.ndarray:
    return mask_utils.decode(pred["segmentation"]).astype(bool)


def build_prediction_pathology_rows(
    *,
    dataset_root: str,
    results_json: str,
    split: str = "val",
) -> list[dict[str, Any]]:
    ann_payload = _read_json(_annotation_path(dataset_root, split))
    results = json.loads(Path(results_json).read_text(encoding="utf-8"))
    image_by_id = {int(image["id"]): image for image in ann_payload.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gt_areas: list[float] = []
    gt_diags: list[float] = []
    for ann in ann_payload.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)
        mask = ann_to_mask(ann, int(image_by_id[int(ann["image_id"])]["height"]), int(image_by_id[int(ann["image_id"])]["width"]))
        gt_areas.append(float(mask.sum()))
        gt_diags.append(_bbox_diag_from_mask(mask))
    gt_area_median = float(median(gt_areas)) if gt_areas else 1.0
    gt_diag_median = float(median(gt_diags)) if gt_diags else 1.0

    rows: list[dict[str, Any]] = []
    for pred_index, pred in enumerate(results):
        image_id = int(pred["image_id"])
        image_info = image_by_id[image_id]
        pred_mask = _decode_prediction_mask(pred)
        pred_area = float(pred_mask.sum())
        pred_diag = _bbox_diag_from_mask(pred_mask)
        depth = _load_depth_array(_depth_path(dataset_root, split, str(image_info["file_name"])))
        discontinuity = build_depth_feature_pack(depth, feature_mode="depth_geometry_dense")[4]
        if discontinuity.shape != pred_mask.shape:
            discontinuity = cv2.resize(discontinuity.astype(np.float32), (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        if depth.shape != pred_mask.shape:
            depth = cv2.resize(depth.astype(np.float32), (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        blob_depth = depth[pred_mask]
        depth_median = float(np.median(blob_depth)) if blob_depth.size else 0.0
        depth_residual_mad = float(np.median(np.abs(blob_depth - depth_median))) if blob_depth.size else 0.0
        gt_instance_count_in_blob = 0
        for ann in anns_by_image.get(image_id, []):
            gt_mask = ann_to_mask(ann, int(image_info["height"]), int(image_info["width"])).astype(bool)
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if bool((gt_mask & pred_mask).any()):
                gt_instance_count_in_blob += 1
        rows.append(
            {
                "image_id": image_id,
                "file_name": str(image_info["file_name"]),
                "prediction_index": int(pred_index),
                "score": float(pred.get("score", 0.0)),
                "gt_instance_count_in_blob": int(gt_instance_count_in_blob),
                "pred_area_ratio": float(pred_area / (float(pred_mask.shape[0]) * float(pred_mask.shape[1]))),
                "area_multiple_to_gt_median": float(pred_area / gt_area_median) if gt_area_median > 0 else 0.0,
                "diag_multiple_to_gt_median": float(pred_diag / gt_diag_median) if gt_diag_median > 0 else 0.0,
                "depth_residual_mad": depth_residual_mad,
                "depth_discontinuity_mean": float(discontinuity[pred_mask].mean()) if pred_mask.any() else 0.0,
            }
        )
    return rows


def summarize_prediction_pathology(
    *,
    dataset_root: str,
    results_json: str,
    split: str = "val",
) -> dict[str, Any]:
    rows = build_prediction_pathology_rows(dataset_root=dataset_root, results_json=results_json, split=split)
    if not rows:
        return {
            "num_predictions": 0,
            "num_multi_gt_blobs": 0,
            "median_gt_instance_count_in_blob": 0.0,
            "median_area_multiple_to_gt_median": 0.0,
            "median_diag_multiple_to_gt_median": 0.0,
            "median_depth_residual_mad": 0.0,
            "median_depth_discontinuity_mean": 0.0,
        }
    return {
        "num_predictions": int(len(rows)),
        "num_multi_gt_blobs": int(sum(int(row["gt_instance_count_in_blob"] >= 2) for row in rows)),
        "median_gt_instance_count_in_blob": float(median([row["gt_instance_count_in_blob"] for row in rows])),
        "median_area_multiple_to_gt_median": float(median([row["area_multiple_to_gt_median"] for row in rows])),
        "median_diag_multiple_to_gt_median": float(median([row["diag_multiple_to_gt_median"] for row in rows])),
        "median_depth_residual_mad": float(median([row["depth_residual_mad"] for row in rows])),
        "median_depth_discontinuity_mean": float(median([row["depth_discontinuity_mean"] for row in rows])),
    }
