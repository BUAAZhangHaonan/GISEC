from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from gisec.datasets.ecc_query_dataset import _LiteCOCO, ann_to_mask


def get_ultralytics_yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised in real env, not smoke stub
        raise RuntimeError(
            "YOLOv8-seg baseline requires the optional `ultralytics` package."
        ) from exc
    return YOLO


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _normalize_polygon(points: np.ndarray, *, width: int, height: int) -> list[float]:
    coords = points.astype(np.float32).copy()
    coords[:, 0] = np.clip(coords[:, 0] / float(max(width, 1)), 0.0, 1.0)
    coords[:, 1] = np.clip(coords[:, 1] / float(max(height, 1)), 0.0, 1.0)
    return coords.reshape(-1).tolist()


def _ann_to_yolo_segments(ann: dict[str, Any], *, width: int, height: int) -> list[list[float]]:
    segmentation = ann.get("segmentation")
    segments: list[list[float]] = []
    if isinstance(segmentation, list):
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if points.shape[0] < 3:
                continue
            segments.append(_normalize_polygon(points, width=width, height=height))
        if segments:
            return segments
    mask = ann_to_mask(ann, height, width)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = contour.reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        segments.append(_normalize_polygon(points, width=width, height=height))
    return segments


def export_yolov8_seg_dataset(*, dataset_root: str, output_dir: str) -> dict[str, str]:
    root = Path(output_dir).resolve()
    images_root = root / "images"
    labels_root = root / "labels"
    dataset_path = root / "dataset.yaml"
    for split in ["train", "val"]:
        coco = _LiteCOCO(Path(dataset_root) / "annotations" / f"instances_{split}.json")
        for image_id in coco.getImgIds():
            image_info = coco.loadImgs([image_id])[0]
            src_image = Path(dataset_root) / "images" / split / image_info["file_name"]
            dst_image = images_root / split / image_info["file_name"]
            _link_or_copy(src_image, dst_image)
            ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            label_lines: list[str] = []
            width = int(image_info["width"])
            height = int(image_info["height"])
            for ann in anns:
                for segment in _ann_to_yolo_segments(ann, width=width, height=height):
                    if len(segment) < 6:
                        continue
                    label_lines.append("0 " + " ".join(f"{value:.6f}" for value in segment))
            label_path = labels_root / split / f"{Path(image_info['file_name']).stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    payload = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "component"},
    }
    dataset_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return {"root": str(root), "dataset_yaml": str(dataset_path)}


def prediction_to_instance_masks(prediction: Any, *, score_threshold: float) -> tuple[list[np.ndarray], list[float]]:
    masks_obj = getattr(prediction, "masks", None)
    boxes_obj = getattr(prediction, "boxes", None)
    if masks_obj is None:
        return [], []
    mask_data = getattr(masks_obj, "data", None)
    if mask_data is None:
        return [], []
    if hasattr(mask_data, "detach"):
        mask_array = mask_data.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask_data)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    if boxes_obj is not None and getattr(boxes_obj, "conf", None) is not None:
        conf = boxes_obj.conf
        if hasattr(conf, "detach"):
            scores_array = conf.detach().cpu().numpy().astype(np.float32)
        else:
            scores_array = np.asarray(conf, dtype=np.float32)
    else:
        scores_array = np.ones((mask_array.shape[0],), dtype=np.float32)
    masks: list[np.ndarray] = []
    scores: list[float] = []
    for score, mask in zip(scores_array.tolist(), mask_array):
        if float(score) < float(score_threshold):
            continue
        binary = (np.asarray(mask) >= 0.5).astype(np.uint8)
        if int(binary.sum()) <= 0:
            continue
        masks.append(binary)
        scores.append(float(score))
    return masks, scores
