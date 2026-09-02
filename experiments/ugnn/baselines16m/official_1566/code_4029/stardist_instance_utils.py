from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

try:
    from .baseline_adapter_utils import binary_masks_to_coco_rows
    from .baseline_adapter_utils import decode_coco_segmentation
    from .ecc_data_utils import load_ecc_coco_rgb_image, load_ecc_coco_rgb_records
except ImportError:  # pragma: no cover - file execution fallback
    from baseline_adapter_utils import binary_masks_to_coco_rows
    from baseline_adapter_utils import decode_coco_segmentation
    from ecc_data_utils import load_ecc_coco_rgb_image, load_ecc_coco_rgb_records


SUPPORTED_IMAGE_SIZES = (512, 1024)


def _annotations_to_instance_map(
    annotations: Sequence[Mapping[str, Any]],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    instance_map = np.zeros((int(height), int(width)), dtype=np.int32)
    for instance_id, annotation in enumerate(annotations, start=1):
        mask = decode_coco_segmentation(annotation.get("segmentation"), int(height), int(width))
        instance_map[mask > 0] = int(instance_id)
    return instance_map


def _resize_instance_map(instance_map: np.ndarray, image_size: int) -> np.ndarray:
    if instance_map.shape[:2] == (int(image_size), int(image_size)):
        resized = instance_map.astype(np.int32, copy=False)
    else:
        resized = cv2.resize(
            instance_map.astype(np.int32, copy=False),
            (int(image_size), int(image_size)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32, copy=False)
    max_label = int(resized.max()) if resized.size else 0
    if max_label <= np.iinfo(np.uint16).max:
        return resized.astype(np.uint16, copy=False)
    return resized


def load_stardist_ecc_split(
    dataset_root: str | Path,
    split: str,
    image_size: int,
    *,
    max_images: int | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    records = load_ecc_coco_rgb_records(dataset_root, split, max_images=max_images, include_targets=False)

    images: List[np.ndarray] = []
    label_maps: List[np.ndarray] = []
    selected_records: List[Dict[str, Any]] = []
    for record in records:
        image = load_ecc_coco_rgb_image(record["image_path"], image_size=int(image_size))
        instance_map = _annotations_to_instance_map(
            record.get("annotations", []),
            height=int(record["height"]),
            width=int(record["width"]),
        )
        instance_map = _resize_instance_map(
            np.asarray(instance_map, dtype=np.int32),
            int(image_size),
        )
        images.append(image.astype(np.float32, copy=False) / 255.0)
        label_maps.append(instance_map)
        selected_records.append(
            {
                "image_id": int(record["image_id"]),
                "file_name": record["file_name"],
                "image_path": record["image_path"],
                "height": int(record["height"]),
                "width": int(record["width"]),
            }
        )
    return images, label_maps, selected_records


def stardist_prediction_to_coco_rows(
    *,
    image_id: int,
    labels: np.ndarray,
    details: Mapping[str, Any] | Dict[str, Any] | None,
    score_threshold: float = 0.05,
    output_size: Tuple[int, int] | None = None,
) -> List[Dict[str, Any]]:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label image, got shape={labels.shape!r}")
    if output_size is not None:
        height, width = int(output_size[0]), int(output_size[1])
        if labels.shape[:2] != (height, width):
            labels = cv2.resize(labels.astype(np.int32, copy=False), (width, height), interpolation=cv2.INTER_NEAREST)

    object_ids = [int(label_id) for label_id in np.unique(labels).tolist() if int(label_id) > 0]
    masks = [(labels == label_id).astype(np.uint8, copy=False) for label_id in object_ids]
    scores = None if details is None else details.get("prob")
    category_ids = np.zeros((len(masks),), dtype=np.int64)
    return binary_masks_to_coco_rows(
        image_id=int(image_id),
        masks=masks,
        scores=scores,
        category_ids=category_ids,
        score_threshold=float(score_threshold),
        mask_threshold=0.5,
        category_offset=1,
    )
