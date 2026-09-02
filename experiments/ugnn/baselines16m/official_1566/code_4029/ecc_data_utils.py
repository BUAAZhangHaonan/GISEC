from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import cv2
import numpy as np

try:
    from .baseline_adapter_utils import annotations_to_instance_targets
except ImportError:  # pragma: no cover - file execution fallback
    from baseline_adapter_utils import annotations_to_instance_targets


def _resolve_ecc_image_dir(dataset_root: Path, split: str) -> Path:
    return dataset_root / "images" / split


def _resolve_ecc_annotation_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / "annotations" / f"instances_{split}.json"


def _index_annotations_by_image_id(
    annotations: Iterable[Mapping[str, Any]],
    image_ids: Iterable[int],
) -> Dict[int, List[Dict[str, Any]]]:
    selected_ids = {int(image_id) for image_id in image_ids}
    grouped: Dict[int, List[Dict[str, Any]]] = {image_id: [] for image_id in selected_ids}
    for annotation in annotations:
        image_id = int(annotation.get("image_id", -1))
        if image_id in grouped:
            grouped[image_id].append(dict(annotation))
    return grouped


def load_ecc_coco_rgb_records(
    dataset_root: str | Path,
    split: str,
    max_images: int | None = None,
    include_targets: bool = True,
) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    ann_path = _resolve_ecc_annotation_path(root, split)
    img_dir = _resolve_ecc_image_dir(root, split)
    payload = json.loads(ann_path.read_text(encoding="utf-8"))

    images = list(payload.get("images", []))
    if max_images is not None and int(max_images) > 0:
        images = images[: int(max_images)]
    annotations_by_image_id = _index_annotations_by_image_id(payload.get("annotations", []), [int(image_info["id"]) for image_info in images])

    records: List[Dict[str, Any]] = []
    for image_info in images:
        image_id = int(image_info["id"])
        image_path = img_dir / image_info["file_name"]
        annotations = annotations_by_image_id.get(image_id, [])
        record = {
            "image_id": image_id,
            "file_name": image_info["file_name"],
            "image_path": str(image_path),
            "height": int(image_info["height"]),
            "width": int(image_info["width"]),
            "annotations": annotations,
        }
        if include_targets:
            record["annotation_targets"] = annotations_to_instance_targets(
                annotations,
                height=int(image_info["height"]),
                width=int(image_info["width"]),
            )
        records.append(record)
    return records


def load_ecc_coco_rgb_image(image_path: str | Path, image_size: int | None = None) -> np.ndarray:
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        image = cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.uint8, copy=False)
