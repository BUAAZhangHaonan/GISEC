from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any] | list[list[float]]:
    try:
        from pycocotools import mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"size": list(rle["size"]), "counts": counts}
    except ImportError:  # pragma: no cover - exercised in lean envs
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            polygons.append(contour.reshape(-1, 2).astype(float).flatten().tolist())
        return polygons or [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]


def evaluate_json(ann_file: Path, results_json: Path) -> dict[str, Any]:
    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(str(results_json))
    payload: dict[str, Any] = {}
    for metric in ("bbox", "segm"):
        coco_eval = COCOeval(coco_gt, coco_dt, metric)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        payload[f"{metric}/AP"] = float(coco_eval.stats[0])
        payload[f"{metric}/AP50"] = float(coco_eval.stats[1])
        payload[f"{metric}/AP75"] = float(coco_eval.stats[2])
    return payload


def build_benchmark_payload(latencies_ms: list[float], device: torch.device) -> dict[str, Any]:
    values = np.asarray(latencies_ms, dtype=np.float32)
    if values.size == 0:
        return {"device": device.type, "images": 0, "latency_ms_mean": 0.0, "latency_ms_p50": 0.0, "latency_ms_p90": 0.0}
    return {
        "device": device.type,
        "images": int(values.size),
        "latency_ms_mean": float(values.mean()),
        "latency_ms_p50": float(np.quantile(values, 0.50)),
        "latency_ms_p90": float(np.quantile(values, 0.90)),
    }


def build_device(device_name: str, local_rank: int | None = None) -> torch.device:
    requested = str(device_name or "cpu").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        if local_rank is not None:
            return torch.device("cuda", int(local_rank))
        return torch.device(requested)
    return torch.device("cpu")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
