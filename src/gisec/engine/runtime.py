from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any] | list[list[float]]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(rle["size"]), "counts": counts}


def evaluate_json(ann_file: Path, results_json: Path) -> dict[str, Any]:
    results = json.loads(results_json.read_text(encoding="utf-8"))
    if not results:
        # An empty candidate set is a legitimate outcome (a checkpoint may
        # predict nothing above the score threshold); report zero AP with a
        # note instead of crashing inside COCOeval.
        payload: dict[str, Any] = {"note": "no predictions"}
        for metric in ("bbox", "segm"):
            for suffix in ("AP", "AP50", "AP75"):
                payload[f"{metric}/{suffix}"] = 0.0
        return payload
    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(str(results_json))
    payload = {}
    for metric in ("bbox", "segm"):
        coco_eval = COCOeval(coco_gt, coco_dt, metric)
        # Standard COCO candidate protocol: evaluate over the full candidate
        # set with the default maxDets ladder instead of a truncated one.
        coco_eval.params.maxDets = [1, 10, 100]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        payload[f"{metric}/AP"] = float(coco_eval.stats[0])
        payload[f"{metric}/AP50"] = float(coco_eval.stats[1])
        payload[f"{metric}/AP75"] = float(coco_eval.stats[2])
    return payload


def build_benchmark_payload(
    latencies_ms: list[float],
    device: torch.device,
    *,
    scope: str = "backbone_forward",
) -> dict[str, Any]:
    values = np.asarray(latencies_ms, dtype=np.float32)
    if values.size == 0:
        return {"device": device.type, "scope": str(scope), "images": 0, "latency_ms_mean": 0.0, "latency_ms_p50": 0.0, "latency_ms_p90": 0.0}
    return {
        "device": device.type,
        "scope": str(scope),
        "images": int(values.size),
        "latency_ms_mean": float(values.mean()),
        "latency_ms_p50": float(np.quantile(values, 0.50)),
        "latency_ms_p90": float(np.quantile(values, 0.90)),
    }


def build_device(device_name: str) -> torch.device:
    requested = str(device_name or "cpu").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2,
                    ensure_ascii=False) + "\n", encoding="utf-8")
