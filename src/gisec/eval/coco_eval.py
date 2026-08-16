from __future__ import annotations

from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_json(ann_file: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
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
    coco_dt = coco_gt.loadRes(results)
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
