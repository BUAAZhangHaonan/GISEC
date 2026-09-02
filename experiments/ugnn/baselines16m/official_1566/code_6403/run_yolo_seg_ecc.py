#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from yolo_stats_norm import StatsNormalizedSegmentationTrainer


def _parse_triplet(raw: str) -> list[float]:
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Expected a JSON list of 3 floats, got: {raw}")
    return [float(v) for v in value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", type=str, required=True)
    ap.add_argument("--name", type=str, default="train")
    ap.add_argument("--pretrained", type=str, default="False")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--cos-lr", type=str, default="False")
    ap.add_argument("--plots", type=str, default="False")
    ap.add_argument("--rgb-mean", type=str, required=True)
    ap.add_argument("--rgb-std", type=str, required=True)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.train(
        trainer=StatsNormalizedSegmentationTrainer,
        data=str(args.data),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        epochs=int(args.epochs),
        device=str(args.device),
        workers=int(args.workers),
        project=str(args.project),
        name=str(args.name),
        pretrained=str(args.pretrained).lower() == "true",
        lr0=float(args.lr0),
        warmup_epochs=float(args.warmup_epochs),
        cos_lr=str(args.cos_lr).lower() == "true",
        plots=str(args.plots).lower() == "true",
        rgb_mean=_parse_triplet(args.rgb_mean),
        rgb_std=_parse_triplet(args.rgb_std),
    )


if __name__ == "__main__":
    main()
