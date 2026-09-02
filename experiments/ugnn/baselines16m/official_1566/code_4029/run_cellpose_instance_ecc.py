#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from cellpose_instance_models import CELLPOSE_TARGET_CACHE_VERSION, precompute_cellpose_target_cache, run_experiment


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--max-train-steps", type=int, default=0)
    ap.add_argument("--max-val-images", type=int, default=0)
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="val")
    ap.add_argument("--target-cache-dir", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--inference-batch", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--precompute-targets-only", action="store_true")
    ap.add_argument("--precompute-split", type=str, default="train")
    ap.add_argument("--max-precompute-images", type=int, default=0)
    args = ap.parse_args()
    if int(args.image_size) not in {512, 1024}:
        raise ValueError("--image-size must be one of {512, 1024}")
    return args


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.precompute_targets_only:
        cache_dir = args.target_cache_dir or output_dir / "target_cache" / f"{args.image_size}_{CELLPOSE_TARGET_CACHE_VERSION}"
        summary = precompute_cellpose_target_cache(
            dataset_root=args.dataset_root,
            split=args.precompute_split,
            image_size=args.image_size,
            target_cache_dir=cache_dir,
            num_workers=args.num_workers,
            max_images=args.max_precompute_images,
        )
        (output_dir / "target_cache_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[cellpose-precompute] {summary}", flush=True)
        return
    run_experiment(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        num_workers=args.num_workers,
        min_area=args.min_area,
        device=args.device,
        max_train_steps=args.max_train_steps,
        max_val_images=args.max_val_images,
        train_split=args.train_split,
        val_split=args.val_split,
        target_cache_dir=args.target_cache_dir,
        log_every=args.log_every,
        inference_batch_size=args.inference_batch,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
