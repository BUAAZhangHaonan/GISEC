"""COCO segm evaluation on the 32254 val split (3276 images).

Same protocol as the GISEC eval path: score threshold 0.05, mask
threshold 0.5, pycocotools COCOeval on the 'segm' task (and bbox for
reference). Writes metrics.json + coco_instances_results.json.

Memory discipline (per-image lazy, RLE-only intermediates, del after
use): predictions are RLE-encoded inside the inference loop and
appended to json_results as COCO dicts; no raw mask arrays survive
past their image iteration, so RAM stays flat across the 3276 images.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from build_models import build_model
from common import Baseline16mDataset, collate_m2f, collate_mrcnn

SCORE_THRESHOLD = 0.05
MASK_THRESHOLD = 0.5


def masks_to_rle(masks: np.ndarray) -> list[dict]:
    rles = [
        mask_util.encode(np.asfortranarray(masks[i]).astype(np.uint8))
        for i in range(masks.shape[0])
    ]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")
    return rles


def predict_mrcnn(model, loader, device, emit):
    model.eval()
    with torch.no_grad():
        for images, _targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out in outputs:
                keep = out["scores"] > SCORE_THRESHOLD
                scores = out["scores"][keep].cpu().numpy()
                masks = (out["masks"][keep, 0] > MASK_THRESHOLD).cpu().numpy()
                emit(scores, masks)
                del masks


def predict_m2f(model, loader, device, emit, target_size=(1024, 1024)):
    """Mask2Former decode: per-query softmax class score (null class
    dropped), sigmoid mask upsampled to 1024, mask threshold 0.5."""
    model.eval()
    with torch.no_grad():
        for pixel_values, _pm, _ml, _cl in loader:
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            class_logits = outputs.class_queries_logits  # (B, Q, C+1)
            mask_logits = outputs.masks_queries_logits  # (B, Q, h, w)
            class_probs = F.softmax(class_logits, dim=-1)
            scores, labels = class_probs.max(-1)  # (B, Q)
            for i in range(pixel_values.shape[0]):
                keep = (labels[i] < class_probs.shape[-1] - 1) & (
                    scores[i] > SCORE_THRESHOLD
                )
                idx = torch.nonzero(keep).flatten()
                if idx.numel() == 0:
                    emit(np.zeros(0, np.float32), np.zeros((0, *target_size), np.uint8))
                    continue
                masks = F.interpolate(
                    mask_logits[i : i + 1, idx].sigmoid(),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )[0]
                masks = (masks > MASK_THRESHOLD).to(torch.uint8).cpu().numpy()
                emit(scores[i][idx].cpu().numpy(), masks)
                del masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", required=True, choices=["mrcnn16", "m2f16", "m2f16cat"]
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    model = build_model(args.family)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    include_depth = args.family == "m2f16cat"
    dataset = Baseline16mDataset("val", include_depth=include_depth)
    if args.limit:
        dataset.image_ids = dataset.image_ids[: args.limit]
    collate = collate_m2f if args.family != "mrcnn16" else collate_mrcnn
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
    )

    gt_path = Path(__file__).resolve().parents[3] / (
        "datasets/20260318_1K_32254/annotations/instances_val.json"
    )
    coco_gt = COCO(str(gt_path))
    category_id = int(dataset.coco.categories[0]["id"])
    image_ids = [int(i) for i in dataset.image_ids]

    json_results = []
    n_done = 0

    def emit(scores: np.ndarray, masks: np.ndarray) -> None:
        nonlocal n_done
        image_id = image_ids[n_done]
        n_done += 1
        if masks.shape[0] == 0:
            return
        for score, rle in zip(scores, masks_to_rle(masks)):
            json_results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "score": float(score),
                }
            )

    t0 = time.time()
    if args.family == "mrcnn16":
        predict_mrcnn(model, loader, device, emit)
    else:
        predict_m2f(model, loader, device, emit)
    print(
        f"[{time.time() - t0:.0f}s] predicted {n_done} images, "
        f"{len(json_results)} RLE instances",
        flush=True,
    )

    results_path = out_dir / "coco_instances_results.json"
    with results_path.open("w") as handle:
        json.dump(json_results, handle)

    metrics: dict = {"family": args.family, "num_images": len(dataset.image_ids)}
    if json_results:
        for task in ("segm", "bbox"):
            coco_dt = coco_gt.loadRes(str(results_path))
            evaler = COCOeval(coco_gt, coco_dt, task)
            evaler.evaluate()
            evaler.accumulate()
            evaler.summarize()
            stats = evaler.stats
            metrics[f"{task}/AP"] = float(stats[0])
            metrics[f"{task}/AP50"] = float(stats[1])
            metrics[f"{task}/AP75"] = float(stats[2])
            metrics[f"{task}/APs"] = float(stats[3])
            metrics[f"{task}/APm"] = float(stats[4])
    metrics["eval_sec"] = round(time.time() - t0, 1)
    with (out_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
