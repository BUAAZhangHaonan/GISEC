"""COCO segm evaluation on the 32254 val split (3276 images).

Same protocol as the GISEC eval path: score threshold 0.05, mask
threshold 0.5, pycocotools COCOeval on the 'segm' task (and bbox for
reference). Writes metrics.json + coco_instances_results.json.
--limit restricts both prediction and the COCOeval imgIds set, so a
subset is scored against subset GT only.

--calibrate sweeps mask_thr x score_thr over the same split (combine
with --limit for a small calibration subset; inference reruns per
combination) and reports the best combination by segm AP into
calibration.json. Without the flag, behaviour and outputs are
unchanged.

Memory discipline (per-image lazy, RLE-only intermediates, del after
use): predictions are RLE-encoded inside the inference loop and
appended to json_results as COCO dicts; no raw mask arrays survive
past their image iteration, so RAM stays flat across the 3276 images.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from build_models import build_model
from common import DATA, Baseline16mDataset, collate_m2f, collate_mrcnn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

SCORE_THRESHOLD = 0.05
MASK_THRESHOLD = 0.5

CALIBRATE_MASK_THRS = (0.3, 0.4, 0.5, 0.6, 0.7)
CALIBRATE_SCORE_THRS = (0.03, 0.05, 0.1, 0.2)


def masks_to_rle(masks: np.ndarray) -> list[dict]:
    rles = [
        mask_util.encode(np.asfortranarray(masks[i]).astype(np.uint8))
        for i in range(masks.shape[0])
    ]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")
    return rles


def foreground_keep(
    labels: torch.Tensor, scores: torch.Tensor, score_thr: float
) -> torch.Tensor:
    """Single-class decode rule: class 0 is the only foreground class
    (num_labels=1); every other index is the null class and is dropped."""
    return (labels == 0) & (scores > score_thr)


def predict_mrcnn(
    model,
    loader,
    device,
    emit,
    score_thr: float = SCORE_THRESHOLD,
    mask_thr: float = MASK_THRESHOLD,
):
    model.eval()
    with torch.no_grad():
        for images, _targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out in outputs:
                keep = out["scores"] > score_thr
                scores = out["scores"][keep].cpu().numpy()
                masks = (out["masks"][keep, 0] > mask_thr).cpu().numpy()
                emit(scores, masks)
                del masks


def predict_m2f(
    model,
    loader,
    device,
    emit,
    target_size=(1024, 1024),
    score_thr: float = SCORE_THRESHOLD,
    mask_thr: float = MASK_THRESHOLD,
):
    """Mask2Former decode: per-query softmax class score, keep class 0
    only (null dropped), sigmoid mask upsampled to 1024, mask threshold."""
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
                keep = foreground_keep(labels[i], scores[i], score_thr)
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
                masks = (masks > mask_thr).to(torch.uint8).cpu().numpy()
                emit(scores[i][idx].cpu().numpy(), masks)
                del masks


def run_predict(model, family, loader, device, emit, score_thr, mask_thr):
    if family.startswith("mrcnn16"):
        predict_mrcnn(model, loader, device, emit, score_thr, mask_thr)
    else:
        predict_m2f(model, loader, device, emit, score_thr=score_thr, mask_thr=mask_thr)


def make_emitter(
    image_ids: list[int], json_results: list[dict], category_id: int
) -> Callable[[np.ndarray, np.ndarray], None]:
    state = {"n_done": 0}

    def emit(scores: np.ndarray, masks: np.ndarray) -> None:
        image_id = image_ids[state["n_done"]]
        state["n_done"] += 1
        if masks.shape[0] == 0:
            return
        for score, rle in zip(scores, masks_to_rle(masks), strict=True):
            json_results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "score": float(score),
                }
            )

    return emit


def run_coco_eval(
    coco_gt: COCO,
    json_results: list[dict],
    image_ids: list[int],
    tasks=("segm", "bbox"),
) -> dict:
    if not json_results:
        return {f"{task}/AP": 0.0 for task in tasks}
    metrics: dict = {}
    for task in tasks:
        # loadRes mutates the records it receives (adds bbox/area), so
        # hand it a fresh shallow copy per task.
        coco_dt = coco_gt.loadRes([dict(record) for record in json_results])
        evaler = COCOeval(coco_gt, coco_dt, task)
        evaler.params.imgIds = image_ids  # score only the evaluated subset
        evaler.evaluate()
        evaler.accumulate()
        evaler.summarize()
        stats = evaler.stats
        metrics[f"{task}/AP"] = float(stats[0])
        metrics[f"{task}/AP50"] = float(stats[1])
        metrics[f"{task}/AP75"] = float(stats[2])
        metrics[f"{task}/APs"] = float(stats[3])
        metrics[f"{task}/APm"] = float(stats[4])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        required=True,
        choices=["mrcnn16", "mrcnn16d", "m2f16", "m2f16cat", "m2f16fix"],
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "sweep mask_thr x score_thr and report the best segm AP; "
            "inference reruns per combination, so pass --limit for a "
            "small calibration subset"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    model = build_model(args.family)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:  # resume_last.pth training payload
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)

    include_depth = args.family in ("m2f16cat", "mrcnn16d")
    dataset = Baseline16mDataset(
        "val",
        include_depth=include_depth,
        imagenet_norm=args.family == "m2f16fix",
    )
    if args.limit:
        dataset.image_ids = dataset.image_ids[: args.limit]
    collate = collate_m2f if not args.family.startswith("mrcnn16") else collate_mrcnn
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
    )

    gt_path = DATA / "annotations" / "instances_val.json"
    coco_gt = COCO(str(gt_path))
    category_id = int(dataset.coco.categories[0]["id"])
    image_ids = [int(i) for i in dataset.image_ids]

    if args.calibrate:
        if not args.limit:
            print(
                "warning: --calibrate without --limit reruns full-val "
                "inference 20 times",
                flush=True,
            )
        sweep = []
        best = None
        for score_thr in CALIBRATE_SCORE_THRS:
            for mask_thr in CALIBRATE_MASK_THRS:
                json_results = []
                emit = make_emitter(image_ids, json_results, category_id)
                run_predict(
                    model, args.family, loader, device, emit, score_thr, mask_thr
                )
                combo = run_coco_eval(coco_gt, json_results, image_ids, tasks=("segm",))
                combo.update(
                    {
                        "score_thr": score_thr,
                        "mask_thr": mask_thr,
                        "num_images": len(image_ids),
                    }
                )
                sweep.append(combo)
                print(json.dumps(combo), flush=True)
                if best is None or combo["segm/AP"] > best["segm/AP"]:
                    best = combo
        report = {
            "family": args.family,
            "num_images": len(image_ids),
            "best": best,
            "sweep": sweep,
        }
        with (out_dir / "calibration.json").open("w") as handle:
            json.dump(report, handle, indent=2)
        print(f"best: {json.dumps(best)}")
        return

    json_results = []
    emit = make_emitter(image_ids, json_results, category_id)
    t0 = time.time()
    run_predict(
        model, args.family, loader, device, emit, SCORE_THRESHOLD, MASK_THRESHOLD
    )
    print(
        f"[{time.time() - t0:.0f}s] predicted {len(image_ids)} images, "
        f"{len(json_results)} RLE instances",
        flush=True,
    )

    results_path = out_dir / "coco_instances_results.json"
    with results_path.open("w") as handle:
        json.dump(json_results, handle)

    metrics: dict = {"family": args.family, "num_images": len(image_ids)}
    metrics.update(run_coco_eval(coco_gt, json_results, image_ids))
    metrics["eval_sec"] = round(time.time() - t0, 1)
    with (out_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
