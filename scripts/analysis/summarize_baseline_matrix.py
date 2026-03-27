#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis._suite_utils import markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--dataset-root")
    parser.add_argument("--prep-seconds", type=float, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_scalar(path: Path) -> str:
    if not path.exists():
        return "n/a"
    return path.read_text(encoding="utf-8").strip() or "n/a"


def _read_float(path: Path) -> float | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return float(raw)


def _resolve_fragment_quality(run_dir: Path, payload: dict[str, Any]) -> dict[str, float | int | None]:
    empty = {
        "fragment_count": None,
        "pair_count": None,
        "contact_pair_count": None,
        "bridge_pair_count": None,
        "fragment_purity_mean": None,
        "fragment_purity_median": None,
        "same_instance_total_pairs": None,
        "same_instance_recalled_pairs": None,
        "same_instance_recall": None,
    }
    fragment_quality = payload.get("fragment_quality")
    if isinstance(fragment_quality, dict):
        resolved = dict(empty)
        resolved.update(fragment_quality)
        return resolved
    summary_path = run_dir / "fragment_quality_summary.json"
    if not summary_path.exists():
        return empty
    resolved = dict(empty)
    resolved.update(_read_json(summary_path))
    return resolved


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0.0 else float(num) / float(den)


def _resolve_dataset_root(payload: dict[str, Any], cli_dataset_root: str | None) -> Path | None:
    if cli_dataset_root:
        return Path(cli_dataset_root).resolve()
    dataset_root = payload.get("dataset_root")
    if dataset_root:
        return Path(str(dataset_root)).resolve()
    return None


def _compute_f1_at_50(annotation_path: Path, results_path: Path) -> dict[str, float]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not results:
        return {"P@50": 0.0, "R@50": 0.0, "F1@50": 0.0}
    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(str(results_path))
    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.params.iouThrs = [0.5]
    evaluator.evaluate()
    evaluator.accumulate()
    tp = 0
    fp = 0
    fn = 0
    for eval_img in evaluator.evalImgs:
        if eval_img is None:
            continue
        dt_matches = eval_img["dtMatches"]
        dt_ignore = eval_img["dtIgnore"]
        gt_matches = eval_img["gtMatches"]
        gt_ignore = eval_img["gtIgnore"]
        if dt_matches.size:
            tp += int(((dt_matches[0] > 0) & (~dt_ignore[0].astype(bool))).sum())
            fp += int(((dt_matches[0] == 0) & (~dt_ignore[0].astype(bool))).sum())
        if gt_matches.size:
            fn += int(((gt_matches[0] == 0) & (~gt_ignore.astype(bool))).sum())
        else:
            fn += int((~gt_ignore.astype(bool)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {"P@50": precision, "R@50": recall, "F1@50": f1}


def _compute_pathology(annotation_path: Path, results_path: Path) -> dict[str, float]:
    payload = _read_json(annotation_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gt_counts: dict[int, int] = defaultdict(int)
    image_areas: dict[int, float] = {}
    for image in payload.get("images", []):
        image_areas[int(image["id"])] = float(image["width"]) * float(image["height"])
    for ann in payload.get("annotations", []):
        gt_counts[int(ann["image_id"])] += 1
    for row in results:
        by_image[int(row["image_id"])].append(row)

    pred_counts: list[int] = []
    gt_count_values: list[int] = []
    pred_gt_ratios: list[float] = []
    largest_mask_ratios: list[float] = []
    masks_ge_1pct = 0
    masks_ge_5pct = 0
    for image in payload.get("images", []):
        image_id = int(image["id"])
        preds = by_image.get(image_id, [])
        gt_count = int(gt_counts.get(image_id, 0))
        pred_counts.append(len(preds))
        gt_count_values.append(gt_count)
        pred_gt_ratios.append(_safe_div(len(preds), gt_count))
        image_area = float(image_areas[image_id])
        ratios: list[float] = []
        for pred in preds:
            ratio = _safe_div(float(mask_utils.area(pred["segmentation"])), image_area)
            ratios.append(ratio)
            masks_ge_1pct += int(ratio >= 0.01)
            masks_ge_5pct += int(ratio >= 0.05)
        largest_mask_ratios.append(max(ratios) if ratios else 0.0)
    largest_sorted = sorted(largest_mask_ratios)
    p90_index = max(int(len(largest_sorted) * 0.9) - 1, 0)
    return {
        "avg_pred_count": float(mean(pred_counts)),
        "avg_gt_count": float(mean(gt_count_values)),
        "pred_gt_count_ratio": float(mean(pred_gt_ratios)),
        "median_largest_mask_ratio": float(median(largest_mask_ratios)),
        "p90_largest_mask_ratio": float(largest_sorted[p90_index]),
        "masks_ge_1pct": float(masks_ge_1pct),
        "masks_ge_5pct": float(masks_ge_5pct),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_path = Path(args.output).resolve()
    rows: list[dict[str, Any]] = []
    for run_summary_path in sorted(input_root.rglob("run_summary.json")):
        payload = _read_json(run_summary_path)
        run_dir = run_summary_path.parent
        metrics = dict(payload.get("metrics", {}))
        speed = dict(payload.get("inference_speed", {}))
        dataset_root = _resolve_dataset_root(payload, args.dataset_root)
        annotation_path = None if dataset_root is None else dataset_root / "annotations" / "instances_val.json"
        results_path = Path(str(payload.get("results_json", run_dir / "coco_instances_results.json"))).resolve()
        f1_metrics = {"P@50": None, "R@50": None, "F1@50": None}
        pathology = {
            "avg_pred_count": None,
            "avg_gt_count": None,
            "pred_gt_count_ratio": None,
            "median_largest_mask_ratio": None,
            "p90_largest_mask_ratio": None,
            "masks_ge_1pct": None,
            "masks_ge_5pct": None,
        }
        if annotation_path is not None and annotation_path.exists() and results_path.exists():
            f1_metrics = _compute_f1_at_50(annotation_path, results_path)
            pathology = _compute_pathology(annotation_path, results_path)
        timing = dict(payload.get("timing", {}))
        fragment_quality = _resolve_fragment_quality(run_dir, payload)
        rows.append(
            {
                "model": str(payload.get("model", run_dir.name)),
                "variant": str(payload.get("variant", run_dir.name)),
                "modality": str(payload.get("modality", "unknown")),
                "boundary_iou": metrics.get("boundary/IoU"),
                "segm_ap": float(metrics.get("segm/AP", 0.0)),
                "segm_ap50": float(metrics.get("segm/AP50", 0.0)),
                "segm_ap75": float(metrics.get("segm/AP75", 0.0)),
                "bbox_ap": float(metrics.get("bbox/AP", 0.0)),
                "bbox_ap50": float(metrics.get("bbox/AP50", 0.0)),
                "bbox_ap75": float(metrics.get("bbox/AP75", 0.0)),
                "P@50": f1_metrics["P@50"],
                "R@50": f1_metrics["R@50"],
                "F1@50": f1_metrics["F1@50"],
                "fps": speed.get("throughput_fps"),
                "infer_peak_memory_mb": speed.get("inference_peak_memory_mb"),
                "train_peak_memory_mb": payload.get("training_peak_memory_mb")
                if payload.get("training_peak_memory_mb") is not None
                else _read_float(run_dir / "peak_memory_mb.txt"),
                "params_trainable": payload.get("params_trainable")
                if payload.get("params_trainable") is not None
                else _read_scalar(run_dir / "params_trainable.txt"),
                "prep_offline_sec": timing.get("prep_offline_sec", args.prep_seconds),
                "train_only_sec": timing.get("train_only_sec"),
                "eval_post_sec": timing.get("eval_post_sec"),
                "end_to_end_sec": timing.get("end_to_end_sec", payload.get("wall_time_sec")),
                "path": str(run_dir),
                **fragment_quality,
                **pathology,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No run_summary.json files found under {input_root}")

    rows.sort(key=lambda item: (-item["segm_ap"], item["model"], item["variant"]))
    best = rows[0]
    table_rows = [
        [
            row["model"],
            row["variant"],
            row["modality"],
            "n/a" if row["boundary_iou"] is None else f"{float(row['boundary_iou']):.4f}",
            f"{row['segm_ap']:.4f}",
            f"{row['segm_ap50']:.4f}",
            f"{row['bbox_ap']:.4f}",
            "n/a" if row["F1@50"] is None else f"{float(row['F1@50']):.4f}",
            "n/a" if row["fps"] is None else f"{float(row['fps']):.4f}",
            "n/a" if row["train_peak_memory_mb"] is None else f"{float(row['train_peak_memory_mb']):.4f}",
            "n/a" if row["infer_peak_memory_mb"] is None else f"{float(row['infer_peak_memory_mb']):.4f}",
            str(row["params_trainable"]),
            "n/a" if row["prep_offline_sec"] is None else f"{float(row['prep_offline_sec']):.4f}",
            "n/a" if row["train_only_sec"] is None else f"{float(row['train_only_sec']):.4f}",
            "n/a" if row["eval_post_sec"] is None else f"{float(row['eval_post_sec']):.4f}",
            "n/a" if row["end_to_end_sec"] is None else f"{float(row['end_to_end_sec']):.4f}",
            "n/a" if row["pred_gt_count_ratio"] is None else f"{float(row['pred_gt_count_ratio']):.4f}",
            "n/a" if row["median_largest_mask_ratio"] is None else f"{float(row['median_largest_mask_ratio']):.4f}",
            "n/a" if row["fragment_purity_mean"] is None else f"{float(row['fragment_purity_mean']):.4f}",
            "n/a" if row["fragment_purity_median"] is None else f"{float(row['fragment_purity_median']):.4f}",
            "n/a" if row["same_instance_recall"] is None else f"{float(row['same_instance_recall']):.4f}",
            row["path"],
        ]
        for row in rows
    ]
    markdown = "\n".join(
        [
            "# Baseline Benchmark Matrix",
            "",
            f"- input_root: `{input_root}`",
            f"- num_runs: `{len(rows)}`",
            f"- best_model: `{best['model']}`",
            f"- best_variant: `{best['variant']}`",
            f"- best_segm_ap: `{best['segm_ap']:.4f}`",
            "",
            markdown_table(
                [
                    "Model",
                    "Variant",
                    "Modality",
                    "boundary/IoU",
                    "segm/AP",
                    "segm/AP50",
                    "bbox/AP",
                    "F1@50",
                    "FPS",
                    "Train Peak MB",
                    "Infer Peak MB",
                    "Params",
                    "Prep Sec",
                    "Train Sec",
                    "Eval Sec",
                    "End-to-End Sec",
                    "Pred/GT Ratio",
                    "Largest Mask Ratio",
                    "Fragment Purity Mean",
                    "Fragment Purity Median",
                    "Same-Instance Recall",
                    "Run Dir",
                ],
                table_rows,
            ).rstrip(),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "input_root": str(input_root),
                    "dataset_root": None if args.dataset_root is None else str(Path(args.dataset_root).resolve()),
                    "num_runs": len(rows),
                    "best_model": best["model"],
                    "best_variant": best["variant"],
                    "best_segm_ap": best["segm_ap"],
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
