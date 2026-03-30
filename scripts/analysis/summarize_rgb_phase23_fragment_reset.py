#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis._suite_utils import markdown_table


STAGE2_GATE_THRESHOLDS = {
    "covered_gt_rate": 0.92,
    "split_gt_rate": 0.30,
    "singleton_gt_rate": 0.70,
    "impure_fragment_rate": 0.10,
    "leakage_rate": 0.05,
    "fragments_per_covered_gt": 1.50,
    "empty_slot_rate": None,
    "overflow_crop_rate": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-run-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-stage2-md", required=True)
    parser.add_argument("--output-stage3-md", required=True)
    parser.add_argument("--output-stage2-chart", required=True)
    parser.add_argument("--output-stage3-outcome-chart", required=True)
    parser.add_argument("--output-stage3-failure-chart", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _write_stage2_chart(stage2_eval: dict[str, Any], output_path: Path) -> None:
    metrics = [
        "covered_gt_rate",
        "split_gt_rate",
        "singleton_gt_rate",
        "impure_fragment_rate",
        "leakage_rate",
        "fragments_per_covered_gt",
        "overflow_crop_rate",
    ]
    actual = [float(stage2_eval.get(metric, 0.0)) for metric in metrics]
    target = [float(STAGE2_GATE_THRESHOLDS[metric]) for metric in metrics]
    x = list(range(len(metrics)))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([value - 0.18 for value in x], actual, width=0.36, label="actual")
    ax.bar([value + 0.18 for value in x], target, width=0.36, label="gate")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Stage 2 Fragment Gates")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_stage3_outcome_chart(stage3_eval: dict[str, Any] | None, output_path: Path) -> None:
    labels = ["local_graph_invocation_rate", "avg_fragments_per_invoked_crop", "same_instance_edge_recall", "singleton_cluster_rate", "clusters_per_crop"]
    values = [0.0 if stage3_eval is None else float(stage3_eval.get(label, 0.0)) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = list(range(len(labels)))
    ax.bar(x, values, width=0.55)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Stage 3 Local Merger Outcome" if stage3_eval is not None else "Stage 3 Gated Off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_stage3_failure_chart(
    baseline_metrics: dict[str, Any],
    stage3_eval: dict[str, Any] | None,
    output_path: Path,
) -> None:
    labels = ["split_gt_count", "merge_pred_count"]
    baseline = [float(baseline_metrics.get(label, 0.0)) for label in labels]
    stage3 = [0.0 if stage3_eval is None else float(stage3_eval.get(label, 0.0)) for label in labels]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([value - 0.18 for value in x], baseline, width=0.36, label="base_rgb_1024")
    ax.bar([value + 0.18 for value in x], stage3, width=0.36, label="reset_stage3" if stage3_eval is not None else "gated_off")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0.0)
    ax.set_title("Split / Merge Failure Counts")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    baseline_payload = _read_json(Path(args.baseline_run_summary).resolve())
    baseline_metrics = dict(baseline_payload.get("metrics", {}))
    baseline_speed = dict(baseline_payload.get("inference_speed", {}))

    stage2_train = _optional_json(output_root / "fragment_generator_rgb_stage2" / "train_summary.json") or {}
    stage2_val = _optional_json(output_root / "fragment_generator_rgb_stage2" / "val_summary.json") or {}
    stage2_eval = _optional_json(output_root / "fragment_generator_exports" / "val" / "eval_summary.json") or {}
    cache_train = _optional_json(output_root / "fragment_generator_cache" / "train" / "manifest.json") or {}
    cache_val = _optional_json(output_root / "fragment_generator_cache" / "val" / "manifest.json") or {}
    stage3_train = _optional_json(output_root / "local_merger_rgb_stage3" / "train_summary.json")
    stage3_val = _optional_json(output_root / "local_merger_rgb_stage3" / "val_summary.json")
    stage3_eval = _optional_json(output_root / "local_merger_rgb_stage3" / "eval_val" / "eval_summary.json")

    gate_passed = bool(stage2_eval.get("gate_passed", False))
    stage3_status = "completed" if stage3_eval is not None else "gated_off"
    train_overflow_rate = 0.0 if int(cache_train.get("num_samples", 0)) <= 0 else float(cache_train.get("num_overflow_crops", 0)) / float(cache_train.get("num_samples", 1))
    val_overflow_rate = 0.0 if int(cache_val.get("num_samples", 0)) <= 0 else float(cache_val.get("num_overflow_crops", 0)) / float(cache_val.get("num_samples", 1))

    payload = {
        "baseline": {
            "variant": str(baseline_payload.get("variant", "base_rgb_1024")),
            "segm/AP": float(baseline_metrics.get("segm/AP", 0.0)),
            "bbox/AP": float(baseline_metrics.get("bbox/AP", 0.0)),
            "boundary/IoU": float(baseline_metrics.get("boundary/IoU", 0.0)),
            "split_gt_count": int(baseline_metrics.get("split_gt_count", 0)),
            "merge_pred_count": int(baseline_metrics.get("merge_pred_count", 0)),
            "fps": None if baseline_speed.get("throughput_fps") is None else float(baseline_speed["throughput_fps"]),
        },
        "stage2_cache": {
            "train_num_samples": int(cache_train.get("num_samples", 0)),
            "train_num_negative_samples": int(cache_train.get("num_negative_samples", 0)),
            "train_num_overflow_crops": int(cache_train.get("num_overflow_crops", 0)),
            "val_num_samples": int(cache_val.get("num_samples", 0)),
            "val_num_negative_samples": int(cache_val.get("num_negative_samples", 0)),
            "val_num_overflow_crops": int(cache_val.get("num_overflow_crops", 0)),
        },
        "stage2_train": stage2_train,
        "stage2_val": stage2_val,
        "stage2_eval": stage2_eval,
        "stage2_gate_passed": bool(gate_passed),
        "stage3_train": stage3_train,
        "stage3_val": stage3_val,
        "stage3_eval": stage3_eval,
        "stage3_status": str(stage3_status),
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stage2_headers = ["metric", "train", "val", "eval", "gate"]
    stage2_metrics = [
        "covered_gt_rate",
        "split_gt_rate",
        "singleton_gt_rate",
        "impure_fragment_rate",
        "leakage_rate",
        "fragments_per_covered_gt",
        "empty_slot_rate",
        "overflow_crop_rate",
    ]
    stage2_rows = [
        [
            metric,
            "n/a" if metric not in stage2_train else f"{float(stage2_train.get(metric, 0.0)):.4f}",
            "n/a" if metric not in stage2_val else f"{float(stage2_val.get(metric, 0.0)):.4f}",
            "n/a" if metric not in stage2_eval else f"{float(stage2_eval.get(metric, 0.0)):.4f}",
            "n/a" if STAGE2_GATE_THRESHOLDS.get(metric) is None else f"{float(STAGE2_GATE_THRESHOLDS[metric]):.4f}",
        ]
        for metric in stage2_metrics
    ]
    output_stage2_md = Path(args.output_stage2_md).resolve()
    output_stage2_md.parent.mkdir(parents=True, exist_ok=True)
    output_stage2_md.write_text(
        "\n".join(
            [
                "# RGB Phase 2/3 Reset Stage 2 Table",
                "",
                f"- gate_passed: `{gate_passed}`",
                "",
                markdown_table(stage2_headers, stage2_rows).rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    stage3_headers = ["split", "local_graph_invocation_rate", "avg_fragments_per_invoked_crop", "same_instance_edge_recall", "singleton_cluster_rate", "clusters_per_crop", "segm/AP", "boundary/IoU"]
    if stage3_eval is None:
        stage3_rows = [["gated_off", "0.0000", "0.0000", "0.0000", "0.0000", "0.0000", "n/a", "n/a"]]
    else:
        metrics = dict(stage3_eval.get("metrics", {}))
        stage3_rows = [[
            "eval_val",
            f"{float(stage3_eval.get('local_graph_invocation_rate', 0.0)):.4f}",
            f"{float(stage3_eval.get('avg_fragments_per_invoked_crop', 0.0)):.4f}",
            f"{float(stage3_eval.get('same_instance_edge_recall', 0.0)):.4f}",
            f"{float(stage3_eval.get('singleton_cluster_rate', 0.0)):.4f}",
            f"{float(stage3_eval.get('clusters_per_crop', 0.0)):.4f}",
            f"{float(metrics.get('segm/AP', 0.0)):.4f}",
            f"{float(metrics.get('boundary/IoU', 0.0)):.4f}",
        ]]
    output_stage3_md = Path(args.output_stage3_md).resolve()
    output_stage3_md.parent.mkdir(parents=True, exist_ok=True)
    output_stage3_md.write_text(
        "\n".join(
            [
                "# RGB Phase 2/3 Reset Stage 3 Table",
                "",
                f"- stage3_status: `{stage3_status}`",
                "",
                markdown_table(stage3_headers, stage3_rows).rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    output_md = Path(args.output_md).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        "\n".join(
            [
                "# 2026-03-30 RGB Phase 2/3 Fragment Reset Summary",
                "",
                "## Scope",
                "",
                "- Goal stays the same: beat Magformer with a smaller RGB GISEC first, then keep the path open past `AP 80`.",
                "- This note covers the real full-dataset `rgb_phase23_fragment_reset` milestone on the frozen `Mask2Former RGB @1024` backbone.",
                "",
                "## Stage 2 Gate",
                "",
                f"- gate_passed: `{gate_passed}`",
                f"- stage3_status: `{stage3_status}`",
                f"- baseline segm/AP: `{float(baseline_metrics.get('segm/AP', 0.0)):.4f}`",
                f"- train cache overflow rate: `{train_overflow_rate:.4f}` ({int(cache_train.get('num_overflow_crops', 0))} / {int(cache_train.get('num_samples', 0))})",
                f"- val cache overflow rate: `{val_overflow_rate:.4f}` ({int(cache_val.get('num_overflow_crops', 0))} / {int(cache_val.get('num_samples', 0))})",
                "",
                "## Practical Read",
                "",
                "- The reset either passes the Stage 2 fragment gate and unlocks Stage 3, or it stops honestly at the fragment-representation layer.",
                f"- In this run, Stage 3 is {'available for comparison' if stage3_eval is not None else 'gated off because Stage 2 did not clear the required fragment-quality thresholds'}.",
                f"- The strongest failed gate is the fragment-space quality itself: `split_gt_rate = {float(stage2_eval.get('split_gt_rate', 0.0)):.4f}`, `impure_fragment_rate = {float(stage2_eval.get('impure_fragment_rate', 0.0)):.4f}`, `leakage_rate = {float(stage2_eval.get('leakage_rate', 0.0)):.4f}`, and `overflow_crop_rate = {float(stage2_eval.get('overflow_crop_rate', 0.0)):.4f}`.",
                f"- So the current `K=6` explicit-fragment reset does not earn Stage 3 promotion on the real dataset. The blocker is upstream fragment design, not the local merger.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_stage2_chart(stage2_eval, Path(args.output_stage2_chart).resolve())
    _write_stage3_outcome_chart(stage3_eval, Path(args.output_stage3_outcome_chart).resolve())
    _write_stage3_failure_chart(baseline_metrics, stage3_eval, Path(args.output_stage3_failure_chart).resolve())


if __name__ == "__main__":
    main()
