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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-run-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-cache-md", required=True)
    parser.add_argument("--output-oracle-md", required=True)
    parser.add_argument("--output-fragment-chart", required=True)
    parser.add_argument("--output-oracle-chart", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _int_value(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in payload:
            return int(payload.get(key, 0))
    return 0


def _float_value(payload: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in payload:
            return float(payload.get(key, 0.0))
    return 0.0


def _write_fragment_chart(*, pred_manifest: dict[str, Any], gt_manifest: dict[str, Any], output_path: Path) -> None:
    labels = ["mean", "p50", "p75", "p90", "p95", "max"]
    pred_values = [
        float(pred_manifest.get("raw_fragment_count_mean", 0.0)),
        float(pred_manifest.get("raw_fragment_count_p50", 0.0)),
        float(pred_manifest.get("raw_fragment_count_p75", 0.0)),
        float(pred_manifest.get("raw_fragment_count_p90", 0.0)),
        float(pred_manifest.get("raw_fragment_count_p95", 0.0)),
        float(pred_manifest.get("raw_fragment_count_max", 0.0)),
    ]
    gt_values = [
        float(gt_manifest.get("raw_fragment_count_mean", 0.0)),
        float(gt_manifest.get("raw_fragment_count_p50", 0.0)),
        float(gt_manifest.get("raw_fragment_count_p75", 0.0)),
        float(gt_manifest.get("raw_fragment_count_p90", 0.0)),
        float(gt_manifest.get("raw_fragment_count_p95", 0.0)),
        float(gt_manifest.get("raw_fragment_count_max", 0.0)),
    ]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([value - 0.18 for value in x], pred_values, width=0.36, label="pred anchors")
    ax.bar([value + 0.18 for value in x], gt_values, width=0.36, label="gt anchors")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0.0)
    ax.set_title("Instance-Local Fragment Counts")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_oracle_chart(
    *,
    baseline: dict[str, Any],
    fragments_summary: dict[str, Any],
    owner_summary: dict[str, Any],
    output_path: Path,
) -> None:
    labels = ["segm/AP", "boundary/IoU", "split_gt_count", "merge_pred_count"]
    baseline_values = [
        float(baseline.get("segm/AP", 0.0)),
        float(baseline.get("boundary/IoU", 0.0)),
        float(baseline.get("split_gt_count", 0.0)),
        float(baseline.get("merge_pred_count", 0.0)),
    ]
    fragment_metrics = dict(fragments_summary.get("metrics", {}))
    owner_metrics = dict(owner_summary.get("metrics", {}))
    fragment_values = [
        float(fragment_metrics.get("segm/AP", 0.0)),
        float(fragment_metrics.get("boundary/IoU", 0.0)),
        float(fragments_summary.get("split_gt_count", 0.0)),
        float(fragments_summary.get("merge_pred_count", 0.0)),
    ]
    owner_values = [
        float(owner_metrics.get("segm/AP", 0.0)),
        float(owner_metrics.get("boundary/IoU", 0.0)),
        float(owner_summary.get("split_gt_count", 0.0)),
        float(owner_summary.get("merge_pred_count", 0.0)),
    ]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([value - 0.24 for value in x], baseline_values, width=0.24, label="base_rgb_1024")
    ax.bar(x, fragment_values, width=0.24, label="oracle_fragments_no_merge")
    ax.bar([value + 0.24 for value in x], owner_values, width=0.24, label="oracle_owner_union")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0.0)
    ax.set_title("Instance-Local Oracle Comparison")
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
    pred_manifest = _optional_json(output_root / "instance_fragment_cache_pred" / "val" / "manifest.json")
    gt_manifest = _optional_json(output_root / "instance_fragment_cache_gt" / "val" / "manifest.json")
    fragments_summary = _optional_json(output_root / "instance_fragment_oracles" / "val" / "oracle_fragments_no_merge" / "eval_summary.json")
    owner_summary = _optional_json(output_root / "instance_fragment_oracles" / "val" / "oracle_owner_union" / "eval_summary.json")

    owner_metrics = dict(owner_summary.get("metrics", {}))
    oracle_gate_passed = bool(
        float(owner_metrics.get("segm/AP", 0.0)) >= float(baseline_metrics.get("segm/AP", 0.0)) + 0.02
        and float(owner_summary.get("split_gt_count", 0.0)) < float(baseline_metrics.get("split_gt_count", 0.0))
        and float(owner_summary.get("merge_pred_count", 0.0)) < float(baseline_metrics.get("merge_pred_count", 0.0))
    )
    payload = {
        "baseline": {
            "variant": str(baseline_payload.get("variant", "base_rgb_1024")),
            "segm/AP": float(baseline_metrics.get("segm/AP", 0.0)),
            "bbox/AP": float(baseline_metrics.get("bbox/AP", 0.0)),
            "boundary/IoU": float(baseline_metrics.get("boundary/IoU", 0.0)),
            "split_gt_count": int(baseline_metrics.get("split_gt_count", 0)),
            "merge_pred_count": int(baseline_metrics.get("merge_pred_count", 0)),
        },
        "cache_pred": pred_manifest,
        "cache_gt": gt_manifest,
        "oracles": {
            "oracle_fragments_no_merge": fragments_summary,
            "oracle_owner_union": owner_summary,
        },
        "decision": {
            "oracle_gate_passed": bool(oracle_gate_passed),
            "reason": (
                "oracle_owner_union clears the AP and failure-count gate"
                if bool(oracle_gate_passed)
                else "oracle_owner_union does not yet justify Stage 2 model training"
            ),
        },
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cache_headers = ["cache", "samples", "positive", "negative", "matchable_gt_rate", "mean", "p50", "p75", "p90", "p95", "max"]
    cache_rows = [
        [
            "pred",
            str(int(pred_manifest.get("num_samples", 0))),
            str(_int_value(pred_manifest, "positive_anchor_count", "num_positive_samples")),
            str(_int_value(pred_manifest, "negative_anchor_count", "num_negative_samples")),
            f"{_float_value(pred_manifest, 'matchable_gt_rate'):.4f}",
            f"{_float_value(pred_manifest, 'raw_fragment_count_mean'):.4f}",
            f"{_float_value(pred_manifest, 'raw_fragment_count_p50'):.4f}",
            f"{_float_value(pred_manifest, 'raw_fragment_count_p75'):.4f}",
            f"{_float_value(pred_manifest, 'raw_fragment_count_p90'):.4f}",
            f"{_float_value(pred_manifest, 'raw_fragment_count_p95'):.4f}",
            str(int(pred_manifest.get("raw_fragment_count_max", 0))),
        ],
        [
            "gt",
            str(int(gt_manifest.get("num_samples", 0))),
            "-",
            "-",
            "-",
            f"{float(gt_manifest.get('raw_fragment_count_mean', 0.0)):.4f}",
            f"{float(gt_manifest.get('raw_fragment_count_p50', 0.0)):.4f}",
            f"{float(gt_manifest.get('raw_fragment_count_p75', 0.0)):.4f}",
            f"{float(gt_manifest.get('raw_fragment_count_p90', 0.0)):.4f}",
            f"{float(gt_manifest.get('raw_fragment_count_p95', 0.0)):.4f}",
            str(int(gt_manifest.get("raw_fragment_count_max", 0))),
        ],
    ]
    output_cache_md = Path(args.output_cache_md).resolve()
    output_cache_md.parent.mkdir(parents=True, exist_ok=True)
    output_cache_md.write_text(
        "\n".join(
            [
                "# RGB Phase 2/3 Instance-Local Reset Cache Table",
                "",
                markdown_table(cache_headers, cache_rows).rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    oracle_headers = ["oracle", "segm/AP", "boundary/IoU", "split_gt_count", "merge_pred_count"]
    output_oracle_md = Path(args.output_oracle_md).resolve()
    output_oracle_md.parent.mkdir(parents=True, exist_ok=True)
    output_oracle_md.write_text(
        "\n".join(
            [
                "# RGB Phase 2/3 Instance-Local Reset Oracle Table",
                "",
                markdown_table(
                    oracle_headers,
                    [
                        [
                            "base_rgb_1024",
                            f"{float(baseline_metrics.get('segm/AP', 0.0)):.4f}",
                            f"{float(baseline_metrics.get('boundary/IoU', 0.0)):.4f}",
                            str(int(baseline_metrics.get("split_gt_count", 0))),
                            str(int(baseline_metrics.get("merge_pred_count", 0))),
                        ],
                        [
                            "oracle_fragments_no_merge",
                            f"{float(dict(fragments_summary.get('metrics', {})).get('segm/AP', 0.0)):.4f}",
                            f"{float(dict(fragments_summary.get('metrics', {})).get('boundary/IoU', 0.0)):.4f}",
                            str(int(fragments_summary.get("split_gt_count", 0))),
                            str(int(fragments_summary.get("merge_pred_count", 0))),
                        ],
                        [
                            "oracle_owner_union",
                            f"{float(owner_metrics.get('segm/AP', 0.0)):.4f}",
                            f"{float(owner_metrics.get('boundary/IoU', 0.0)):.4f}",
                            str(int(owner_summary.get("split_gt_count", 0))),
                            str(int(owner_summary.get("merge_pred_count", 0))),
                        ],
                    ],
                ).rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_fragment_chart(pred_manifest=pred_manifest, gt_manifest=gt_manifest, output_path=Path(args.output_fragment_chart).resolve())
    _write_oracle_chart(
        baseline={
            "segm/AP": float(baseline_metrics.get("segm/AP", 0.0)),
            "boundary/IoU": float(baseline_metrics.get("boundary/IoU", 0.0)),
            "split_gt_count": int(baseline_metrics.get("split_gt_count", 0)),
            "merge_pred_count": int(baseline_metrics.get("merge_pred_count", 0)),
        },
        fragments_summary=fragments_summary,
        owner_summary=owner_summary,
        output_path=Path(args.output_oracle_chart).resolve(),
    )

    output_md = Path(args.output_md).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        "\n".join(
            [
                "# 2026-03-30 RGB Phase 2/3 Instance-Local Reset Summary",
                "",
                "## Result",
                "",
                f"- oracle_gate_passed: `{bool(oracle_gate_passed)}`",
                f"- decision: `{payload['decision']['reason']}`",
                "",
                "## Cache Read",
                "",
                f"- pred positive anchors: `{_int_value(pred_manifest, 'positive_anchor_count', 'num_positive_samples')}`",
                f"- pred negative anchors: `{_int_value(pred_manifest, 'negative_anchor_count', 'num_negative_samples')}`",
                f"- pred raw fragment p95: `{_float_value(pred_manifest, 'raw_fragment_count_p95'):.4f}`",
                f"- pred raw fragment max: `{int(pred_manifest.get('raw_fragment_count_max', 0))}`",
                "",
                "## Oracle Read",
                "",
                f"- baseline segm/AP: `{float(baseline_metrics.get('segm/AP', 0.0)):.4f}`",
                f"- oracle_fragments_no_merge segm/AP: `{float(dict(fragments_summary.get('metrics', {})).get('segm/AP', 0.0)):.4f}`",
                f"- oracle_owner_union segm/AP: `{float(owner_metrics.get('segm/AP', 0.0)):.4f}`",
                f"- oracle_owner_union split_gt_count: `{int(owner_summary.get('split_gt_count', 0))}`",
                f"- oracle_owner_union merge_pred_count: `{int(owner_summary.get('merge_pred_count', 0))}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
