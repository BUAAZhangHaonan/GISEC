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

from scripts.analysis._suite_utils import markdown_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-run-summary", required=True)
    parser.add_argument("--oracle-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-table-md", required=True)
    parser.add_argument("--output-chart", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_fragment_gate(learned_summary: dict[str, Any]) -> bool:
    return bool(
        float(learned_summary.get("covered_instance_rate", 0.0)) >= 0.92
        and float(learned_summary.get("split_instance_rate", 0.0)) >= 0.30
        and float(learned_summary.get("impure_fragment_rate", 1.0)) <= 0.10
        and float(learned_summary.get("leakage_rate", 1.0)) <= 0.05
        and float(learned_summary.get("fragments_per_covered_instance", 0.0)) >= 1.5
        and float(learned_summary.get("singleton_instance_rate", 1.0)) <= 0.70
    )


def _stage3_reentry_allowed(
    *,
    learned_summary: dict[str, Any],
    baseline_metrics: dict[str, Any],
    oracle_owner_union: dict[str, Any],
) -> bool:
    learned_owner_ap = float(learned_summary.get("owner_union_segm/AP", 0.0))
    learned_owner_boundary = float(learned_summary.get("owner_union_boundary/IoU", 0.0))
    learned_owner_split = int(learned_summary.get("owner_union_split_gt_count", 0))
    learned_owner_merge = int(learned_summary.get("owner_union_merge_pred_count", 0))
    baseline_ap = float(baseline_metrics.get("segm/AP", 0.0))
    baseline_boundary = float(baseline_metrics.get("boundary/IoU", 0.0))
    baseline_split = int(baseline_metrics.get("split_gt_count", 0))
    baseline_merge = int(baseline_metrics.get("merge_pred_count", 0))
    oracle_ap = float(dict(oracle_owner_union.get("metrics", {})).get("segm/AP", 0.0))
    return bool(
        _bool_fragment_gate(learned_summary)
        and learned_owner_ap > baseline_ap
        and learned_owner_boundary > baseline_boundary
        and learned_owner_split < baseline_split
        and learned_owner_merge < baseline_merge
        and (oracle_ap - learned_owner_ap) > 0.05
        and learned_owner_merge > learned_owner_split
    )


def _write_chart(
    *,
    baseline_metrics: dict[str, Any],
    oracle_fragments: dict[str, Any],
    oracle_owner_union: dict[str, Any],
    learned_summary: dict[str, Any],
    output_path: Path,
) -> None:
    labels = ["segm/AP", "boundary/IoU", "split_gt_count", "merge_pred_count"]
    baseline_values = [
        float(baseline_metrics.get("segm/AP", 0.0)),
        float(baseline_metrics.get("boundary/IoU", 0.0)),
        float(baseline_metrics.get("split_gt_count", 0.0)),
        float(baseline_metrics.get("merge_pred_count", 0.0)),
    ]
    learned_fragment_values = [
        float(learned_summary.get("learned_fragments_no_merge_segm/AP", 0.0)),
        float(learned_summary.get("learned_fragments_no_merge_boundary/IoU", 0.0)),
        float(learned_summary.get("learned_fragments_no_merge_split_gt_count", 0.0)),
        float(learned_summary.get("learned_fragments_no_merge_merge_pred_count", 0.0)),
    ]
    learned_owner_values = [
        float(learned_summary.get("owner_union_segm/AP", 0.0)),
        float(learned_summary.get("owner_union_boundary/IoU", 0.0)),
        float(learned_summary.get("owner_union_split_gt_count", 0.0)),
        float(learned_summary.get("owner_union_merge_pred_count", 0.0)),
    ]
    oracle_owner_values = [
        float(dict(oracle_owner_union.get("metrics", {})).get("segm/AP", 0.0)),
        float(dict(oracle_owner_union.get("metrics", {})).get("boundary/IoU", 0.0)),
        float(oracle_owner_union.get("split_gt_count", 0.0)),
        float(oracle_owner_union.get("merge_pred_count", 0.0)),
    ]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([value - 0.27 for value in x], baseline_values, width=0.18, label="base_rgb_1024")
    ax.bar([value - 0.09 for value in x], learned_fragment_values, width=0.18, label="learned_fragments_no_merge")
    ax.bar([value + 0.09 for value in x], learned_owner_values, width=0.18, label="learned_owner_union")
    ax.bar([value + 0.27 for value in x], oracle_owner_values, width=0.18, label="oracle_owner_union")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0.0)
    ax.set_title("RGB Phase 2/3 Stage 2 Owner-Union Comparison")
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
    oracle_payload = _read_json(Path(args.oracle_summary_json).resolve())
    oracle_fragments = dict(dict(oracle_payload.get("oracles", {})).get("oracle_fragments_no_merge", {}))
    oracle_owner_union = dict(dict(oracle_payload.get("oracles", {})).get("oracle_owner_union", {}))
    learned_summary = _read_json(output_root / "instance_fragment_generator_rgb_stage2" / "eval_val" / "eval_summary.json")

    fragment_gate_passed = _bool_fragment_gate(learned_summary)
    stage3_reentry_allowed = _stage3_reentry_allowed(
        learned_summary=learned_summary,
        baseline_metrics=baseline_metrics,
        oracle_owner_union=oracle_owner_union,
    )
    payload = {
        "baseline": {
            "variant": str(baseline_payload.get("variant", "base_rgb_1024")),
            **{key: baseline_metrics.get(key, 0.0) for key in ["segm/AP", "bbox/AP", "boundary/IoU", "split_gt_count", "merge_pred_count"]},
        },
        "oracles": {
            "fragments_no_merge": {
                "segm/AP": float(dict(oracle_fragments.get("metrics", {})).get("segm/AP", 0.0)),
                "boundary/IoU": float(dict(oracle_fragments.get("metrics", {})).get("boundary/IoU", 0.0)),
                "split_gt_count": int(oracle_fragments.get("split_gt_count", 0)),
                "merge_pred_count": int(oracle_fragments.get("merge_pred_count", 0)),
            },
            "owner_union": {
                "segm/AP": float(dict(oracle_owner_union.get("metrics", {})).get("segm/AP", 0.0)),
                "boundary/IoU": float(dict(oracle_owner_union.get("metrics", {})).get("boundary/IoU", 0.0)),
                "split_gt_count": int(oracle_owner_union.get("split_gt_count", 0)),
                "merge_pred_count": int(oracle_owner_union.get("merge_pred_count", 0)),
            },
        },
        "learned": {
            "fragments_no_merge": {
                "segm/AP": float(learned_summary.get("learned_fragments_no_merge_segm/AP", 0.0)),
                "boundary/IoU": float(learned_summary.get("learned_fragments_no_merge_boundary/IoU", 0.0)),
                "split_gt_count": int(learned_summary.get("learned_fragments_no_merge_split_gt_count", 0)),
                "merge_pred_count": int(learned_summary.get("learned_fragments_no_merge_merge_pred_count", 0)),
            },
            "owner_union": {
                "segm/AP": float(learned_summary.get("owner_union_segm/AP", 0.0)),
                "boundary/IoU": float(learned_summary.get("owner_union_boundary/IoU", 0.0)),
                "split_gt_count": int(learned_summary.get("owner_union_split_gt_count", 0)),
                "merge_pred_count": int(learned_summary.get("owner_union_merge_pred_count", 0)),
            },
            "fragment_quality": {
                key: learned_summary.get(key, 0.0)
                for key in [
                    "covered_instance_rate",
                    "split_instance_rate",
                    "singleton_instance_rate",
                    "impure_fragment_rate",
                    "leakage_rate",
                    "fragments_per_covered_instance",
                    "negative_anchor_empty_precision",
                    "negative_anchor_false_fragment_mean",
                    "query_overflow_rate",
                    "truncated_fragment_total",
                ]
            },
        },
        "decision": {
            "fragment_gate_passed": bool(fragment_gate_passed),
            "stage3_reentry_allowed": bool(stage3_reentry_allowed),
            "reason": (
                "learned owner-union now beats the backbone and still leaves a merge-dominant gap"
                if bool(stage3_reentry_allowed)
                else "keep Stage 3 paused until learned owner-union is strong and still clearly merge-limited"
            ),
        },
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table_headers = ["row", "segm/AP", "boundary/IoU", "split_gt_count", "merge_pred_count"]
    table_rows = [
        [
            "base_rgb_1024",
            f"{float(baseline_metrics.get('segm/AP', 0.0)):.4f}",
            f"{float(baseline_metrics.get('boundary/IoU', 0.0)):.4f}",
            str(int(baseline_metrics.get("split_gt_count", 0))),
            str(int(baseline_metrics.get("merge_pred_count", 0))),
        ],
        [
            "learned_fragments_no_merge",
            f"{float(learned_summary.get('learned_fragments_no_merge_segm/AP', 0.0)):.4f}",
            f"{float(learned_summary.get('learned_fragments_no_merge_boundary/IoU', 0.0)):.4f}",
            str(int(learned_summary.get("learned_fragments_no_merge_split_gt_count", 0))),
            str(int(learned_summary.get("learned_fragments_no_merge_merge_pred_count", 0))),
        ],
        [
            "learned_owner_union",
            f"{float(learned_summary.get('owner_union_segm/AP', 0.0)):.4f}",
            f"{float(learned_summary.get('owner_union_boundary/IoU', 0.0)):.4f}",
            str(int(learned_summary.get("owner_union_split_gt_count", 0))),
            str(int(learned_summary.get("owner_union_merge_pred_count", 0))),
        ],
        [
            "oracle_fragments_no_merge",
            f"{float(dict(oracle_fragments.get('metrics', {})).get('segm/AP', 0.0)):.4f}",
            f"{float(dict(oracle_fragments.get('metrics', {})).get('boundary/IoU', 0.0)):.4f}",
            str(int(oracle_fragments.get("split_gt_count", 0))),
            str(int(oracle_fragments.get("merge_pred_count", 0))),
        ],
        [
            "oracle_owner_union",
            f"{float(dict(oracle_owner_union.get('metrics', {})).get('segm/AP', 0.0)):.4f}",
            f"{float(dict(oracle_owner_union.get('metrics', {})).get('boundary/IoU', 0.0)):.4f}",
            str(int(oracle_owner_union.get("split_gt_count", 0))),
            str(int(oracle_owner_union.get("merge_pred_count", 0))),
        ],
    ]
    output_table = Path(args.output_table_md).resolve()
    output_table.parent.mkdir(parents=True, exist_ok=True)
    output_table.write_text(
        "\n".join(
            [
                "# RGB Phase 2/3 Instance-Local Stage 2 Comparison",
                "",
                markdown_table(table_headers, table_rows).rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_chart(
        baseline_metrics=baseline_metrics,
        oracle_fragments=oracle_fragments,
        oracle_owner_union=oracle_owner_union,
        learned_summary=learned_summary,
        output_path=Path(args.output_chart).resolve(),
    )

    output_md = Path(args.output_md).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        "\n".join(
            [
                "# 2026-03-31 RGB Phase 2/3 Instance-Local Stage 2 Summary",
                "",
                f"- fragment_gate_passed: `{bool(fragment_gate_passed)}`",
                f"- stage3_reentry_allowed: `{bool(stage3_reentry_allowed)}`",
                f"- learned_owner_union segm/AP: `{float(learned_summary.get('owner_union_segm/AP', 0.0)):.4f}`",
                f"- oracle_owner_union segm/AP: `{float(dict(oracle_owner_union.get('metrics', {})).get('segm/AP', 0.0)):.4f}`",
                "",
                "Stage 3 stays paused unless learned owner-union is both strong enough to beat the frozen backbone and still clearly merge-limited.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
