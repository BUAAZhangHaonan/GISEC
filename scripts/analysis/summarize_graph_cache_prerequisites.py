#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis._suite_utils import markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", action="append", default=[], required=True)
    parser.add_argument("--label", action="append", default=[], required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-chart", required=True)
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    return 0.0 if float(den) <= 0.0 else float(num) / float(den)


def _compute_cache_metrics(cache_dir: Path) -> dict[str, Any]:
    sample_paths = sorted(path for path in cache_dir.glob("*.pt") if path.is_file())
    total_gt = 0
    total_fragments = 0
    total_covered_gt = 0
    total_split_gt = 0
    total_singleton_gt = 0
    total_fragment_assignments = 0
    total_impure_fragments = 0
    total_purity = 0.0
    total_same_pairs = 0
    total_same_pairs_covered = 0
    total_positive_edges = 0
    total_valid_edges = 0

    for sample_path in sample_paths:
        payload = torch.load(sample_path, map_location="cpu")
        summary = dict(payload.get("summary", {}))
        fragment_stats = [dict(row) for row in payload.get("fragment_stats", [])]
        gt_count = int(summary.get("gt_count", 0))
        gt_fragment_counts: dict[int, int] = defaultdict(int)
        for row in fragment_stats:
            gt_instance = int(row.get("gt_instance", 0))
            purity = float(row.get("purity", 0.0))
            total_purity += purity
            total_impure_fragments += int(purity < 0.99)
            if gt_instance > 0:
                gt_fragment_counts[int(gt_instance)] += 1
        covered_gt = len(gt_fragment_counts)
        split_gt = sum(1 for count in gt_fragment_counts.values() if int(count) >= 2)
        singleton_gt = sum(1 for count in gt_fragment_counts.values() if int(count) == 1)

        total_gt += gt_count
        total_fragments += len(fragment_stats)
        total_covered_gt += covered_gt
        total_split_gt += split_gt
        total_singleton_gt += singleton_gt
        total_fragment_assignments += sum(gt_fragment_counts.values())
        total_same_pairs += int(summary.get("same_instance_pairs_total", 0))
        total_same_pairs_covered += int(summary.get("same_instance_pairs_covered", 0))
        total_positive_edges += int(summary.get("positive_edge_count", 0))
        total_valid_edges += int(summary.get("valid_edge_count", 0))

    sample_count = len(sample_paths)
    return {
        "num_samples": int(sample_count),
        "avg_gt_per_image": _safe_div(float(total_gt), float(sample_count)),
        "avg_fragments_per_image": _safe_div(float(total_fragments), float(sample_count)),
        "covered_gt_rate": _safe_div(float(total_covered_gt), float(total_gt)),
        "missing_gt_rate": 1.0 - _safe_div(float(total_covered_gt), float(total_gt)),
        "split_gt_rate": _safe_div(float(total_split_gt), float(total_gt)),
        "singleton_gt_rate": _safe_div(float(total_singleton_gt), float(total_gt)),
        "avg_fragments_per_covered_gt": _safe_div(float(total_fragment_assignments), float(total_covered_gt)),
        "impure_fragment_rate": _safe_div(float(total_impure_fragments), float(total_fragments)),
        "fragment_purity_mean": _safe_div(float(total_purity), float(total_fragments)),
        "same_instance_recall": _safe_div(float(total_same_pairs_covered), float(total_same_pairs)),
        "positive_edge_ratio": _safe_div(float(total_positive_edges), float(total_valid_edges)),
    }


def _write_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in rows]
    x = list(range(len(rows)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar([value - 0.27 for value in x], [float(row["covered_gt_rate"]) for row in rows], width=0.18, label="covered_gt_rate")
    axes[0].bar([value - 0.09 for value in x], [float(row["split_gt_rate"]) for row in rows], width=0.18, label="split_gt_rate")
    axes[0].bar([value + 0.09 for value in x], [float(row["impure_fragment_rate"]) for row in rows], width=0.18, label="impure_fragment_rate")
    axes[0].bar([value + 0.27 for value in x], [float(row["same_instance_recall"]) for row in rows], width=0.18, label="same_instance_recall")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Stage 3 Preconditions Rates")
    axes[0].legend()

    axes[1].bar([value - 0.15 for value in x], [float(row["avg_gt_per_image"]) for row in rows], width=0.3, label="avg_gt_per_image")
    axes[1].bar([value + 0.15 for value in x], [float(row["avg_fragments_per_image"]) for row in rows], width=0.3, label="avg_fragments_per_image")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_title("GT vs Fragment Counts")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cache_dirs = list(args.cache_dir)
    labels = list(args.label)
    if len(cache_dirs) != len(labels):
        raise SystemExit("--cache-dir and --label must have the same length")

    rows: list[dict[str, Any]] = []
    for label, cache_dir in zip(labels, cache_dirs):
        row = _compute_cache_metrics(Path(cache_dir).resolve())
        row["label"] = str(label)
        row["cache_dir"] = str(Path(cache_dir).resolve())
        rows.append(row)

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows}
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table_rows = [
        [
            str(row["label"]),
            f"{float(row['avg_gt_per_image']):.2f}",
            f"{float(row['avg_fragments_per_image']):.2f}",
            f"{float(row['covered_gt_rate']):.4f}",
            f"{float(row['split_gt_rate']):.4f}",
            f"{float(row['singleton_gt_rate']):.4f}",
            f"{float(row['impure_fragment_rate']):.4f}",
            f"{float(row['same_instance_recall']):.4f}",
            f"{float(row['positive_edge_ratio']):.4f}",
        ]
        for row in rows
    ]
    output_md = Path(args.output_md).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        "\n".join(
            [
                "# GISEC Graph Cache Preconditions",
                "",
                markdown_table(
                    [
                        "Label",
                        "avg_gt_per_image",
                        "avg_fragments_per_image",
                        "covered_gt_rate",
                        "split_gt_rate",
                        "singleton_gt_rate",
                        "impure_fragment_rate",
                        "same_instance_recall",
                        "positive_edge_ratio",
                    ],
                    table_rows,
                ).rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_chart(rows, Path(args.output_chart).resolve())


if __name__ == "__main__":
    main()
