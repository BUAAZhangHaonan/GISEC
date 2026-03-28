#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis._suite_utils import markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", action="append", default=[], required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-ap-chart", required=True)
    parser.add_argument("--output-failure-chart", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path_str in paths:
        payload = _read_json(Path(path_str).resolve())
        metrics = dict(payload.get("metrics", {}))
        speed = dict(payload.get("inference_speed", {}))
        rows.append(
            {
                "variant": str(payload.get("variant", Path(path_str).parent.name)),
                "segm/AP": float(metrics.get("segm/AP", 0.0)),
                "bbox/AP": float(metrics.get("bbox/AP", 0.0)),
                "boundary/IoU": float(metrics.get("boundary/IoU", 0.0)),
                "split_gt_count": int(metrics.get("split_gt_count", 0)),
                "merge_pred_count": int(metrics.get("merge_pred_count", 0)),
                "refinement_invocation_rate": float(metrics.get("refinement_invocation_rate", 0.0)),
                "local_graph_invocation_rate": float(metrics.get("local_graph_invocation_rate", 0.0)),
                "throughput_fps": speed.get("throughput_fps"),
                "path": str(Path(path_str).resolve()),
            }
        )
    return rows


def _write_ap_chart(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variants = [row["variant"] for row in rows]
    segm_ap = [row["segm/AP"] for row in rows]
    boundary = [row["boundary/IoU"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(rows))
    ax.bar([value - 0.18 for value in x], segm_ap, width=0.36, label="segm/AP")
    ax.bar([value + 0.18 for value in x], boundary, width=0.36, label="boundary/IoU")
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Active Pilot Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_failure_chart(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variants = [row["variant"] for row in rows]
    split_counts = [row["split_gt_count"] for row in rows]
    merge_counts = [row["merge_pred_count"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(rows))
    ax.bar([value - 0.18 for value in x], split_counts, width=0.36, label="split_gt_count")
    ax.bar([value + 0.18 for value in x], merge_counts, width=0.36, label="merge_pred_count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Active Pilot Failure Counts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = _load_rows(list(args.run_summary))
    rows.sort(key=lambda item: item["variant"])
    best = max(rows, key=lambda item: item["segm/AP"])
    payload = {
        "num_runs": len(rows),
        "best_variant": best["variant"],
        "best_segm_ap": best["segm/AP"],
        "rows": rows,
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    headers = [
        "variant",
        "segm/AP",
        "bbox/AP",
        "boundary/IoU",
        "split_gt_count",
        "merge_pred_count",
        "refine_rate",
        "graph_rate",
        "fps",
    ]
    table_rows = [
        [
            row["variant"],
            f"{row['segm/AP']:.4f}",
            f"{row['bbox/AP']:.4f}",
            f"{row['boundary/IoU']:.4f}",
            str(row["split_gt_count"]),
            str(row["merge_pred_count"]),
            f"{row['refinement_invocation_rate']:.4f}",
            f"{row['local_graph_invocation_rate']:.4f}",
            "n/a" if row["throughput_fps"] is None else f"{float(row['throughput_fps']):.2f}",
        ]
        for row in rows
    ]
    output_md = Path(args.output_md).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        "\n".join(
            [
                "# GISEC Active Pilot Summary",
                "",
                f"- runs: `{len(rows)}`",
                f"- best_variant: `{best['variant']}`",
                f"- best segm/AP: `{best['segm/AP']:.4f}`",
                "",
                markdown_table(headers, table_rows),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_ap_chart(rows, Path(args.output_ap_chart).resolve())
    _write_failure_chart(rows, Path(args.output_failure_chart).resolve())


if __name__ == "__main__":
    main()
