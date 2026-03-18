#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis._suite_utils import load_suite_rows, markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    output = Path(args.output).resolve()
    rows = load_suite_rows(suite_root)
    table_rows: list[list[str]] = []
    for row in sorted(rows, key=lambda item: item.variant):
        table_rows.append(
            [
                row.variant,
                f"{float(row.metrics.get('segm/AP', 0.0)):.4f}",
                f"{float(row.metrics.get('segm/AP50', 0.0)):.4f}",
                f"{float(row.metrics.get('segm/AP75', 0.0)):.4f}",
                f"{float(row.metrics.get('segm/APs', 0.0)):.4f}",
                f"{float(row.metrics.get('segm/APm', 0.0)):.4f}",
                f"{float(row.metrics.get('segm/APl', 0.0)):.4f}",
                str(row.params_trainable if row.params_trainable is not None else ""),
                str(row.wall_time_sec if row.wall_time_sec is not None else ""),
                f"{float(row.inference_speed.get('throughput_fps', 0.0)):.4f}",
                f"{float(row.inference_speed.get('inference_peak_memory_mb', 0.0)):.4f}",
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        markdown_table(
            [
                "Model",
                "segm/AP",
                "segm/AP50",
                "segm/AP75",
                "segm/APs",
                "segm/APm",
                "segm/APl",
                "params_trainable",
                "wall_time_sec",
                "throughput_fps",
                "peak_memory_mb",
            ],
            table_rows,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
