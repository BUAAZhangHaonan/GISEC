#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis._suite_utils import load_suite_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    rows = load_suite_rows(suite_root)
    if not rows:
        raise FileNotFoundError(f"No run_summary.json files found under {suite_root}")

    best = max(rows, key=lambda row: row.segm_ap)
    summary = {
        "suite_root": str(suite_root),
        "num_runs": len(rows),
        "best_variant": best.variant,
        "best segm/AP": best.segm_ap,
        "variants": [
            {
                "variant": row.variant,
                "segm/AP": row.segm_ap,
                "throughput_fps": row.inference_speed.get("throughput_fps"),
                "peak_memory_mb": row.inference_speed.get("inference_peak_memory_mb"),
                "path": str(row.path),
            }
            for row in sorted(rows, key=lambda item: item.variant)
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        "\n".join(
            [
                "# GISEC Suite Summary",
                "",
                f"- suite_root: `{suite_root}`",
                f"- num_runs: `{len(rows)}`",
                f"- best_variant: `{best.variant}`",
                f"- best segm/AP: `{best.segm_ap:.4f}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
