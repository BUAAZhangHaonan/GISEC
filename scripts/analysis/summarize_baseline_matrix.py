#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis._suite_utils import markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_scalar(path: Path) -> str:
    if not path.exists():
        return "n/a"
    return path.read_text(encoding="utf-8").strip() or "n/a"


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
        rows.append(
            {
                "model": str(payload.get("model", run_dir.name)),
                "variant": str(payload.get("variant", run_dir.name)),
                "modality": str(payload.get("modality", "unknown")),
                "segm_ap": float(metrics.get("segm/AP", 0.0)),
                "segm_ap50": float(metrics.get("segm/AP50", 0.0)),
                "fps": speed.get("throughput_fps"),
                "peak_memory_mb": speed.get("inference_peak_memory_mb"),
                "params_trainable": _read_scalar(run_dir / "params_trainable.txt"),
                "path": str(run_dir),
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
            f"{row['segm_ap']:.4f}",
            f"{row['segm_ap50']:.4f}",
            "n/a" if row["fps"] is None else f"{float(row['fps']):.4f}",
            "n/a" if row["peak_memory_mb"] is None else f"{float(row['peak_memory_mb']):.4f}",
            str(row["params_trainable"]),
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
                ["Model", "Variant", "Modality", "segm/AP", "segm/AP50", "FPS", "Peak Memory MB", "Params", "Run Dir"],
                table_rows,
            ).rstrip(),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
