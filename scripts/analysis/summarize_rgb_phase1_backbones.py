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
    parser.add_argument("--short-run-summary", action="append", default=[], required=True)
    parser.add_argument("--full-run-summary", action="append", default=[], required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-short-chart", required=True)
    parser.add_argument("--output-full-chart", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_label(payload: dict[str, Any]) -> str:
    benchmark = dict(payload.get("benchmark", {}))
    model_family = str(benchmark.get("model_family") or payload.get("model") or "unknown")
    if model_family == "mask2former":
        return "Mask2Former"
    if model_family == "mask_rcnn":
        return "Mask R-CNN"
    return model_family


def _resolution(payload: dict[str, Any]) -> int:
    benchmark = dict(payload.get("benchmark", {}))
    resolution = benchmark.get("resolution")
    if resolution is not None:
        return int(resolution)
    variant = str(payload.get("variant", ""))
    for token in variant.split("_"):
        if token.isdigit():
            return int(token)
    return 0


def _input_mode(payload: dict[str, Any]) -> str:
    benchmark = dict(payload.get("benchmark", {}))
    mode = str(benchmark.get("input_mode") or payload.get("modality") or "unknown")
    return mode


def _friendly_name(payload: dict[str, Any]) -> str:
    return f"{_model_label(payload)} {_input_mode(payload)} {_resolution(payload)}"


def _load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_str in paths:
        payload = _read_json(Path(path_str).resolve())
        metrics = dict(payload.get("metrics", {}))
        speed = dict(payload.get("inference_speed", {}))
        rows.append(
            {
                "model_label": _model_label(payload),
                "friendly_name": _friendly_name(payload),
                "variant": str(payload.get("variant", Path(path_str).parent.name)),
                "input_mode": _input_mode(payload),
                "resolution": _resolution(payload),
                "segm/AP": float(metrics.get("segm/AP", 0.0)),
                "bbox/AP": float(metrics.get("bbox/AP", 0.0)),
                "boundary/IoU": float(metrics.get("boundary/IoU", 0.0)),
                "fps": None if speed.get("throughput_fps") is None else float(speed.get("throughput_fps")),
                "path": str(Path(path_str).resolve()),
            }
        )
    return rows


def _write_short_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolutions = sorted({int(row["resolution"]) for row in rows})
    family_order = ["Mask R-CNN", "Mask2Former"]
    offsets = {
        "Mask R-CNN": -0.18,
        "Mask2Former": 0.18,
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(resolutions)))
    for family in family_order:
        values = []
        for resolution in resolutions:
            match = next(
                (row for row in rows if row["model_label"] == family and int(row["resolution"]) == int(resolution)),
                None,
            )
            values.append(0.0 if match is None else float(match["segm/AP"]))
        ax.bar([value + offsets[family] for value in x], values, width=0.36, label=family)
    ax.set_xticks(x)
    ax.set_xticklabels([str(resolution) for resolution in resolutions])
    ax.set_xlabel("Resolution")
    ax.set_ylabel("segm/AP")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Phase 1 RGB Short Matrix")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_full_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["model_label"] for row in rows]
    segm_ap = [float(row["segm/AP"]) for row in rows]
    bbox_ap = [float(row["bbox/AP"]) for row in rows]
    boundary = [float(row["boundary/IoU"]) for row in rows]
    fps = [0.0 if row["fps"] is None else float(row["fps"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    left = axes[0]
    x = list(range(len(rows)))
    left.bar([value - 0.24 for value in x], segm_ap, width=0.24, label="segm/AP")
    left.bar(x, bbox_ap, width=0.24, label="bbox/AP")
    left.bar([value + 0.24 for value in x], boundary, width=0.24, label="boundary/IoU")
    left.set_xticks(x)
    left.set_xticklabels(labels, rotation=15, ha="right")
    left.set_ylim(bottom=0.0)
    left.set_title("Phase 1 RGB Full Accuracy")
    left.legend()

    right = axes[1]
    right.bar(labels, fps, width=0.5)
    right.set_ylim(bottom=0.0)
    right.set_title("Phase 1 RGB Full Throughput")
    right.set_ylabel("FPS")
    right.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    short_rows = _load_rows(list(args.short_run_summary))
    full_rows = _load_rows(list(args.full_run_summary))
    short_rows.sort(key=lambda row: (int(row["resolution"]), row["model_label"]))
    full_rows.sort(key=lambda row: row["model_label"])
    best_full = max(full_rows, key=lambda row: row["segm/AP"])

    payload = {
        "num_short_runs": len(short_rows),
        "num_full_runs": len(full_rows),
        "phase1_winner": best_full["model_label"],
        "phase1_winner_segm_ap": best_full["segm/AP"],
        "short_rows": short_rows,
        "full_rows": full_rows,
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    short_table = [
        [
            row["model_label"],
            str(row["resolution"]),
            row["input_mode"],
            f"{row['segm/AP']:.4f}",
            f"{row['bbox/AP']:.4f}",
            f"{row['boundary/IoU']:.4f}",
            "n/a" if row["fps"] is None else f"{row['fps']:.2f}",
        ]
        for row in short_rows
    ]
    full_table = [
        [
            row["model_label"],
            str(row["resolution"]),
            row["input_mode"],
            f"{row['segm/AP']:.4f}",
            f"{row['bbox/AP']:.4f}",
            f"{row['boundary/IoU']:.4f}",
            "n/a" if row["fps"] is None else f"{row['fps']:.2f}",
        ]
        for row in full_rows
    ]
    output_md = Path(args.output_md).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        "\n".join(
            [
                "# GISEC RGB Phase 1 Backbone Summary",
                "",
                f"- phase1_winner: `{best_full['model_label']}`",
                f"- phase1_winner_segm_ap: `{best_full['segm/AP']:.4f}`",
                "",
                "## Short Matrix",
                "",
                markdown_table(
                    ["Model", "Resolution", "Input", "segm/AP", "bbox/AP", "boundary/IoU", "FPS"],
                    short_table,
                ).rstrip(),
                "",
                "## Full Runs",
                "",
                markdown_table(
                    ["Model", "Resolution", "Input", "segm/AP", "bbox/AP", "boundary/IoU", "FPS"],
                    full_table,
                ).rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_short_chart(short_rows, Path(args.output_short_chart).resolve())
    _write_full_chart(full_rows, Path(args.output_full_chart).resolve())


if __name__ == "__main__":
    main()
