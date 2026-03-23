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


ORDER = {"v1.5 legacy": 0, "UQ-s": 1, "UQ-m": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _row_from_run(run_summary_path: Path) -> dict[str, Any]:
    payload = _read_json(run_summary_path)
    run_dir = run_summary_path.parent
    variant = str(payload.get("variant", payload.get("model_id", run_dir.name)))
    if variant != "v1.5 legacy":
        if "use_reference" not in payload or "use_graph_rescue" not in payload:
            raise ValueError(f"Alpha ladder requires explicit module flags for {variant}: {run_dir}")
        if bool(payload.get("use_reference")) or bool(payload.get("use_graph_rescue")):
            raise ValueError(f"Alpha ladder received contaminated run: {run_dir}")
    metrics = dict(payload.get("metrics", {}))
    match = _read_optional(run_dir / "match_diagnostics_summary.json")
    pathology = _read_optional(run_dir / "object_pathology_summary.json")
    failures = _read_optional(run_dir / "failure_summary.json")
    return {
        "variant": variant,
        "segm/AP": float(metrics.get("segm/AP", 0.0)),
        "bbox/AP": float(metrics.get("bbox/AP", 0.0)),
        "pred_count_mean": float(match.get("pred_count_mean", 0.0)),
        "gt_count_mean": float(match.get("gt_count_mean", 0.0)),
        "best_mask_iou_mean": float(match.get("best_mask_iou_mean", 0.0)),
        "best_bbox_iou_mean": float(match.get("best_bbox_iou_mean", 0.0)),
        "object_count_mean": float(pathology.get("object_count_mean", 0.0)),
        "split_count_mean": float(pathology.get("split_count_mean", 0.0)),
        "failure_counts": dict(failures.get("counts", {})),
        "params_trainable": payload.get("params_trainable"),
        "throughput_fps": payload.get("inference_speed", {}).get("throughput_fps"),
        "path": str(run_dir),
    }


def _compute_gates(rows: list[dict[str, Any]]) -> dict[str, bool]:
    lookup = {row["variant"]: row for row in rows}
    legacy = lookup.get("v1.5 legacy")
    uq_s = lookup.get("UQ-s")
    uq_m = lookup.get("UQ-m")
    gate_a = bool(
        legacy
        and uq_s
        and uq_s["segm/AP"] >= legacy["segm/AP"]
        and uq_s["best_mask_iou_mean"] >= legacy["best_mask_iou_mean"]
    )
    gate_b = bool(
        uq_s
        and uq_m
        and uq_m["segm/AP"] >= uq_s["segm/AP"]
        and uq_m["best_mask_iou_mean"] >= uq_s["best_mask_iou_mean"]
    )
    return {
        "gate_a_pass": gate_a,
        "gate_b_pass": gate_b,
    }


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(suite_root.rglob("run_summary.json")):
        row = _row_from_run(path)
        if row["variant"] in ORDER:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No run_summary.json found under {suite_root}")
    seen: set[str] = set()
    for row in rows:
        variant = str(row["variant"])
        if variant in seen:
            raise ValueError(f"Duplicate alpha summary row for variant: {variant}")
        seen.add(variant)
    rows.sort(key=lambda row: (ORDER.get(row["variant"], 999), row["variant"]))
    gates = _compute_gates(rows)
    payload = {
        "suite_root": str(suite_root),
        "rows": rows,
        "gates": gates,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table_rows = [
        [
            row["variant"],
            f"{row['segm/AP']:.4f}",
            f"{row['bbox/AP']:.4f}",
            f"{row['pred_count_mean']:.4f}",
            f"{row['gt_count_mean']:.4f}",
            f"{row['best_mask_iou_mean']:.4f}",
            f"{row['best_bbox_iou_mean']:.4f}",
            f"{row['object_count_mean']:.4f}",
            f"{row['split_count_mean']:.4f}",
        ]
        for row in rows
    ]
    markdown = "\n".join(
        [
            "# GISEC Query Alpha Ladder Summary",
            "",
            f"- suite_root: `{suite_root}`",
            f"- gate_a_pass: `{gates['gate_a_pass']}`",
            f"- gate_b_pass: `{gates['gate_b_pass']}`",
            "",
            markdown_table(
                [
                    "Variant",
                    "segm/AP",
                    "bbox/AP",
                    "pred_count_mean",
                    "gt_count_mean",
                    "best_mask_iou_mean",
                    "best_bbox_iou_mean",
                    "object_count_mean",
                    "split_count_mean",
                ],
                table_rows,
            ).rstrip(),
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
