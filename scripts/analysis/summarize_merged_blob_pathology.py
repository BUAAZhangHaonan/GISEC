#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.common.pathology import build_prediction_pathology_rows, summarize_prediction_pathology  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_prediction_pathology(
        dataset_root=str(Path(args.dataset_root).resolve()),
        results_json=str(Path(args.results_json).resolve()),
        split=str(args.split),
    )
    rows = build_prediction_pathology_rows(
        dataset_root=str(Path(args.dataset_root).resolve()),
        results_json=str(Path(args.results_json).resolve()),
        split=str(args.split),
    )
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl).resolve()
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
