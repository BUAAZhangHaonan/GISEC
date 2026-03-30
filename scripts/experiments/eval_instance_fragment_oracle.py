#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.instance_fragment_generator.oracle import evaluate_instance_fragment_oracles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_instance_fragment_oracles(
        cache_root=str(Path(args.cache_root).resolve()),
        dataset_root=str(Path(args.dataset_root).resolve()),
        output_root=str(Path(args.output_root).resolve()),
        split=str(args.split),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
