#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.common.reference_split_cache import build_reference_split_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_reference_split_cache(
        dataset_root=str(Path(args.dataset_root).resolve()),
        reference_root=str(Path(args.reference_root).resolve()),
        split=str(args.split),
        image_size=int(args.image_size),
        output_root=str(Path(args.output_root).resolve()),
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
