#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gisec.utils.visualization import render_fragment_merge_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--fragments", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    fragments_path = Path(args.fragments)
    merged_path = Path(args.merged)
    output_path = Path(args.output)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fragments = np.load(fragments_path)
    merged = np.load(merged_path)
    render_fragment_merge_preview(image=image, fragments=fragments, merged=merged, output_path=output_path)


if __name__ == "__main__":
    main()
