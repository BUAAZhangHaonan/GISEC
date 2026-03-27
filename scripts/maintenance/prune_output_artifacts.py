#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output"
EXPERIMENTS_ROOT = OUTPUT_ROOT / "experiments"
BASELINES_ROOT = EXPERIMENTS_ROOT / "baselines"

KEEP_EXPERIMENT_DIRS = {
    "reference_graph_merge_pilot_20260326",
    "reference_graph_merge_edgetype_pilot_20260326",
    "reference_graph_geom_pilot_20260327",
}

KEEP_BASELINE_DIRS = {
    "rgb_standalone_phase1_20260324",
    "rgb_standalone_phase1_cache_20260325",
    "rgbd_standalone_phase1_20260325",
    "splitfirst_phase1_20260326",
    "splitfirst_phase1_calibrated_20260326",
}

ANALYSIS_TMP_PATTERNS = (
    "eval_profile_overlays_tmp*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune obsolete output artifacts while keeping current reference experiments.")
    parser.add_argument("--execute", action="store_true", help="Actually delete files instead of printing the plan.")
    return parser.parse_args()


def _collect_removals() -> list[Path]:
    removals: list[Path] = []
    analysis_root = OUTPUT_ROOT / "analysis"
    for pattern in ANALYSIS_TMP_PATTERNS:
        removals.extend(sorted(path.resolve() for path in analysis_root.glob(pattern) if path.is_dir()))

    if EXPERIMENTS_ROOT.exists():
        for path in sorted(EXPERIMENTS_ROOT.iterdir()):
            if not path.is_dir():
                continue
            if path.name == "baselines":
                continue
            if path.name not in KEEP_EXPERIMENT_DIRS:
                removals.append(path.resolve())

    if BASELINES_ROOT.exists():
        for path in sorted(BASELINES_ROOT.iterdir()):
            if not path.is_dir():
                continue
            if path.name not in KEEP_BASELINE_DIRS:
                removals.append(path.resolve())

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in removals:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def main() -> None:
    args = parse_args()
    removals = _collect_removals()
    if not removals:
        print("No obsolete output artifacts found.")
        return
    print(f"Found {len(removals)} paths to remove:")
    for path in removals:
        print(path)
    if not args.execute:
        print("\nDry-run only. Re-run with --execute to delete these paths.")
        return
    for path in removals:
        shutil.rmtree(path)
    print(f"\nRemoved {len(removals)} paths.")


if __name__ == "__main__":
    main()
