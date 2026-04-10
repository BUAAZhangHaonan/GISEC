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


def _collect_removals(
    *,
    output_root: Path | None = None,
    experiments_root: Path | None = None,
    baselines_root: Path | None = None,
    analysis_tmp_patterns: tuple[str, ...] | None = None,
    keep_experiment_dirs: set[str] | None = None,
    keep_baseline_dirs: set[str] | None = None,
) -> list[Path]:
    output_root = OUTPUT_ROOT if output_root is None else output_root
    experiments_root = EXPERIMENTS_ROOT if experiments_root is None else experiments_root
    baselines_root = BASELINES_ROOT if baselines_root is None else baselines_root
    analysis_tmp_patterns = ANALYSIS_TMP_PATTERNS if analysis_tmp_patterns is None else analysis_tmp_patterns
    keep_experiment_dirs = KEEP_EXPERIMENT_DIRS if keep_experiment_dirs is None else keep_experiment_dirs
    keep_baseline_dirs = KEEP_BASELINE_DIRS if keep_baseline_dirs is None else keep_baseline_dirs
    removals: list[Path] = []
    analysis_root = output_root / "analysis"
    for pattern in analysis_tmp_patterns:
        removals.extend(sorted(path for path in analysis_root.glob(pattern) if path.is_dir()))

    if experiments_root.exists():
        for path in sorted(experiments_root.iterdir()):
            if not path.is_dir():
                continue
            if path.name == "baselines":
                continue
            if path.name not in keep_experiment_dirs:
                removals.append(path)

    if baselines_root.exists():
        for path in sorted(baselines_root.iterdir()):
            if not path.is_dir():
                continue
            if path.name not in keep_baseline_dirs:
                removals.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in removals:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _approve_removal_path(path: Path, *, output_root: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"Refusing to delete symlinked path: {path}")
    resolved_root = output_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete path outside output root: {path}") from exc
    return path


def _approve_removals(paths: list[Path], *, output_root: Path | None = None) -> list[Path]:
    output_root = OUTPUT_ROOT if output_root is None else output_root
    approved: list[Path] = []
    for path in paths:
        approved.append(_approve_removal_path(path, output_root=output_root))
    return approved


def _remove_paths(paths: list[Path]) -> None:
    for path in paths:
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    removals = _approve_removals(_collect_removals(), output_root=OUTPUT_ROOT)
    if not removals:
        print("No obsolete output artifacts found.")
        return
    print(f"Found {len(removals)} paths to remove:")
    for path in removals:
        print(path)
    if not args.execute:
        print("\nDry-run only. Re-run with --execute to delete these paths.")
        return
    _remove_paths(removals)
    print(f"\nRemoved {len(removals)} paths.")


if __name__ == "__main__":
    main()
