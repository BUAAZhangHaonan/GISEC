#!/usr/bin/env python3
"""
Run UOAIS (vendored baseline) on ECC datasets (0831 / 0909) with COCOeval (segm/bbox AP).

This wrapper:
- registers the ECC RGBD COCO dataset in-process (adds `depth_file_name`)
- executes the vendored UOAIS `train_net.py` with passthrough args
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
from pathlib import Path
from typing import List

# Ensure sibling baseline utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from ecc_datasets import register_ecc_coco_rgbd


def _patch_short_run_common_metric_printer() -> None:
    from detectron2.utils.events import CommonMetricPrinter

    if getattr(CommonMetricPrinter._get_eta, "__name__", "") == "_magformer_safe_get_eta":
        return

    original_get_eta = CommonMetricPrinter._get_eta

    def _magformer_safe_get_eta(self, storage):
        try:
            return original_get_eta(self, storage)
        except ZeroDivisionError:
            self._last_write = (storage.iter, time.perf_counter())
            return None

    CommonMetricPrinter._get_eta = _magformer_safe_get_eta


def _split_args(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def _rewrite_passthrough_paths(argv: List[str], repo_root: Path) -> List[str]:
    rewritten = list(argv)
    for idx, token in enumerate(rewritten):
        if token == "--config-file" and idx + 1 < len(rewritten):
            candidate = Path(rewritten[idx + 1])
            if not candidate.is_absolute():
                repo_candidate = (repo_root / candidate).resolve()
                if repo_candidate.exists():
                    rewritten[idx + 1] = str(repo_candidate)
        elif token.startswith("--config-file="):
            _, raw_path = token.split("=", 1)
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                repo_candidate = (repo_root / candidate).resolve()
                if repo_candidate.exists():
                    rewritten[idx] = f"--config-file={repo_candidate}"
    return rewritten


def _ensure_dataset_name_overrides(argv: List[str], train_name: str, val_name: str) -> List[str]:
    rewritten = list(argv)
    if "DATASETS.TRAIN" not in rewritten:
        rewritten.extend(["DATASETS.TRAIN", f"('{train_name}',)"])
    if "DATASETS.TEST" not in rewritten:
        rewritten.extend(["DATASETS.TEST", f"('{val_name}',)"])
    return rewritten


def _ensure_uoais_gpu_argument(argv: List[str]) -> List[str]:
    rewritten = list(argv)
    if "--gpu" in rewritten or any(token.startswith("--gpu=") for token in rewritten):
        return rewritten
    if "--num-gpus" in rewritten:
        idx = rewritten.index("--num-gpus")
        if idx + 1 < len(rewritten):
            try:
                num_gpus = int(rewritten[idx + 1])
            except ValueError:
                return rewritten
            if num_gpus > 1:
                insertion = idx + 2
                rewritten[insertion:insertion] = ["--gpu", ",".join(str(i) for i in range(num_gpus))]
    return rewritten


def main() -> None:
    wrapper_argv, passthrough = _split_args(sys.argv[1:])
    workspace_root = BASELINES_DIR.parent

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--register",
        type=str,
        default="0831",
        help="ECC dataset id: 0831 | 0909",
    )
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/<dataset> root (default: env + workspace-relative).",
    )
    ap.add_argument(
        "--uoais-root",
        type=str,
        default="baselines/uoais",
        help="Path to vendored UOAIS repo checkout.",
    )
    args = ap.parse_args(wrapper_argv)

    train_name, val_name = register_ecc_coco_rgbd(args.register, args.dataset_root)

    uoais_root = Path(args.uoais_root).resolve()
    train_py = uoais_root / "train_net.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train_net.py not found: {train_py}")

    os.chdir(uoais_root)
    sys.path.insert(0, str(uoais_root))
    _patch_short_run_common_metric_printer()

    passthrough = _rewrite_passthrough_paths(passthrough, workspace_root)
    passthrough = _ensure_dataset_name_overrides(passthrough, train_name=train_name, val_name=val_name)
    passthrough = _ensure_uoais_gpu_argument(passthrough)
    sys.argv = [str(train_py)] + passthrough
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
