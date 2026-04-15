#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.rgbd.depth_cache import (  # noqa: E402
    build_depth_feature_pack,
    depth_feature_cache_path,
    resolve_depth_feature_cache_dir,
    save_depth_feature_cache,
    write_depth_feature_cache_manifest,
)
from gisec.datasets.ecc_query_dataset import _LiteCOCO, _load_depth_array  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--feature-mode", required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve_depth_dir(root: Path, split: str) -> Path:
    candidates = [
        root / "depth" / split,
        root / "depth" / "depth_npy" / split,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No depth directory found for split={split} under {root}")


def _build_one(
    *,
    dataset_root: str,
    split: str,
    image_size: int,
    feature_mode: str,
    image_id: int,
    file_name: str,
    force: bool,
) -> str:
    root = Path(dataset_root).resolve()
    cache_dir = resolve_depth_feature_cache_dir(
        str(root),
        split=split,
        image_size=image_size,
        feature_mode=feature_mode,
    )
    cache_path = depth_feature_cache_path(cache_dir=cache_dir, image_id=image_id, file_name=file_name)
    if cache_path.exists() and not force:
        return "skipped"
    depth_dir = _resolve_depth_dir(root, split)
    depth_path = depth_dir / f"{Path(file_name).stem}.npy"
    depth = _load_depth_array(depth_path)
    depth = cv2.resize(depth, (int(image_size), int(image_size)), interpolation=cv2.INTER_NEAREST)
    features = build_depth_feature_pack(depth, feature_mode=feature_mode)
    save_depth_feature_cache(
        cache_dir=cache_dir,
        image_id=int(image_id),
        file_name=str(file_name),
        features=features,
    )
    return "written"


def _build_one_from_job(job: dict[str, object]) -> str:
    return _build_one(**job)


def _process_split(dataset_root: str, *, split: str, image_size: int, feature_mode: str, workers: int, force: bool) -> None:
    root = Path(dataset_root).resolve()
    coco = _LiteCOCO(root / "annotations" / f"instances_{split}.json")
    image_ids = sorted(coco.getImgIds())
    jobs = []
    for image_id in image_ids:
        info = coco.loadImgs([int(image_id)])[0]
        jobs.append(
            {
                "dataset_root": str(root),
                "split": str(split),
                "image_size": int(image_size),
                "feature_mode": str(feature_mode),
                "image_id": int(image_id),
                "file_name": str(info["file_name"]),
                "force": bool(force),
            }
        )

    start = time.perf_counter()
    written = 0
    skipped = 0
    max_workers = max(int(workers), 1)
    if max_workers == 1:
        for job in jobs:
            result = _build_one(**job)
            written += int(result == "written")
            skipped += int(result == "skipped")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(_build_one_from_job, jobs):
                written += int(result == "written")
                skipped += int(result == "skipped")
    elapsed_sec = float(time.perf_counter() - start)
    cache_dir = resolve_depth_feature_cache_dir(
        str(root),
        split=split,
        image_size=image_size,
        feature_mode=feature_mode,
    )
    write_depth_feature_cache_manifest(
        cache_dir=cache_dir,
        dataset_root=str(root),
        split=str(split),
        image_size=int(image_size),
        feature_mode=str(feature_mode),
        num_images=len(image_ids),
        elapsed_sec=elapsed_sec,
        num_written=written,
        num_skipped=skipped,
    )
    print(
        f"[baseline-depth-cache] split={split} feature_mode={feature_mode} "
        f"image_size={image_size} written={written} skipped={skipped} "
        f"elapsed_sec={elapsed_sec:.4f} cache_dir={cache_dir}"
    )


def main() -> None:
    args = parse_args()
    splits = args.split or ["train", "val"]
    for split in splits:
        _process_split(
            str(Path(args.dataset_root).resolve()),
            split=str(split),
            image_size=int(args.image_size),
            feature_mode=str(args.feature_mode),
            workers=int(args.workers),
            force=bool(args.force),
        )


if __name__ == "__main__":
    main()
