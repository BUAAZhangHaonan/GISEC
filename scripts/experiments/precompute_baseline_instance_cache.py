#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.common.instance_targets import (  # noqa: E402
    build_instance_target_pack,
    instance_target_cache_path,
    resolve_instance_target_cache_dir,
    save_instance_target_cache,
    write_instance_target_cache_manifest,
)
from gisec.datasets.ecc_query_dataset import _LiteCOCO, ann_to_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _build_one(
    *,
    dataset_root: str,
    split: str,
    image_size: int,
    image_id: int,
    file_name: str,
    anns: list[dict],
    height: int,
    width: int,
    force: bool,
) -> str:
    cache_dir = resolve_instance_target_cache_dir(dataset_root, split=split, image_size=image_size)
    cache_path = instance_target_cache_path(cache_dir=cache_dir, image_id=image_id, file_name=file_name)
    if cache_path.exists() and not force:
        return "skipped"
    instance_map = np.zeros((int(image_size), int(image_size)), dtype=np.int64)
    next_id = 0
    for ann in anns:
        mask = ann_to_mask(ann, int(height), int(width))
        mask = cv2.resize(mask, (int(image_size), int(image_size)), interpolation=cv2.INTER_NEAREST)
        if int(mask.max()) <= 0:
            continue
        next_id += 1
        instance_map[mask > 0] = next_id
    targets = build_instance_target_pack(instance_map)
    save_instance_target_cache(
        cache_dir=cache_dir,
        image_id=int(image_id),
        file_name=str(file_name),
        instance_map=instance_map,
        targets=targets,
    )
    return "written"


def _process_split(dataset_root: str, *, split: str, image_size: int, workers: int, force: bool) -> None:
    root = Path(dataset_root).resolve()
    coco = _LiteCOCO(root / "annotations" / f"instances_{split}.json")
    image_ids = sorted(coco.getImgIds())
    cache_dir = resolve_instance_target_cache_dir(str(root), split=split, image_size=image_size)
    jobs: list[dict[str, object]] = []
    for image_id in image_ids:
        info = coco.loadImgs([int(image_id)])[0]
        ann_ids = coco.getAnnIds(imgIds=[int(image_id)], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        jobs.append(
            {
                "dataset_root": str(root),
                "split": str(split),
                "image_size": int(image_size),
                "image_id": int(image_id),
                "file_name": str(info["file_name"]),
                "anns": anns,
                "height": int(info["height"]),
                "width": int(info["width"]),
                "force": bool(force),
            }
        )

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

    write_instance_target_cache_manifest(
        cache_dir=cache_dir,
        dataset_root=str(root),
        split=str(split),
        image_size=int(image_size),
        num_images=len(image_ids),
    )
    print(f"[baseline-cache] split={split} image_size={image_size} written={written} skipped={skipped} cache_dir={cache_dir}")


def _build_one_from_job(job: dict[str, object]) -> str:
    return _build_one(**job)


def main() -> None:
    args = parse_args()
    splits = args.split or ["train", "val"]
    for split in splits:
        _process_split(
            str(Path(args.dataset_root).resolve()),
            split=str(split),
            image_size=int(args.image_size),
            workers=int(args.workers),
            force=bool(args.force),
        )


if __name__ == "__main__":
    main()
