#!/usr/bin/env python3
"""
Convert COCO instance segmentation annotations to Ultralytics YOLO segmentation format.

Outputs:
  <output_root>/
    images/{train,val}/  (symlinks by default; fallback to copy)
    labels/{train,val}/*.txt
    dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _default_dataset_root() -> Path:
    # File: <workspace>/magformer/baselines/ultralytics_tools/convert_coco_to_yolo_seg.py
    # Dataset: <workspace>/magformer_datasets/0831_1K
    workspace_root = Path(__file__).resolve().parents[3]
    return workspace_root / "magformer_datasets" / "0831_1K"


def _safe_symlink_or_copy(src: Path, dst: Path, prefer_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    # Always symlink/copy from an absolute source path. If `src` is relative, creating a
    # symlink with a relative target will be resolved relative to `dst`, which can easily
    # produce broken links when the converter is run from different working directories.
    src = src.resolve()
    if prefer_symlink:
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _normalize_poly(poly_xy: List[float], w: int, h: int) -> List[float]:
    out = []
    for i, v in enumerate(poly_xy):
        if i % 2 == 0:
            out.append(max(0.0, min(1.0, float(v) / float(w))))
        else:
            out.append(max(0.0, min(1.0, float(v) / float(h))))
    return out


def _rle_to_polys(segm_obj: Dict, h: int, w: int) -> List[List[float]]:
    """
    Decode COCO RLE to one or more polygons.
    Uses OpenCV contours (external only).
    """
    import numpy as np
    import cv2
    from pycocotools import mask as mask_utils

    rle = segm_obj
    if isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, h, w)
    m = mask_utils.decode(rle)  # (h,w) uint8 or (h,w,n)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = (m.astype(np.uint8) * 255).astype(np.uint8)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[float]] = []
    for c in contours:
        if c.shape[0] < 3:
            continue
        pts = c.reshape(-1, 2).astype(float)
        poly = []
        for x, y in pts:
            poly.extend([float(x), float(y)])
        if len(poly) >= 6:
            polys.append(poly)
    return polys


def convert_split(
    *,
    dataset_root: Path,
    split: str,
    output_root: Path,
    cat_id_to_yolo: Dict[int, int],
    class_names: Dict[int, str],
    prefer_symlink: bool,
) -> Tuple[int, int]:
    ann_path = dataset_root / "annotations" / f"instances_{split}.json"
    img_dir = dataset_root / "images" / split
    out_img_dir = output_root / "images" / split
    out_lbl_dir = output_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    img_by_id = {int(im["id"]): im for im in images}
    anns_by_img: Dict[int, List[dict]] = defaultdict(list)
    for a in anns:
        anns_by_img[int(a["image_id"])].append(a)

    num_images = 0
    num_instances = 0
    for img_id, img in img_by_id.items():
        num_images += 1
        file_name = img["file_name"]
        w = int(img["width"])
        h = int(img["height"])

        src_img = img_dir / file_name
        dst_img = out_img_dir / file_name
        _safe_symlink_or_copy(src_img, dst_img, prefer_symlink=prefer_symlink)

        label_path = out_lbl_dir / f"{Path(file_name).stem}.txt"
        lines: List[str] = []
        for a in anns_by_img.get(img_id, []):
            if int(a.get("iscrowd", 0)) == 1:
                # YOLO-seg expects polygons; try best-effort conversion from RLE if present.
                segm = a.get("segmentation", None)
                if isinstance(segm, dict):
                    polys = _rle_to_polys(segm, h=h, w=w)
                else:
                    polys = []
            else:
                segm = a.get("segmentation", None)
                polys = segm if isinstance(segm, list) else []

            if not polys:
                continue

            cat_id = int(a["category_id"])
            cls = int(cat_id_to_yolo.get(cat_id, 0))

            for poly in polys:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                poly_n = _normalize_poly(poly, w=w, h=h)
                # Ultralytics expects: cls x1 y1 x2 y2 ...
                lines.append(" ".join([str(cls)] + [f"{v:.6f}" for v in poly_n]))
                num_instances += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return num_images, num_instances


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/0831_1K (default: workspace-relative).",
    )
    ap.add_argument(
        "--output-root",
        type=str,
        default="output/experiments/_shared/yolo_dataset",
        help="Output root directory under magformer repo.",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated splits to convert (default: train,val).",
    )
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    ap.add_argument("--verify", action="store_true", help="Randomly sample 5 images and check label files.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for --verify sampling.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else _default_dataset_root()
    output_root = Path(args.output_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    prefer_symlink = not args.copy

    # Load categories once from train json (assume consistent across splits)
    train_ann = dataset_root / "annotations" / "instances_train.json"
    coco_train = json.loads(train_ann.read_text(encoding="utf-8"))
    cats = coco_train.get("categories", [])
    cats_sorted = sorted(cats, key=lambda c: int(c["id"]))
    cat_id_to_yolo = {int(c["id"]): i for i, c in enumerate(cats_sorted)}
    class_names = {i: str(c.get("name", i)) for i, c in enumerate(cats_sorted)}

    output_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_instances = 0
    for split in splits:
        n_img, n_inst = convert_split(
            dataset_root=dataset_root,
            split=split,
            output_root=output_root,
            cat_id_to_yolo=cat_id_to_yolo,
            class_names=class_names,
            prefer_symlink=prefer_symlink,
        )
        print(f"[convert] split={split} images={n_img} instances(lines)={n_inst}")
        total_images += n_img
        total_instances += n_inst

    # Write Ultralytics dataset spec
    names_lines = "\n".join([f"  {i}: {name}" for i, name in sorted(class_names.items())])
    dataset_yaml = (
        f"path: {output_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_lines}\n"
    )
    (output_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    print(f"[convert] wrote: {(output_root / 'dataset.yaml')}")
    print(f"[convert] total images={total_images} total label lines={total_instances}")

    if args.verify:
        random.seed(args.seed)
        img_dir = output_root / "images" / "train"
        lbl_dir = output_root / "labels" / "train"
        imgs = sorted([p for p in img_dir.iterdir() if p.is_file()])
        sample = random.sample(imgs, k=min(5, len(imgs)))
        for p in sample:
            lbl = lbl_dir / f"{p.stem}.txt"
            txt = lbl.read_text(encoding="utf-8").strip().splitlines() if lbl.exists() else []
            ok = lbl.exists() and all(len(line.split()) >= 7 for line in txt if line.strip())
            print(f"[verify] {p.name} -> {lbl.name} lines={len(txt)} ok={ok}")


if __name__ == "__main__":
    main()
