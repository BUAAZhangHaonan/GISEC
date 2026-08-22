"""E2: scoring simulation and fragmentation cost on the 1566 val split.

Zero-training, offline. Two questions:

(a) With GT masks as a perfect segmenter, which cheap scoring schemes keep
    the COCO AP ranking close to the oracle (score = 1.0)?
(b) How much AP dies when GT masks are shredded by connected components
    (the March U-Net pipeline failure mode), and how much does an oracle
    merge recover? Also: grouping fragments by an unsupervised depth
    ordering instead of GT identity, to price wrong fragment identity.

Usage: python run_scoring_sim.py [--split val]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

from gisec.eval.coco_eval import evaluate_json
from gisec.eval.coco_export import masks_to_coco_results

SCORE_MIN = 0.1
SCORE_MAX = 0.9
SCORE_FLOOR = 0.05  # pipeline score threshold, matches the 0.05 protocol
CATEGORY_ID = 1
RNG = np.random.default_rng(20260817)


def load_image(coco: COCO, img_id: int, images_dir: Path) -> np.ndarray:
    info = coco.loadImgs(img_id)[0]
    path = images_dir / info["file_name"]
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def load_depth(coco: COCO, img_id: int, depth_dir: Path) -> np.ndarray:
    info = coco.loadImgs(img_id)[0]
    path = depth_dir / (Path(info["file_name"]).stem + ".npy")
    return np.load(str(path))


def boundary_perimeter(mask: np.ndarray) -> int:
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return int(np.count_nonzero(mask & ~eroded))


def compactness(mask: np.ndarray) -> float:
    area = int(np.count_nonzero(mask))
    perim = boundary_perimeter(mask)
    if perim == 0:
        return 0.0
    return 4.0 * np.pi * area / (perim * perim)


def grad_energy(gray: np.ndarray, mask: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    inside = mag[mask > 0]
    if inside.size == 0:
        return 0.0
    return float(inside.mean())


def normalize(global_raw: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    vals = np.array(list(global_raw.values()))
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo if hi > lo else 1.0
    return {
        k: SCORE_MIN + (SCORE_MAX - SCORE_MIN) * (v - lo) / span
        for k, v in global_raw.items()
    }


def fragments_of(mask: np.ndarray) -> list[np.ndarray]:
    n_cc, labels = cv2.connectedComponents(mask)
    frags = []
    for lab in range(1, n_cc):
        frag = (labels == lab).astype(np.uint8)
        if np.count_nonzero(frag) > 0:
            frags.append(frag)
    return frags


def depth_group_merge(
    frags: list[np.ndarray], depth: np.ndarray, n_groups: int
) -> list[np.ndarray]:
    """Group fragments by unsupervised depth ordering.

    Sort fragments by mean depth, then cut the sorted sequence into
    n_groups contiguous runs at the largest mean-depth gaps. This is what
    a pipeline that trusts depth ordering for fragment identity would do.
    """
    means = [float(depth[f > 0].mean()) for f in frags]
    order = np.argsort(means)
    if n_groups >= len(frags):
        return [frags[i] for i in order]
    sorted_means = np.array([means[i] for i in order])
    gaps = sorted_means[1:] - sorted_means[:-1]
    cut_idx = np.argsort(gaps)[-(n_groups - 1) :]
    bounds = sorted(cut_idx + 1)
    groups = []
    start = 0
    for b in [*bounds, len(order)]:
        merged = np.zeros_like(frags[0])
        for i in order[start:b]:
            merged |= frags[i]
        groups.append(merged)
        start = b
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--dataset",
        default="datasets/20260318_1K_1566",
        help="dataset root relative to the repo",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    data = repo / args.dataset
    ann_file = data / "annotations" / f"instances_{args.split}.json"
    images_dir = data / "images" / args.split
    depth_dir = data / "depth" / args.split

    coco = COCO(str(ann_file))

    # One pass collects per-image GT masks, per-instance raw scores and the
    # fragment structures, so scoring variants are pure re-labellings.
    gt_masks: dict[int, list[np.ndarray]] = {}
    raw: dict[str, dict[tuple[int, int], float]] = {
        "random": {},
        "area": {},
        "compactness": {},
        "grad_energy": {},
    }
    frag_masks: dict[int, list[np.ndarray]] = {}  # CC split of each GT
    frag_owner: dict[int, list[int]] = {}  # GT index of each fragment
    depth_groups: dict[int, list[np.ndarray]] = {}
    n_multi = 0

    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        masks = [coco.annToMask(a).astype(np.uint8) for a in anns]
        gt_masks[img_id] = masks
        gray = load_image(coco, img_id, images_dir)
        h, w = masks[0].shape[:2]
        img_area = float(h * w)
        for gi, m in enumerate(masks):
            key = (img_id, gi)
            raw["random"][key] = float(RNG.uniform(0.0, 1.0))
            raw["area"][key] = float(np.count_nonzero(m)) / img_area
            raw["compactness"][key] = compactness(m)
            raw["grad_energy"][key] = grad_energy(gray, m)
        frags, owners = [], []
        for gi, m in enumerate(masks):
            cc = fragments_of(m)
            if len(cc) > 1:
                n_multi += 1
            frags.extend(cc)
            owners.extend([gi] * len(cc))
        frag_masks[img_id] = frags
        frag_owner[img_id] = owners
        if frags:
            depth = load_depth(coco, img_id, depth_dir)
            depth_groups[img_id] = depth_group_merge(frags, depth, len(masks))

    total = sum(len(v) for v in gt_masks.values())
    n_frag = sum(len(v) for v in frag_masks.values())

    scores: dict[str, dict[tuple[int, int], float]] = {
        name: normalize(vals) for name, vals in raw.items()
    }
    scores["const_0.5"] = {k: 0.5 for k in scores["random"]}

    def run_gt(scheme: dict[tuple[int, int], float], tag: str) -> dict:
        results = []
        for img_id, masks in gt_masks.items():
            s = [scheme[(img_id, gi)] for gi in range(len(masks))]
            s = [max(v, SCORE_FLOOR + 1e-6) for v in s]
            results += masks_to_coco_results(
                image_id=img_id,
                masks=masks,
                scores=s,
                category_id=CATEGORY_ID,
            )
        return evaluate_json(ann_file, results)

    def oracle_noised(sigma: float) -> dict[tuple[int, int], float]:
        return {
            k: float(np.clip(1.0 + RNG.normal(0.0, sigma), 0.06, 1.0))
            for k in scores["random"]
        }

    out: dict = {
        "dataset": str(data),
        "split": args.split,
        "n_images": len(gt_masks),
        "n_gt_instances": total,
        "n_multi_cc_gt": n_multi,
        "multi_cc_fraction": round(n_multi / total, 4),
        "n_fragments": n_frag,
        "scoring": {},
        "fragmentation": {},
    }

    order = [
        "const_0.5",
        "random",
        "area",
        "compactness",
        "grad_energy",
        "oracle_1.0",
        "oracle_noise_sigma_0.1",
        "oracle_noise_sigma_0.3",
    ]
    for name in order[:5]:
        out["scoring"][name] = run_gt(scores[name], name)
    out["scoring"]["oracle_1.0"] = run_gt({k: 1.0 for k in scores["random"]}, "oracle")
    out["scoring"]["oracle_noise_sigma_0.1"] = run_gt(oracle_noised(0.1), "noise01")
    out["scoring"]["oracle_noise_sigma_0.3"] = run_gt(oracle_noised(0.3), "noise03")

    def run_frag(masks_by_img: dict[int, list[np.ndarray]], tag: str):
        results = []
        for img_id, masks in masks_by_img.items():
            s = [0.9] * len(masks)
            results += masks_to_coco_results(
                image_id=img_id,
                masks=masks,
                scores=s,
                category_id=CATEGORY_ID,
            )
        return evaluate_json(ann_file, results)

    merged: dict[int, list[np.ndarray]] = {}
    for img_id, frags in frag_masks.items():
        owners = frag_owner[img_id]
        by_gt: dict[int, np.ndarray] = {}
        for f, o in zip(frags, owners, strict=True):
            by_gt[o] = by_gt.get(o, np.zeros_like(f)) | f
        merged[img_id] = [by_gt[g] for g in sorted(by_gt)]

    out["fragmentation"]["cc_split_oracle_conf"] = run_frag(frag_masks, "cc_split")
    out["fragmentation"]["merge_by_gt_oracle"] = run_frag(merged, "merge_gt")
    out["fragmentation"]["merge_by_depth_order"] = run_frag(depth_groups, "merge_depth")

    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
