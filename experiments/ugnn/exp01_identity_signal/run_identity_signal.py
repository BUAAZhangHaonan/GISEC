"""E1 identity signal: separability of same-instance vs different-instance
connected-component pairs using cheap depth / spatial / appearance features.

Zero training on neural nets: pure offline statistics over GT masks of the
1566 val split (149 frames). Pass bar: pair-classification AUC >= 0.85 or
depth-rule merge accuracy >= 0.9.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask, load_depth_array

MIN_AREA = 5  # ignore speck components below 5 px
HIST_BINS = 16

FEATURES: dict[str, list[str]] = {
    "depth": [
        "d_mean", "d_median", "d_q10", "d_q90", "d_std",
    ],
    "spatial": [
        "centroid_dist", "min_dist", "bbox_gap", "log_area_ratio",
        "norm_centroid_dist",
    ],
    "appearance": [
        "color_l1", "hist_intersect",
    ],
}


def component_features(
    comp: np.ndarray, depth: np.ndarray, rgb: np.ndarray
) -> dict:
    ys, xs = np.nonzero(comp)
    area = float(len(ys))
    dvals = depth[ys, xs]
    dvals = dvals[np.isfinite(dvals)]
    if dvals.size == 0:
        dvals = np.array([np.nan])
    pixels = rgb[ys, xs].astype(np.float32)  # (N, 3) BGR
    hist = [
        np.histogram(pixels[:, c], bins=HIST_BINS, range=(0.0, 255.0))[0]
        for c in range(3)
    ]
    hist = [h / max(area, 1.0) for h in hist]
    x0, y0, w0, h0 = cv2.boundingRect(comp.astype(np.uint8))
    return {
        "area": area,
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "bbox": (x0, y0, x0 + w0, y0 + h0),
        "d_mean": float(np.mean(dvals)),
        "d_median": float(np.median(dvals)),
        "d_q10": float(np.percentile(dvals, 10)),
        "d_q90": float(np.percentile(dvals, 90)),
        "d_std": float(np.std(dvals)),
        "mean_color": pixels.mean(axis=0),
        "hist": hist,
        "comp_mask": comp,
    }


def bbox_gap(b1: tuple, b2: tuple) -> float:
    gx = max(b1[0] - b2[2], b2[0] - b1[2], 0)
    gy = max(b1[1] - b2[3], b2[1] - b1[3], 0)
    return float(np.hypot(gx, gy))


def pair_features(c1: dict, c2: dict) -> dict[str, float]:
    # nearest-pixel distance via a bbox-cropped distance transform
    b1, b2 = c1["bbox"], c2["bbox"]
    x0 = min(b1[0], b2[0])
    y0 = min(b1[1], b2[1])
    x1 = max(b1[2], b2[2])
    y1 = max(b1[3], b2[3])
    m1 = c1["comp_mask"][y0:y1, x0:x1]
    m2 = c2["comp_mask"][y0:y1, x0:x1]
    dt = cv2.distanceTransform(
        (m1 == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    ys2, xs2 = np.nonzero(m2)
    mdist = float(dt[ys2, xs2].min())
    centroid = float(np.hypot(c1["cx"] - c2["cx"], c1["cy"] - c2["cy"]))
    scale = float(np.sqrt(max(c1["area"], c2["area"])))
    color_diff = float(
        np.abs(c1["mean_color"] - c2["mean_color"]).mean() / 255.0
    )
    inter = float(np.mean([
        np.minimum(c1["hist"][c], c2["hist"][c]).sum() for c in range(3)
    ]))
    return {
        "d_mean": abs(c1["d_mean"] - c2["d_mean"]),
        "d_median": abs(c1["d_median"] - c2["d_median"]),
        "d_q10": abs(c1["d_q10"] - c2["d_q10"]),
        "d_q90": abs(c1["d_q90"] - c2["d_q90"]),
        "d_std": abs(c1["d_std"] - c2["d_std"]),
        "centroid_dist": centroid,
        "min_dist": mdist,
        "bbox_gap": bbox_gap(c1["bbox"], c2["bbox"]),
        "log_area_ratio": float(
            abs(np.log(c1["area"] / c2["area"] + 1e-6))
        ),
        "norm_centroid_dist": centroid / (scale + 1e-6),
        "color_l1": color_diff,
        "hist_intersect": inter,
    }


def load_image_components(
    coco: LiteCOCO, image: dict, depth_dir: Path, image_dir: Path
) -> list[dict]:
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image["id"]]))
    h, w = int(image["height"]), int(image["width"])
    stem = Path(image["file_name"]).stem
    depth = load_depth_array(depth_dir / f"{stem}.npy")
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.imread(str(image_dir / image["file_name"]))
    if rgb is None:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
    comps: list[dict] = []
    for ann in anns:
        mask = ann_to_mask(ann, h, w)
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        for lab in range(1, n):
            comp = labels == lab
            if comp.sum() < MIN_AREA:
                continue
            feats = component_features(comp, depth, rgb)
            feats["instance"] = int(ann["id"])
            comps.append(feats)
    return comps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = (
        Path(args.data_root)
        if args.data_root
        else (here / "../../../datasets/20260318_1K_1566").resolve()
    )
    coco = LiteCOCO(root / "annotations/instances_val.json")
    depth_dir = root / "depth/val"
    image_dir = root / "images/val"
    image_ids = coco.getImgIds()

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    groups: list[int] = []
    n_instances = 0
    n_multi = 0
    n_components = 0

    for img_id in image_ids:
        image = coco.loadImgs([img_id])[0]
        comps = load_image_components(coco, image, depth_dir, image_dir)
        per_instance: dict[int, int] = {}
        for c in comps:
            per_instance[c["instance"]] = (
                per_instance.get(c["instance"], 0) + 1
            )
        n_instances += len(per_instance)
        n_multi += sum(1 for v in per_instance.values() if v > 1)
        n_components += len(comps)
        for c1, c2 in combinations(comps, 2):
            rows.append(pair_features(c1, c2))
            labels.append(int(c1["instance"] == c2["instance"]))
            groups.append(int(img_id))
        print(f"img {img_id}: {len(comps)} comps, {len(rows)} pairs",
              flush=True)

    y = np.array(labels)
    groups_arr = np.array(groups)
    feat_names = list(rows[0].keys())
    X = np.array([[r[f] for f in feat_names] for r in rows], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    results: dict = {
        "n_images": len(image_ids),
        "n_instances": n_instances,
        "n_multi_component_instances": n_multi,
        "multi_component_fraction": n_multi / max(n_instances, 1),
        "n_components": n_components,
        "n_pairs": len(y),
        "n_pos_pairs": int(y.sum()),
        "pos_rate": float(y.mean()),
    }

    # single-feature AUC (oriented so that higher score = same instance)
    single_auc: dict[str, float] = {}
    for i, name in enumerate(feat_names):
        score = X[:, i] if name == "hist_intersect" else -X[:, i]
        single_auc[name] = float(roc_auc_score(y, score))
    results["single_feature_auc"] = dict(
        sorted(single_auc.items(), key=lambda kv: -kv[1])
    )

    def combo_auc(layers: list[str]) -> float:
        cols = [
            i
            for i, name in enumerate(feat_names)
            if any(name in FEATURES[layer] for layer in layers)
        ]
        Xs = X[:, cols]
        aucs = []
        for tr, te in GroupKFold(n_splits=5).split(Xs, y, groups_arr):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced"
                ),
            )
            model.fit(Xs[tr], y[tr])
            if len(np.unique(y[te])) < 2:
                continue
            aucs.append(
                roc_auc_score(y[te], model.predict_proba(Xs[te])[:, 1])
            )
        return float(np.mean(aucs))

    results["combo_auc"] = {
        "depth": combo_auc(["depth"]),
        "spatial": combo_auc(["spatial"]),
        "appearance": combo_auc(["appearance"]),
        "depth+spatial": combo_auc(["depth", "spatial"]),
        "depth+appearance": combo_auc(["depth", "appearance"]),
        "spatial+appearance": combo_auc(["spatial", "appearance"]),
        "all": combo_auc(["depth", "spatial", "appearance"]),
    }

    # pure depth rule: same instance iff |mean depth diff| < tau
    dmean = X[:, feat_names.index("d_mean")]
    taus = np.arange(0.001, 0.201, 0.001)
    accs = [accuracy_score(y, (dmean < t).astype(int)) for t in taus]
    best_i = int(np.argmax(accs))
    results["depth_rule"] = {
        "best_tau": float(taus[best_i]),
        "best_accuracy": float(accs[best_i]),
        "auc_of_d_mean": single_auc["d_mean"],
    }

    best_auc = max(results["combo_auc"].values())
    results["pass"] = bool(
        best_auc >= 0.85
        or max(single_auc.values()) >= 0.85
        or results["depth_rule"]["best_accuracy"] >= 0.9
    )

    (here / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
