#!/usr/bin/env python3
"""
Run UCN (Unseen Object Clustering Network) baseline on ECC datasets (0831 / 0909)
and evaluate with COCOeval (segm/bbox AP).

This baseline is not Detectron2-based. We:
- train the embedding network on ECC instance masks converted to instance-id labels
- run mean-shift clustering on embeddings to obtain instance masks
- export COCO results to `coco_instances_results.json` and compute COCOeval metrics
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# Ensure sibling baseline utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from depth_stats import load_0831_1k_depth_stats, load_0909_512_depth_stats, load_depth_stats_for_dataset_root
from ecc_datasets import normalize_register
from normalization_stats import load_dataset_normalization_stats
from rgbd_geometry import depth_to_xyz
from ucn_coco_utils import coco_eval_stats, encode_binary_mask_rle, write_json


def _workspace_root() -> Path:
    # File: <ws>/magformer/baselines/run_ucn_ecc.py
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dataset_root(register: str) -> Path:
    ws = _workspace_root()
    if register == "0831":
        return ws / "magformer_datasets" / "0831_1K"
    if register == "0909":
        return ws / "magformer_datasets" / "0909_512_0.12K"
    raise ValueError(f"Custom register requires explicit --dataset-root: {register}")


def _add_ucn_lib_to_syspath() -> Path:
    ucn_lib = _repo_root() / "baselines" / "unseen_object_clustering" / "lib"
    if str(ucn_lib) not in sys.path:
        sys.path.insert(0, str(ucn_lib))
    return ucn_lib


def _load_pixel_mean_bgr_255(register: str, dataset_root: str | None = None) -> List[float]:
    repo_root = _repo_root()
    if register == "0831":
        p = repo_root / "configs" / "stats" / "0831_1k_rgb_stats.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        mean_bgr = d.get("mean_bgr")
        if not isinstance(mean_bgr, list) or len(mean_bgr) != 3:
            raise ValueError(f"Invalid mean_bgr in: {p}")
        return [float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])]
    if register == "0909":
        p = repo_root / "configs" / "stats" / "0909_512_rgb_stats.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        mean_bgr = d.get("mean_bgr")
        if not isinstance(mean_bgr, list) or len(mean_bgr) != 3:
            raise ValueError(f"Invalid mean_bgr in: {p}")
        return [float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])]
    if dataset_root is None:
        raise ValueError(f"Custom register requires explicit dataset_root for RGB stats: {register}")
    stats = load_dataset_normalization_stats(dataset_root)
    return [
        float(stats.rgb_mean_bgr_255[0]),
        float(stats.rgb_mean_bgr_255[1]),
        float(stats.rgb_mean_bgr_255[2]),
    ]


def _default_ucn_pretrained_path() -> Path:
    return _repo_root() / "output" / "pretrained" / "seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth"


def _load_depth_clip(register: str, dataset_root: str | None = None) -> Tuple[float, float]:
    if register == "0831":
        stats = load_0831_1k_depth_stats()
    elif register == "0909":
        stats = load_0909_512_depth_stats()
    else:
        if dataset_root is None:
            raise ValueError(f"Custom register requires explicit dataset_root for depth stats: {register}")
        stats = load_depth_stats_for_dataset_root(dataset_root)
    return float(stats.p1), float(stats.p99)


def _ann_to_mask(ann: Dict[str, Any], h: int, w: int) -> np.ndarray:
    segm = ann.get("segmentation")
    if segm is None:
        raise KeyError("annotation missing `segmentation`")
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm, dict):
        rle = segm
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segm)}")
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8)


@dataclass(frozen=True)
class UCNRecipe:
    input_type: str
    fusion_type: str
    num_units: int
    learning_rate: float
    weight_decay: float
    embedding_pretrain: bool
    embedding_normalization: bool
    embedding_metric: str
    embedding_alpha: float
    embedding_delta: float
    embedding_lambda_intra: float
    embedding_lambda_inter: float
    chromatic: bool
    add_noise: bool
    batch_size: int
    num_seeds: int
    kappa: float


def build_ucn_recipe(register: str) -> UCNRecipe:
    _ = register
    return UCNRecipe(
        input_type="RGBD",
        fusion_type="add",
        num_units=64,
        learning_rate=1.0e-5,
        weight_decay=5.0e-4,
        embedding_pretrain=False,
        embedding_normalization=True,
        embedding_metric="cosine",
        embedding_alpha=0.02,
        embedding_delta=0.5,
        embedding_lambda_intra=10.0,
        embedding_lambda_inter=10.0,
        chromatic=True,
        add_noise=True,
        batch_size=16,
        num_seeds=100,
        kappa=20.0,
    )


@dataclass(frozen=True)
class ECCPaths:
    root: Path

    def images_dir(self, split: str) -> Path:
        return self.root / "images" / split

    def depth_dir(self, split: str) -> Path:
        candidates = [
            self.root / "depth" / "depth_npy" / split,
            self.root / "depth" / split,
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def ann_file(self, split: str) -> Path:
        return self.root / "annotations" / f"instances_{split}.json"


class ECCUCNDataset(torch.utils.data.Dataset):
    """
    Produce UCN-style samples:
      - image_color: float32 tensor (3,H,W) in BGR order, normalized (img/255 - mean/255)
      - depth: float32 tensor (3,H,W) in XYZ geometry derived from metric depth
      - label: int64 tensor (1,H,W) with instance ids in {0..K-1}, background=-1
    """

    def __init__(
        self,
        dataset_root: str,
        split: str,
        img_size: int,
        train: bool,
        pixel_mean_bgr_255: List[float],
        depth_clip: Tuple[float, float],
    ):
        self.paths = ECCPaths(Path(dataset_root))
        self.split = split
        self.img_size = int(img_size)
        self.train = bool(train)
        self.coco = COCO(str(self.paths.ann_file(split)))
        self.image_ids = sorted(self.coco.getImgIds())
        self.pixel_mean = torch.tensor(np.asarray(pixel_mean_bgr_255, dtype=np.float32) / 255.0).view(1, 1, 3)
        self.depth_min = float(depth_clip[0])
        self.depth_max = float(depth_clip[1])

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_rgb(self, file_name: str) -> np.ndarray:
        p = self.paths.images_dir(self.split) / file_name
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        return im  # BGR uint8

    def _load_depth(self, file_name: str) -> np.ndarray:
        stem = Path(file_name).stem
        p = self.paths.depth_dir(self.split) / f"{stem}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing depth: {p}")
        depth = np.load(str(p)).astype(np.float32)
        depth = np.clip(depth, self.depth_min, self.depth_max)
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min + 1e-6)
        return depth_to_xyz(depth)

    def _build_label_map(self, img_id: int, h: int, w: int) -> np.ndarray:
        label = -1 * np.ones((h, w), dtype=np.int32)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        for inst_id, ann in enumerate(anns):
            m = _ann_to_mask(ann, h, w)
            label[m > 0] = inst_id
        return label

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = int(self.image_ids[idx])
        img_info = self.coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        orig_h, orig_w = int(img_info["height"]), int(img_info["width"])

        rgb = self._load_rgb(file_name)
        depth = self._load_depth(file_name)
        label = self._build_label_map(img_id, orig_h, orig_w)

        # Basic augmentation: random horizontal flip (train only).
        if self.train and random.random() < 0.5:
            rgb = rgb[:, ::-1, :].copy()
            depth = depth[:, ::-1, :].copy()
            label = label[:, ::-1].copy()

        # Resize to fixed square for baseline protocol.
        if (orig_h, orig_w) != (self.img_size, self.img_size):
            rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        label_blob = torch.from_numpy(label).unsqueeze(0).to(torch.int64)

        im_tensor = torch.from_numpy(rgb).to(torch.float32) / 255.0
        im_tensor -= self.pixel_mean
        image_blob = im_tensor.permute(2, 0, 1).contiguous()

        depth_blob = torch.from_numpy(depth).to(torch.float32).permute(2, 0, 1).contiguous()

        return {
            "image_id": img_id,
            "file_name": file_name,
            "orig_size": (orig_h, orig_w),
            "image_color": image_blob,
            "depth": depth_blob,
            "label": label_blob,
        }


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def _cluster_to_instances(
    cluster_map: np.ndarray,
    image_id: int,
    min_area: int,
    max_instances: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    labels = [int(x) for x in np.unique(cluster_map).tolist() if int(x) > 0]

    # Sort by area desc for stable truncation.
    labels = sorted(labels, key=lambda k: int((cluster_map == k).sum()), reverse=True)
    for k in labels[:max_instances]:
        m = (cluster_map == k).astype(np.uint8)
        area = int(m.sum())
        if area < min_area:
            continue
        rle = encode_binary_mask_rle(m)
        bbox = [float(x) for x in mask_utils.toBbox(rle).tolist()]
        results.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "segmentation": rle,
                "bbox": bbox,
                "score": 1.0,
            }
        )
    return results


def evaluate_ucn(
    network: torch.nn.Module,
    dataset: ECCUCNDataset,
    out_dir: Path,
    downsample: int,
    num_seeds: int,
    kappa: float,
    min_area: int,
    max_instances: int,
) -> Dict[str, Any]:
    from utils.mean_shift import mean_shift_smart_init

    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    network.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    start = time.time()
    with torch.no_grad():
        for i, sample in enumerate(loader):
            image = sample["image_color"].cuda()
            depth = sample["depth"].cuda()
            label = sample["label"].cuda()
            img_id = int(sample["image_id"].item())
            orig_h, orig_w = int(sample["orig_size"][0].item()), int(sample["orig_size"][1].item())

            feat = network(image, label, depth)  # (1,C,H,W)
            feat_ds = F.interpolate(feat, size=(downsample, downsample), mode="bilinear", align_corners=False)
            x = feat_ds[0].permute(1, 2, 0).reshape(-1, feat_ds.shape[1])
            x = F.normalize(x, p=2, dim=1)

            cluster_labels, _ = mean_shift_smart_init(
                x, kappa=kappa, num_seeds=num_seeds, max_iters=10, metric="cosine"
            )
            cluster_map = cluster_labels.view(downsample, downsample).cpu().numpy().astype(np.int32)
            cluster_map = cv2.resize(cluster_map, (dataset.img_size, dataset.img_size), interpolation=cv2.INTER_NEAREST)
            cluster_map = cv2.resize(cluster_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            results.extend(_cluster_to_instances(cluster_map, img_id, min_area=min_area, max_instances=max_instances))

            if (i + 1) % 10 == 0:
                dt = time.time() - start
                print(f"[ucn-eval] {i+1}/{len(dataset)} images, elapsed={dt:.1f}s, results={len(results)}")

    dt_path = out_dir / "coco_instances_results.json"
    write_json(str(dt_path), results)

    segm = coco_eval_stats(str(dataset.paths.ann_file("val")), str(dt_path), iou_type="segm")
    bbox = coco_eval_stats(str(dataset.paths.ann_file("val")), str(dt_path), iou_type="bbox")

    # Match Detectron2's metric scale: COCOeval stats are [0,1], Detectron2 logs [0,100].
    segm = {k: float(v) * 100.0 for k, v in segm.items()}
    bbox = {k: float(v) * 100.0 for k, v in bbox.items()}

    metrics = {
        "iteration": -1,
        "segm/AP": float(segm["AP"]),
        "segm/AP50": float(segm["AP50"]),
        "segm/AP75": float(segm["AP75"]),
        "segm/APs": float(segm["APs"]),
        "segm/APm": float(segm["APm"]),
        "segm/APl": float(segm["APl"]),
        "bbox/AP": float(bbox["AP"]),
        "bbox/AP50": float(bbox["AP50"]),
        "bbox/AP75": float(bbox["AP75"]),
        "bbox/APs": float(bbox["APs"]),
        "bbox/APm": float(bbox["APm"]),
        "bbox/APl": float(bbox["APl"]),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    default_recipe = build_ucn_recipe("0831")

    ap = argparse.ArgumentParser()
    ap.add_argument("--register", type=str, default="0831", help="ECC dataset id: 0831 | 0909")
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=default_recipe.batch_size)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=default_recipe.learning_rate)
    ap.add_argument("--num-units", type=int, default=default_recipe.num_units)
    ap.add_argument("--downsample", type=int, default=128)
    ap.add_argument("--num-seeds", type=int, default=default_recipe.num_seeds)
    ap.add_argument("--kappa", type=float, default=default_recipe.kappa)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--max-instances", type=int, default=50)
    ap.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional UCN checkpoint path. Defaults to the official local RGBD-add checkpoint when present.",
    )
    args = ap.parse_args()

    register_id = normalize_register(args.register)
    recipe = build_ucn_recipe(register_id)

    if args.dataset_root is None:
        args.dataset_root = str(_default_dataset_root(register_id))

    pixel_mean_bgr_255 = _load_pixel_mean_bgr_255(register_id, str(args.dataset_root))
    depth_clip = _load_depth_clip(register_id, str(args.dataset_root))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import UCN code (networks/fcn/utils) from the vendored baseline.
    _add_ucn_lib_to_syspath()
    from fcn.config import cfg
    from fcn.train import train_segnet
    import networks

    # Scratch-only policy: disable pretrained weights inside the UCN backbone.
    cfg.INPUT = recipe.input_type
    cfg.TRAIN.FUSION_TYPE = recipe.fusion_type
    cfg.TRAIN.EMBEDDING_PRETRAIN = recipe.embedding_pretrain
    cfg.TRAIN.IMS_PER_BATCH = int(args.batch)
    cfg.TRAIN.NUM_UNITS = int(args.num_units)
    cfg.TRAIN.LEARNING_RATE = float(args.lr)
    cfg.TRAIN.WEIGHT_DECAY = float(recipe.weight_decay)
    cfg.TRAIN.EMBEDDING_NORMALIZATION = recipe.embedding_normalization
    cfg.TRAIN.EMBEDDING_METRIC = recipe.embedding_metric
    cfg.TRAIN.EMBEDDING_ALPHA = float(recipe.embedding_alpha)
    cfg.TRAIN.EMBEDDING_DELTA = float(recipe.embedding_delta)
    cfg.TRAIN.EMBEDDING_LAMBDA_INTRA = float(recipe.embedding_lambda_intra)
    cfg.TRAIN.EMBEDDING_LAMBDA_INTER = float(recipe.embedding_lambda_inter)
    cfg.TRAIN.CHROMATIC = recipe.chromatic
    cfg.TRAIN.ADD_NOISE = recipe.add_noise
    cfg.TRAIN.VISUALIZE = False
    cfg.TRAIN.ITERS = 0
    cfg.epochs = int(args.epochs)

    # ECC pixel mean (BGR, 0..255) for UCN's preprocessing convention.
    cfg.PIXEL_MEANS = np.array([[[pixel_mean_bgr_255[0], pixel_mean_bgr_255[1], pixel_mean_bgr_255[2]]]], dtype=np.float32)

    train_ds = ECCUCNDataset(
        dataset_root=args.dataset_root,
        split="train",
        img_size=args.img_size,
        train=True,
        pixel_mean_bgr_255=pixel_mean_bgr_255,
        depth_clip=depth_clip,
    )
    val_ds = ECCUCNDataset(
        dataset_root=args.dataset_root,
        split="val",
        img_size=args.img_size,
        train=False,
        pixel_mean_bgr_255=pixel_mean_bgr_255,
        depth_clip=depth_clip,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(args.batch),
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    pretrained_path = Path(args.pretrained).expanduser() if args.pretrained else _default_ucn_pretrained_path()
    network_data = None
    if pretrained_path.exists():
        network_data = torch.load(str(pretrained_path))
        if isinstance(network_data, dict) and "model" in network_data:
            network_data = network_data["model"]
        print(f"[ucn] loading pretrained checkpoint: {pretrained_path}")
    else:
        print(f"[ucn] no pretrained checkpoint found, training from scratch: {pretrained_path}")

    network = networks.seg_resnet34_8s_embedding(
        num_classes=2,
        num_units=cfg.TRAIN.NUM_UNITS,
        data=network_data,
    ).cuda()
    network = torch.nn.DataParallel(network).cuda()

    (out_dir / "params_trainable.txt").write_text(str(_count_trainable_params(network)) + "\n", encoding="utf-8")

    param_groups = [
        {"params": network.module.bias_parameters(), "weight_decay": cfg.TRAIN.WEIGHT_DECAY},
        {"params": network.module.weight_parameters(), "weight_decay": cfg.TRAIN.WEIGHT_DECAY},
    ]
    optimizer = torch.optim.Adam(param_groups, cfg.TRAIN.LEARNING_RATE, betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))

    print(f"[ucn] register={register_id} train={len(train_ds)} val={len(val_ds)} epochs={cfg.epochs} batch={args.batch} img={args.img_size}")
    for epoch in range(cfg.epochs):
        train_segnet(train_loader, network, optimizer, epoch)

    # Final eval + COCO export
    metrics = evaluate_ucn(
        network=network,
        dataset=val_ds,
        out_dir=out_dir,
        downsample=int(args.downsample),
        num_seeds=int(args.num_seeds),
        kappa=float(args.kappa),
        min_area=int(args.min_area),
        max_instances=int(args.max_instances),
    )
    print("[ucn] done metrics:", metrics)


if __name__ == "__main__":
    main()
