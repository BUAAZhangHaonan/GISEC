#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure sibling utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
import sys

if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from normalization_stats import load_dataset_normalization_stats
from unet_instance_models import (
    build_instance_model,
    instances_from_boundary_logits,
    instances_from_distance_logits,
    instances_from_semantic_logits,
    merge_fragment_graph,
)


def recommend_main_process_threads(cpu_count: int | None, num_workers: int) -> int:
    cpu_count = max(1, int(cpu_count or 1))
    if int(num_workers) <= 0:
        return max(1, min(8, cpu_count))
    return max(1, min(4, cpu_count // max(2, int(num_workers) * 2)))


def _configure_process_threads(thread_count: int) -> None:
    thread_count = max(1, int(thread_count))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)
    try:
        cv2.setNumThreads(thread_count)
    except Exception:
        pass
    try:
        torch.set_num_threads(thread_count)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _worker_init_fn(_worker_id: int) -> None:
    _configure_process_threads(1)


def build_loader_kwargs(num_workers: int, use_cuda: bool) -> Dict[str, Any]:
    num_workers = int(num_workers)
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(use_cuda),
        "persistent_workers": bool(num_workers > 0),
        "collate_fn": _collate,
    }
    if num_workers > 0:
        kwargs["worker_init_fn"] = _worker_init_fn
        kwargs["prefetch_factor"] = 2
    return kwargs


def _ann_to_mask(ann: Dict[str, Any], h: int, w: int) -> np.ndarray:
    from pycocotools import mask as mask_utils

    segm = ann.get("segmentation")
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm, dict):
        rle = segm
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segm)}")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def _resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def _load_depth_array(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def _mask_to_bbox_aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 1.0
    w = max(1, int(xs.max()) - int(xs.min()) + 1)
    h = max(1, int(ys.max()) - int(ys.min()) + 1)
    return float(w) / float(h)


def load_reference_bank(reference_root: str, image_size: int) -> Dict[str, Any]:
    root = Path(reference_root)
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    mask_dir = root / "mask"
    for required in [rgb_dir, depth_dir, mask_dir]:
        if not required.exists():
            raise FileNotFoundError(f"Reference directory not found: {required}")

    rgb_files = {p.stem: p for p in sorted(rgb_dir.glob("*")) if p.is_file()}
    depth_files = {p.stem: p for p in sorted(depth_dir.glob("*.npy")) if p.is_file()}
    mask_files = {p.stem: p for p in sorted(mask_dir.glob("*")) if p.is_file()}
    view_ids = sorted(set(rgb_files) & set(depth_files) & set(mask_files))
    if not view_ids:
        raise FileNotFoundError(f"No matched rgb/depth/mask reference views found under {root}")

    images, depths, masks = [], [], []
    area_ratios, aspect_ratios = [], []
    for view_id in view_ids:
        rgb = cv2.imread(str(rgb_files[view_id]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_files[view_id])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = _load_depth_array(depth_files[view_id])
        mask = cv2.imread(str(mask_files[view_id]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_files[view_id])
        mask = (mask > 0).astype(np.uint8)

        rgb = _resize_rgb(rgb, image_size)
        depth = _resize_mask(depth, image_size).astype(np.float32)
        mask = _resize_mask(mask, image_size).astype(np.uint8)

        images.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0)
        depths.append(torch.from_numpy(depth[None, ...]).float())
        masks.append(torch.from_numpy(mask[None, ...]).float())
        area_ratios.append(float(mask.mean()))
        aspect_ratios.append(_mask_to_bbox_aspect(mask))

    return {
        "view_ids": view_ids,
        "images": torch.stack(images, dim=0),
        "depths": torch.stack(depths, dim=0),
        "masks": torch.stack(masks, dim=0),
        "shape_stats": {
            "mean_area_ratio": float(np.mean(area_ratios)),
            "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        },
    }


def _build_affinity_target(instance_map: np.ndarray) -> np.ndarray:
    instance_map = instance_map.astype(np.int32)
    affinity = np.zeros((2, instance_map.shape[0], instance_map.shape[1]), dtype=np.float32)
    right_same = (instance_map[:, :-1] > 0) & (instance_map[:, :-1] == instance_map[:, 1:])
    down_same = (instance_map[:-1, :] > 0) & (instance_map[:-1, :] == instance_map[1:, :])
    affinity[0, :, :-1] = right_same.astype(np.float32)
    affinity[1, :-1, :] = down_same.astype(np.float32)
    return affinity


class ECCUnetDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        image_size: int,
        train: bool,
        variant: str,
        *,
        rgb_mean: List[float] | None = None,
        rgb_std: List[float] | None = None,
        depth_clip_min: float | None = None,
        depth_clip_max: float | None = None,
    ):
        from pycocotools.coco import COCO

        self.root = Path(dataset_root)
        self.split = split
        self.image_size = int(image_size)
        self.train = bool(train)
        self.variant = variant
        self.coco = COCO(str(self.root / "annotations" / f"instances_{split}.json"))
        self.image_ids = sorted(self.coco.getImgIds())
        depth_candidates = [
            self.root / "depth" / split,
            self.root / "depth" / "depth_npy" / split,
        ]
        self.depth_dir = next((p for p in depth_candidates if p.exists()), None)
        stats = load_dataset_normalization_stats(str(self.root))
        self.rgb_mean = torch.tensor(
            list(rgb_mean) if rgb_mean is not None else list(stats.rgb_mean_rgb_255),
            dtype=torch.float32,
        ).view(3, 1, 1)
        self.rgb_std = torch.tensor(
            [max(float(v), 1.0) for v in (rgb_std if rgb_std is not None else list(stats.rgb_std_rgb_255))],
            dtype=torch.float32,
        ).view(3, 1, 1)
        self.depth_clip_min = float(stats.depth_clip_min if depth_clip_min is None else depth_clip_min)
        self.depth_clip_max = float(stats.depth_clip_max if depth_clip_max is None else depth_clip_max)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = int(self.image_ids[idx])
        info = self.coco.loadImgs([img_id])[0]
        image = cv2.imread(str(self.root / "images" / self.split / info["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(info["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        fg = np.zeros((h, w), dtype=np.uint8)
        boundary = np.zeros((h, w), dtype=np.uint8)
        distance = np.zeros((h, w), dtype=np.float32)
        instance_map = np.zeros((h, w), dtype=np.int32)
        for inst_id, ann in enumerate(anns, start=1):
            mask = _ann_to_mask(ann, h, w)
            fg = np.maximum(fg, mask)
            instance_map[mask > 0] = inst_id
            if "semantic" in self.variant:
                continue
            if "boundary" in self.variant:
                dilated = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
                eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
                boundary = np.maximum(boundary, (dilated - eroded).clip(min=0))
            else:
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
                if dist.max() > 0:
                    dist = dist / dist.max()
                distance = np.maximum(distance, dist.astype(np.float32))

        if self.train and np.random.rand() < 0.5:
            image = image[:, ::-1].copy()
            fg = fg[:, ::-1].copy()
            boundary = boundary[:, ::-1].copy()
            distance = distance[:, ::-1].copy()
            instance_map = instance_map[:, ::-1].copy()

        depth = None
        if self.depth_dir is not None:
            depth_path = self.depth_dir / f"{Path(info['file_name']).stem}.npy"
            if depth_path.exists():
                depth = _load_depth_array(depth_path)

        if (h, w) != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            fg = cv2.resize(fg, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            boundary = cv2.resize(boundary, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            distance = cv2.resize(distance, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            instance_map = cv2.resize(instance_map, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            if depth is not None:
                depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_tensor = (image_tensor - self.rgb_mean) / self.rgb_std
        fg_tensor = torch.from_numpy(fg[None, ...]).float()
        if "semantic" in self.variant:
            aux_tensor = torch.zeros_like(fg_tensor)
        elif "boundary" in self.variant:
            aux_tensor = torch.from_numpy(boundary[None, ...]).float()
        else:
            aux_tensor = torch.from_numpy(distance[None, ...]).float()
        affinity_tensor = torch.from_numpy(_build_affinity_target(instance_map)).float()

        result = {
            "image_id": img_id,
            "file_name": info["file_name"],
            "orig_size": (h, w),
            "image": image_tensor,
            "fg_target": fg_tensor,
            "aux_target": aux_tensor,
            "affinity_target": affinity_tensor,
        }
        if depth is not None:
            depth_tensor = torch.from_numpy(depth[None, ...]).float()
            depth_tensor = depth_tensor.clamp(min=self.depth_clip_min, max=self.depth_clip_max)
            denom = max(self.depth_clip_max - self.depth_clip_min, 1e-6)
            depth_tensor = (depth_tensor - self.depth_clip_min) / denom
            result["depth"] = depth_tensor
        return result


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "image_ids": [x["image_id"] for x in batch],
        "file_names": [x["file_name"] for x in batch],
        "orig_sizes": [x["orig_size"] for x in batch],
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "fg_target": torch.stack([x["fg_target"] for x in batch], dim=0),
        "aux_target": torch.stack([x["aux_target"] for x in batch], dim=0),
        "affinity_target": torch.stack([x["affinity_target"] for x in batch], dim=0),
    }
    if "depth" in batch[0]:
        result["depths"] = torch.stack([x["depth"] for x in batch], dim=0)
    return result


def _count_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def _encode_results(
    *,
    variant: str,
    image_id: int,
    fg_logits: np.ndarray,
    aux_logits: np.ndarray,
    affinity_logits: np.ndarray | None,
    orig_size: Tuple[int, int],
    min_area: int,
    reference_shape_stats: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    from pycocotools import mask as mask_utils

    orig_h, orig_w = orig_size
    fg_logits = cv2.resize(fg_logits, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    aux_logits = cv2.resize(aux_logits, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    if "reference" in variant:
        fg_mask = (1.0 / (1.0 + np.exp(-fg_logits)) >= 0.5).astype(np.uint8)
        boundary_prob = cv2.resize(1.0 / (1.0 + np.exp(-aux_logits)), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        fragments = cv2.connectedComponents((fg_mask & (boundary_prob < 0.5)).astype(np.uint8), connectivity=8)[1]
        if fragments.max() <= 1:
            fragments = cv2.connectedComponents(fg_mask.astype(np.uint8), connectivity=8)[1]
        affinity_prob = np.zeros((2, orig_h, orig_w), dtype=np.float32)
        if affinity_logits is not None:
            affinity_prob = 1.0 / (1.0 + np.exp(-affinity_logits))
            affinity_prob = np.stack(
                [
                    cv2.resize(affinity_prob[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR),
                    cv2.resize(affinity_prob[1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR),
                ],
                axis=0,
            )
        pair_scores: Dict[Tuple[int, int], float] = {}
        if reference_shape_stats is not None:
            labels = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
            for i, a in enumerate(labels):
                for b in labels[i + 1 :]:
                    merged = ((fragments == a) | (fragments == b)).astype(np.uint8)
                    area_ratio = float(merged.mean())
                    aspect = _mask_to_bbox_aspect(merged)
                    area_score = np.exp(-abs(np.log((area_ratio + 1e-6) / (reference_shape_stats["mean_area_ratio"] + 1e-6))))
                    aspect_score = np.exp(-abs(np.log((aspect + 1e-6) / (reference_shape_stats["mean_aspect_ratio"] + 1e-6))))
                    pair_scores[(a, b)] = float((area_score + aspect_score) / 2.0)
        merged = merge_fragment_graph(
            fragments=fragments,
            boundary_prob=boundary_prob,
            affinity_prob=affinity_prob,
            shape_consistency=pair_scores,
            merge_threshold=0.6,
        )
        masks = [(merged == label_id).astype(np.uint8) for label_id in sorted(x for x in np.unique(merged).tolist() if x > 0) if int((merged == label_id).sum()) >= int(min_area)]
    elif "semantic" in variant:
        masks = instances_from_semantic_logits(
            fg_logits=fg_logits,
            min_area=min_area,
        )
    elif "boundary" in variant:
        masks = instances_from_boundary_logits(
            fg_logits=fg_logits,
            boundary_logits=aux_logits,
            min_area=min_area,
        )
    else:
        masks = instances_from_distance_logits(
            fg_logits=fg_logits,
            distance_logits=aux_logits,
            min_area=min_area,
        )

    rows: List[Dict[str, Any]] = []
    fg_prob = 1.0 / (1.0 + np.exp(-fg_logits))
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        score = float(fg_prob[mask > 0].mean())
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "score": score,
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "segmentation": rle,
            }
        )
    return rows


@torch.no_grad()
def run_eval(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    variant: str,
    ann_file: Path,
    results_json: Path,
    iteration: int,
    min_area: int,
    max_images: int | None = None,
    reference_cache: Dict[str, torch.Tensor] | None = None,
    reference_shape_stats: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    from coco_eval_results import evaluate_coco_results

    model.eval()
    rows: List[Dict[str, Any]] = []
    seen = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=device.type == "cuda")
        depths = batch.get("depths")
        if depths is not None:
            depths = depths.to(device, non_blocking=device.type == "cuda")
        if "reference" in variant:
            fg_logits, aux_logits, affinity_logits = model(images, query_depth=depths, reference_cache=reference_cache)
        else:
            fg_logits, aux_logits = model(images)
            affinity_logits = None
        fg_logits_np = fg_logits.squeeze(1).cpu().numpy()
        aux_logits_np = aux_logits.squeeze(1).cpu().numpy()
        affinity_logits_np = None if affinity_logits is None else affinity_logits.cpu().numpy()
        iterable = zip(
            batch["image_ids"],
            batch["orig_sizes"],
            fg_logits_np,
            aux_logits_np,
            [None] * len(batch["image_ids"]) if affinity_logits_np is None else affinity_logits_np,
        )
        for image_id, orig_size, fg_pred, aux_pred, aff_pred in iterable:
            if max_images is not None and seen >= int(max_images):
                break
            rows.extend(
                _encode_results(
                    variant=variant,
                    image_id=int(image_id),
                    fg_logits=fg_pred,
                    aux_logits=aux_pred,
                    affinity_logits=aff_pred,
                    orig_size=orig_size,
                    min_area=min_area,
                    reference_shape_stats=reference_shape_stats,
                )
            )
            seen += 1
        if max_images is not None and seen >= int(max_images):
            break

    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(rows), encoding="utf-8")
    return evaluate_coco_results(ann_file=ann_file, results_json=results_json, iteration=iteration)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-train-steps", type=int, default=0)
    ap.add_argument("--max-val-images", type=int, default=0)
    ap.add_argument("--reference-root", type=str, default="")
    ap.add_argument("--rgb-mean", type=str, default="")
    ap.add_argument("--rgb-std", type=str, default="")
    ap.add_argument("--depth-clip-min", type=float, default=None)
    ap.add_argument("--depth-clip-max", type=float, default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    _configure_process_threads(recommend_main_process_threads(os.cpu_count(), args.num_workers))
    use_cuda = device.type == "cuda"
    reference_bank = None
    reference_cache = None
    if args.reference_root:
        reference_bank = load_reference_bank(args.reference_root, image_size=args.image_size)

    rgb_mean = json.loads(args.rgb_mean) if args.rgb_mean else None
    rgb_std = json.loads(args.rgb_std) if args.rgb_std else None
    train_ds = ECCUnetDataset(
        args.dataset_root,
        "train",
        args.image_size,
        True,
        args.variant,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        depth_clip_min=args.depth_clip_min,
        depth_clip_max=args.depth_clip_max,
    )
    val_ds = ECCUnetDataset(
        args.dataset_root,
        "val",
        args.image_size,
        False,
        args.variant,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        depth_clip_min=args.depth_clip_min,
        depth_clip_max=args.depth_clip_max,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        **build_loader_kwargs(args.num_workers, use_cuda=use_cuda),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        **build_loader_kwargs(args.num_workers, use_cuda=use_cuda),
    )

    model = build_instance_model(args.variant, in_channels=3, base_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    if args.variant == "unet_reference_inst":
        if reference_bank is None:
            raise FileNotFoundError("--reference-root is required for unet_reference_inst")
        reference_cache = model.build_reference_cache(reference_bank, device)  # type: ignore[attr-defined]
    start = time.time()

    (output_dir / "params_trainable.txt").write_text(str(_count_params(model)) + "\n", encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()

    ann_file = Path(args.dataset_root) / "annotations" / "instances_val.json"
    best_ap = -1.0
    best_epoch = 1
    best_ckpt = output_dir / "model_0000001.pth"

    total_train_steps = 0
    stop_after_epoch = False
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_steps = 0
        epoch_start = time.time()
        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=use_cuda)
            depths = batch.get("depths")
            if depths is not None:
                depths = depths.to(device, non_blocking=use_cuda)
            fg_target = batch["fg_target"].to(device, non_blocking=use_cuda)
            aux_target = batch["aux_target"].to(device, non_blocking=use_cuda)
            affinity_target = batch["affinity_target"].to(device, non_blocking=use_cuda)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                if args.variant == "unet_reference_inst":
                    fg_logits, aux_logits, affinity_logits = model(images, query_depth=depths, reference_cache=reference_cache)
                else:
                    fg_logits, aux_logits = model(images)
                    affinity_logits = None
                loss_fg = F.binary_cross_entropy_with_logits(fg_logits, fg_target)
                if args.variant == "unet_reference_inst":
                    loss_aux = F.binary_cross_entropy_with_logits(aux_logits, aux_target)
                    loss_aff = F.binary_cross_entropy_with_logits(affinity_logits, affinity_target)
                elif "semantic" in args.variant:
                    loss_aux = torch.zeros((), device=device)
                    loss_aff = torch.zeros((), device=device)
                elif "boundary" in args.variant:
                    loss_aux = F.binary_cross_entropy_with_logits(aux_logits, aux_target)
                    loss_aff = torch.zeros((), device=device)
                else:
                    loss_aux = F.l1_loss(torch.sigmoid(aux_logits), aux_target)
                    loss_aff = torch.zeros((), device=device)
                loss = loss_fg + loss_aux + 0.5 * loss_aff
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_steps += 1
            total_train_steps += 1
            if train_steps % 50 == 0:
                print(
                    f"[unet-instance] epoch={epoch} step={train_steps} loss={float(loss.detach().cpu()):.4f}",
                    flush=True,
                )
            if int(args.max_train_steps) > 0 and total_train_steps >= int(args.max_train_steps):
                stop_after_epoch = True
                break

        epoch_results_path = output_dir / f"epoch_{epoch:04d}_results.json"
        eval_start = time.time()
        metrics = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            variant=args.variant,
            ann_file=ann_file,
            results_json=epoch_results_path,
            iteration=epoch,
            min_area=args.min_area,
            max_images=int(args.max_val_images) if int(args.max_val_images) > 0 else None,
            reference_cache=reference_cache,
            reference_shape_stats=None if reference_bank is None else reference_bank["shape_stats"],
        )
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        segm_ap = float(metrics.get("segm/AP", 0.0))
        if segm_ap >= best_ap:
            best_ap = segm_ap
            best_epoch = epoch
            best_ckpt = output_dir / f"model_{epoch:07d}.pth"
            torch.save(model.state_dict(), best_ckpt)
        print(
            f"[unet-instance] epoch={epoch} train_sec={time.time() - epoch_start:.2f} "
            f"eval_sec={time.time() - eval_start:.2f} best_ap={best_ap:.4f}",
            flush=True,
        )
        if stop_after_epoch:
            break

    final_ckpt = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_ckpt)

    final_results_path = output_dir / "coco_instances_results.json"
    final_metrics = run_eval(
        model=model,
        loader=val_loader,
        device=device,
        variant=args.variant,
        ann_file=ann_file,
        results_json=final_results_path,
        iteration=args.epochs,
        min_area=args.min_area,
        max_images=int(args.max_val_images) if int(args.max_val_images) > 0 else None,
        reference_cache=reference_cache,
        reference_shape_stats=None if reference_bank is None else reference_bank["shape_stats"],
    )
    (output_dir / "metrics.cocoeval.json").write_text(json.dumps(final_metrics, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "last_checkpoint").write_text(final_ckpt.name + "\n", encoding="utf-8")
    (output_dir / "wall_time_sec.txt").write_text(str(int(time.time() - start)) + "\n", encoding="utf-8")
    print(f"[unet-instance] best_epoch={best_epoch} best_ap={best_ap:.4f}")


if __name__ == "__main__":
    main()
