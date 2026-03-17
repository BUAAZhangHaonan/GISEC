from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from gnn_reference_prior.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch
from gnn_reference_prior.datasets.reference_bank import load_reference_bank
from gnn_reference_prior.models.graph_utils import (
    GraphBatch,
    heuristic_edge_scores,
    merge_instances_from_edge_scores,
)
from gnn_reference_prior.models.reference_cache import cache_to_device
from gnn_reference_prior.models.reference_unet_gnn import ReferenceUNetGNN


VARIANT_FLAGS = {
    "B0": "baseline_heuristic_shape",
    "G1": "graph_boundary_affinity",
    "G2": "graph_boundary_affinity_shape",
    "G3": "graph_rgb_reference",
    "G4": "graph_rgbd_reference",
    "G5": "graph_rgbd_reference_shape",
}


def variant_feature_key(variant: str) -> str:
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"Unsupported variant: {variant}")
    return VARIANT_FLAGS[variant]


def variant_uses_graph(variant: str) -> bool:
    return variant != "B0"


def _encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(rle["size"]), "counts": counts}


def _masks_to_results(image_id: int, masks: List[np.ndarray]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for mask in masks:
        if int(mask.sum()) <= 0:
            continue
        ys, xs = np.nonzero(mask > 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        results.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "score": 1.0,
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "segmentation": _encode_binary_mask(mask.astype(np.uint8)),
            }
        )
    return results


def _fragment_masks_from_merged(merged: np.ndarray, min_area: int) -> List[np.ndarray]:
    masks = []
    for label in [int(x) for x in np.unique(merged).tolist() if int(x) > 0]:
        mask = (merged == label).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            masks.append(mask)
    return masks


def _evaluate_json(ann_file: Path, results_json: Path) -> Dict[str, Any]:
    coco_gt = COCO(str(ann_file))
    raw_results = json.loads(results_json.read_text(encoding="utf-8"))
    if not raw_results:
        payload: Dict[str, Any] = {"iteration": -1}
        for prefix in ["bbox", "segm"]:
            payload[f"{prefix}/AP"] = 0.0
            payload[f"{prefix}/AP50"] = 0.0
            payload[f"{prefix}/AP75"] = 0.0
            payload[f"{prefix}/APs"] = 0.0
            payload[f"{prefix}/APm"] = 0.0
            payload[f"{prefix}/APl"] = 0.0
        return payload
    coco_dt = coco_gt.loadRes(str(results_json))
    payload: Dict[str, Any] = {"iteration": -1}
    for iou_type, prefix in [("bbox", "bbox"), ("segm", "segm")]:
        evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats.tolist()
        payload[f"{prefix}/AP"] = float(stats[0])
        payload[f"{prefix}/AP50"] = float(stats[1])
        payload[f"{prefix}/AP75"] = float(stats[2])
        payload[f"{prefix}/APs"] = float(stats[3])
        payload[f"{prefix}/APm"] = float(stats[4])
        payload[f"{prefix}/APl"] = float(stats[5])
    return payload


def run_eval(
    *,
    model: ReferenceUNetGNN,
    loader: DataLoader,
    device: torch.device,
    reference_cache,
    variant: str,
    ann_file: Path,
    results_json: Path,
    min_area: int,
    edge_threshold: float,
    max_images: int | None = None,
) -> Dict[str, Any]:
    model.eval()
    results: List[Dict[str, Any]] = []
    feature_key = variant_feature_key(variant)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_images is not None and batch_index >= int(max_images):
                break
            images = batch["images"].to(device)
            depths = batch["depths"].to(device)
            outputs = model(images, query_depth=depths, reference_cache=reference_cache)
            graph_batch = model.build_graph_batch(
                outputs=outputs,
                depth_map=depths,
                instance_map=None,
                reference_cache=reference_cache,
                variant=feature_key,
            )
            if variant_uses_graph(variant):
                edge_logits = model.forward_graph(graph_batch)
                edge_scores = torch.sigmoid(edge_logits)
            else:
                edge_scores = heuristic_edge_scores(graph_batch.edge_features)
            merged = merge_instances_from_edge_scores(
                fragments=graph_batch.fragments,
                edge_index=graph_batch.edge_index,
                edge_scores=edge_scores,
                threshold=edge_threshold,
            )
            masks = _fragment_masks_from_merged(merged, min_area=min_area)
            results.extend(_masks_to_results(int(batch["image_ids"][0]), masks))

    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    return _evaluate_json(ann_file, results_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANT_FLAGS), default="G5")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    reference_bank = load_reference_bank(args.reference_root, image_size=args.image_size)

    train_ds = ECCGraphDataset(args.dataset_root, "train", args.image_size, True)
    val_ds = ECCGraphDataset(args.dataset_root, "val", args.image_size, False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_graph_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_graph_batch,
    )

    model = ReferenceUNetGNN(base_channels=16).to(device)
    reference_cache = cache_to_device(model.build_reference_cache(reference_bank, device), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    ann_file = Path(args.dataset_root) / "annotations" / "instances_val.json"
    feature_key = variant_feature_key(args.variant)

    (output_dir / "params_trainable.txt").write_text(
        str(sum(int(p.numel()) for p in model.parameters() if p.requires_grad)) + "\n",
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()

    best_ap = -1.0
    best_ckpt = output_dir / "model_best.pth"
    start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device)
            depths = batch["depths"].to(device)
            fg_target = batch["fg_target"].to(device)
            boundary_target = batch["boundary_target"].to(device)
            affinity_target = batch["affinity_target"].to(device)
            instance_maps = batch["instance_maps"].to(device)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(images, query_depth=depths, reference_cache=reference_cache)
                loss_fg = F.binary_cross_entropy_with_logits(outputs["fg_logits"], fg_target)
                loss_boundary = F.binary_cross_entropy_with_logits(outputs["boundary_logits"], boundary_target)
                loss_affinity = F.binary_cross_entropy_with_logits(outputs["affinity_logits"], affinity_target)
                loss = loss_fg + loss_boundary + 0.5 * loss_affinity

                if variant_uses_graph(args.variant):
                    graph_losses = []
                    for batch_idx in range(images.shape[0]):
                        graph_batch = model.build_graph_batch(
                            outputs={key: value[batch_idx: batch_idx + 1] for key, value in outputs.items()},
                            depth_map=depths[batch_idx: batch_idx + 1],
                            instance_map=instance_maps[batch_idx],
                            reference_cache=reference_cache,
                            variant=feature_key,
                        )
                        if graph_batch.edge_targets is None or graph_batch.edge_targets.numel() == 0:
                            continue
                        edge_logits = model.forward_graph(graph_batch)
                        graph_losses.append(F.binary_cross_entropy_with_logits(edge_logits, graph_batch.edge_targets))
                    if graph_losses:
                        loss = loss + 0.5 * torch.stack(graph_losses).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 20 == 0:
                print(
                    f"[reference-unet-gnn] epoch={epoch} step={step} loss={float(loss.detach().cpu()):.4f}",
                    flush=True,
                )
            if int(args.max_train_steps) > 0 and step >= int(args.max_train_steps):
                break

        epoch_results = output_dir / f"epoch_{epoch:04d}_results.json"
        metrics = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            reference_cache=reference_cache,
            variant=args.variant,
            ann_file=ann_file,
            results_json=epoch_results,
            min_area=args.min_area,
            edge_threshold=args.edge_threshold,
            max_images=int(args.max_val_images) if int(args.max_val_images) > 0 else None,
        )
        metrics["iteration"] = epoch
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        segm_ap = float(metrics.get("segm/AP", 0.0))
        if segm_ap >= best_ap:
            best_ap = segm_ap
            torch.save(model.state_dict(), best_ckpt)
        print(f"[reference-unet-gnn] epoch={epoch} best_ap={best_ap:.4f}", flush=True)

    final_ckpt = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    final_results = output_dir / "coco_instances_results.json"
    final_metrics = run_eval(
        model=model,
        loader=val_loader,
        device=device,
        reference_cache=reference_cache,
        variant=args.variant,
        ann_file=ann_file,
        results_json=final_results,
        min_area=args.min_area,
        edge_threshold=args.edge_threshold,
        max_images=int(args.max_val_images) if int(args.max_val_images) > 0 else None,
    )
    final_metrics["iteration"] = int(args.epochs)
    (output_dir / "metrics.cocoeval.json").write_text(json.dumps(final_metrics, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "last_checkpoint").write_text(final_ckpt.name + "\n", encoding="utf-8")
    (output_dir / "wall_time_sec.txt").write_text(str(int(time.time() - start)) + "\n", encoding="utf-8")
    print(f"[reference-unet-gnn] final_best_ap={best_ap:.4f}", flush=True)


if __name__ == "__main__":
    main()
