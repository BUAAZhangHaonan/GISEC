from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from affinigraph.config.variants import VariantSpec, get_variant_spec, variant_names
from affinigraph.datasets.reference_bank import load_reference_bank
from affinigraph.graph_refiner import GraphRefiner
from affinigraph.models.reference_cache import cache_to_device
from affinigraph.models.reference_unet_gnn import ReferenceUNetGNN


@dataclass
class RunSummary:
    variant: str
    contract_mode: str
    checkpoint: str | None
    results_json: str
    metrics: Dict[str, Any]
    inference_speed: Dict[str, Any]
    params_trainable: int | None = None
    wall_time_sec: int | None = None


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    try:
        from pycocotools import mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"size": list(rle["size"]), "counts": counts}
    except ImportError:  # pragma: no cover - exercised implicitly in base env
        contours, _ = __import__("cv2").findContours(mask.astype(np.uint8), __import__("cv2").RETR_EXTERNAL, __import__("cv2").CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            polygons.append(contour.reshape(-1, 2).astype(float).flatten().tolist())
        return polygons or [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]


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
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - exercised implicitly in dedicated test
        raise RuntimeError(
            "COCO evaluation requires pycocotools; install it in the active affinigraph environment."
        ) from exc
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


def _build_benchmark_payload(latencies_ms: list[float], device: torch.device) -> Dict[str, Any]:
    if not latencies_ms:
        return {
            "status": "empty",
            "timed_images": 0,
            "latency_ms_mean": None,
            "latency_ms_p50": None,
            "latency_ms_p90": None,
            "throughput_fps": None,
            "inference_peak_memory_mb": None,
        }
    lat = np.asarray(latencies_ms, dtype=np.float64)
    total_sec = float(lat.sum() / 1000.0)
    peak_memory_mb = None
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return {
        "status": "ok",
        "timed_images": int(lat.size),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p90": float(np.percentile(lat, 90)),
        "throughput_fps": float(lat.size / total_sec) if total_sec > 0 else None,
        "inference_peak_memory_mb": peak_memory_mb,
    }


def build_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def build_loader(
    *,
    dataset_root: str,
    split: str,
    image_size: int,
    train: bool,
    batch_size: int,
    num_workers: int,
    use_cuda: bool,
) -> DataLoader:
    from affinigraph.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch

    dataset = ECCGraphDataset(dataset_root, split, image_size, train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_graph_batch,
    )


def build_model(device: torch.device, checkpoint: str | Path | None = None) -> ReferenceUNetGNN:
    model = ReferenceUNetGNN(base_channels=16).to(device)
    if checkpoint is not None:
        state_dict = torch.load(str(checkpoint), map_location=device)
        model.load_state_dict(state_dict)
    return model


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def generate_results(
    *,
    model: ReferenceUNetGNN,
    loader: DataLoader,
    device: torch.device,
    reference_cache,
    variant: str | VariantSpec,
    min_area: int,
    edge_threshold: float,
    max_images: int | None = None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    variant_spec = get_variant_spec(variant)
    refiner = GraphRefiner(model)
    results: list[Dict[str, Any]] = []
    latencies_ms: list[float] = []
    model.eval()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_images is not None and batch_index >= int(max_images):
                break
            images = batch["images"].to(device)
            depths = batch["depths"].to(device)
            _sync_cuda(device)
            start = time.perf_counter()
            outputs = model(images, query_depth=depths, reference_cache=reference_cache)
            graph_batch = refiner.build_graph_batch(
                outputs=outputs,
                depth_map=depths,
                instance_map=None,
                reference_cache=reference_cache,
                variant=variant_spec,
            )
            edge_logits = refiner.score_edges(graph_batch, variant_spec)
            merged = refiner.merge(
                graph_batch=graph_batch,
                edge_logits=edge_logits,
                threshold=edge_threshold,
            ).cpu().numpy()
            _sync_cuda(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            masks = _fragment_masks_from_merged(merged, min_area=min_area)
            results.extend(_masks_to_results(int(batch["image_ids"][0]), masks))
    return results, _build_benchmark_payload(latencies_ms, device)


def evaluate_and_export(
    *,
    model: ReferenceUNetGNN,
    loader: DataLoader,
    device: torch.device,
    reference_cache,
    variant: str | VariantSpec,
    ann_file: Path | None,
    results_json: Path,
    min_area: int,
    edge_threshold: float,
    max_images: int | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    results, inference_speed = generate_results(
        model=model,
        loader=loader,
        device=device,
        reference_cache=reference_cache,
        variant=variant,
        min_area=min_area,
        edge_threshold=edge_threshold,
        max_images=max_images,
    )
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    if ann_file is None or not ann_file.exists():
        return {"iteration": -1}, inference_speed
    return _evaluate_json(ann_file, results_json), inference_speed


def prepare_reference_cache(
    *,
    model: ReferenceUNetGNN,
    device: torch.device,
    reference_root: str,
    image_size: int,
    contract_mode: str,
):
    bank = load_reference_bank(reference_root, image_size=image_size, contract_mode=contract_mode)
    return cache_to_device(model.build_reference_cache(bank, device), device), bank


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=list(variant_names()), default="G5")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--contract-mode", choices=["compat", "strict"], default="compat")
    return parser


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[_common_parser()])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    return parser.parse_args(argv)


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[_common_parser()])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--results-json", type=str, default="")
    return parser.parse_args(argv)


def parse_infer_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[_common_parser()])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--results-json", type=str, default="")
    return parser.parse_args(argv)


def train_main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(args.device)
    use_cuda = device.type == "cuda"
    variant_spec = get_variant_spec(args.variant)

    train_loader = build_loader(
        dataset_root=args.dataset_root,
        split="train",
        image_size=args.image_size,
        train=True,
        batch_size=args.batch,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
    )
    val_loader = build_loader(
        dataset_root=args.dataset_root,
        split="val",
        image_size=args.image_size,
        train=False,
        batch_size=1,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
    )

    model = build_model(device)
    refiner = GraphRefiner(model)
    reference_cache, bank = prepare_reference_cache(
        model=model,
        device=device,
        reference_root=args.reference_root,
        image_size=args.image_size,
        contract_mode=args.contract_mode,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    ann_file = Path(args.dataset_root) / "annotations" / "instances_val.json"
    params_trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    (output_dir / "params_trainable.txt").write_text(str(params_trainable) + "\n", encoding="utf-8")

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

                if variant_spec.use_learned_edge_scorer:
                    graph_losses = []
                    for batch_idx in range(images.shape[0]):
                        graph_batch = refiner.build_graph_batch(
                            outputs={key: value[batch_idx : batch_idx + 1] for key, value in outputs.items()},
                            depth_map=depths[batch_idx : batch_idx + 1],
                            instance_map=instance_maps[batch_idx],
                            reference_cache=reference_cache,
                            variant=variant_spec,
                        )
                        if graph_batch.edge_targets is None or graph_batch.edge_targets.numel() == 0:
                            continue
                        edge_logits = refiner.score_edges(graph_batch, variant_spec)
                        graph_losses.append(F.binary_cross_entropy_with_logits(edge_logits, graph_batch.edge_targets))
                    if graph_losses:
                        loss = loss + 0.5 * torch.stack(graph_losses).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 20 == 0:
                print(f"[affinigraph] epoch={epoch} step={step} loss={float(loss.detach().cpu()):.4f}", flush=True)
            if int(args.max_train_steps) > 0 and step >= int(args.max_train_steps):
                break

        epoch_results = output_dir / f"epoch_{epoch:04d}_results.json"
        metrics, _benchmark = evaluate_and_export(
            model=model,
            loader=val_loader,
            device=device,
            reference_cache=reference_cache,
            variant=variant_spec,
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
        print(f"[affinigraph] epoch={epoch} best_ap={best_ap:.4f}", flush=True)

    final_ckpt = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    final_results = output_dir / "coco_instances_results.json"
    final_metrics, inference_speed = evaluate_and_export(
        model=model,
        loader=val_loader,
        device=device,
        reference_cache=reference_cache,
        variant=variant_spec,
        ann_file=ann_file,
        results_json=final_results,
        min_area=args.min_area,
        edge_threshold=args.edge_threshold,
        max_images=int(args.max_val_images) if int(args.max_val_images) > 0 else None,
    )
    final_metrics["iteration"] = int(args.epochs)
    wall_time_sec = int(time.time() - start)
    write_json(output_dir / "metrics.cocoeval.json", final_metrics)
    write_json(output_dir / "inference_speed.json", inference_speed)
    write_json(
        output_dir / "run_summary.json",
        asdict(
            RunSummary(
                variant=variant_spec.name,
                contract_mode=args.contract_mode,
                checkpoint=str(final_ckpt),
                results_json=str(final_results),
                metrics=final_metrics,
                inference_speed=inference_speed,
                params_trainable=params_trainable,
                wall_time_sec=wall_time_sec,
            )
        ),
    )
    write_json(output_dir / "reference_bank_manifest.json", asdict(bank.manifest))
    (output_dir / "last_checkpoint").write_text(final_ckpt.name + "\n", encoding="utf-8")
    (output_dir / "wall_time_sec.txt").write_text(str(wall_time_sec) + "\n", encoding="utf-8")
    print(f"[affinigraph] final_best_ap={best_ap:.4f}", flush=True)


def _resolve_checkpoint(output_dir: Path, checkpoint: str) -> Path:
    if checkpoint:
        return Path(checkpoint).resolve()
    for candidate in [output_dir / "model_best.pth", output_dir / "model_final.pth"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {output_dir}")


def _run_eval_like(args: argparse.Namespace, *, compute_metrics: bool) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(args.device)
    use_cuda = device.type == "cuda"
    variant_spec = get_variant_spec(args.variant)
    loader = build_loader(
        dataset_root=args.dataset_root,
        split=args.split,
        image_size=args.image_size,
        train=False,
        batch_size=1,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
    )
    checkpoint_path = _resolve_checkpoint(output_dir, args.checkpoint)
    model = build_model(device, checkpoint_path)
    reference_cache, bank = prepare_reference_cache(
        model=model,
        device=device,
        reference_root=args.reference_root,
        image_size=args.image_size,
        contract_mode=args.contract_mode,
    )
    results_json = Path(args.results_json).resolve() if args.results_json else output_dir / "coco_instances_results.json"
    ann_file = None
    if compute_metrics:
        ann_candidate = Path(args.dataset_root) / "annotations" / f"instances_{args.split}.json"
        if ann_candidate.exists():
            ann_file = ann_candidate
    metrics, inference_speed = evaluate_and_export(
        model=model,
        loader=loader,
        device=device,
        reference_cache=reference_cache,
        variant=variant_spec,
        ann_file=ann_file,
        results_json=results_json,
        min_area=args.min_area,
        edge_threshold=args.edge_threshold,
        max_images=int(args.max_images) if int(args.max_images) > 0 else None,
    )
    write_json(output_dir / "metrics.cocoeval.json", metrics)
    write_json(output_dir / "inference_speed.json", inference_speed)
    write_json(
        output_dir / "run_summary.json",
        asdict(
            RunSummary(
                variant=variant_spec.name,
                contract_mode=args.contract_mode,
                checkpoint=str(checkpoint_path),
                results_json=str(results_json),
                metrics=metrics,
                inference_speed=inference_speed,
            )
        ),
    )
    write_json(output_dir / "reference_bank_manifest.json", asdict(bank.manifest))


def eval_main(args: argparse.Namespace) -> None:
    _run_eval_like(args, compute_metrics=True)


def infer_main(args: argparse.Namespace) -> None:
    _run_eval_like(args, compute_metrics=args.split != "test")


def main(argv: list[str] | None = None) -> None:
    train_main(parse_train_args(argv))


if __name__ == "__main__":
    main()
