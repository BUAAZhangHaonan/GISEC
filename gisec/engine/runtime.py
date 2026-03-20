from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.datasets.prototype_bank import load_prototype_bank
from gisec.graph_refiner import GraphRefiner
from gisec.models.gisec_model import GISECModel
from gisec.models.prototype_cache import cache_to_device
from gisec.utils.visualization import render_fragment_merge_preview


@dataclass
class RunContext:
    dataset_root: str
    prototype_root: str
    split: str
    image_size: int
    batch: int
    num_workers: int
    min_area: int
    edge_threshold: float
    contract_mode: str
    device: str
    code_revision: str | None = None


@dataclass
class RunSummary:
    variant: str
    contract_mode: str
    checkpoint: str | None
    results_json: str
    metrics: Dict[str, Any]
    inference_speed: Dict[str, Any]
    dataset_root: str
    prototype_root: str
    split: str
    image_size: int
    batch: int
    num_workers: int
    min_area: int
    edge_threshold: float
    device: str
    code_revision: str | None = None
    params_trainable: int | None = None
    training_peak_memory_mb: float | None = None
    wall_time_sec: int | None = None


def read_git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    try:
        from pycocotools import mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"size": list(rle["size"]), "counts": counts}
    except ImportError:  # pragma: no cover - exercised implicitly in base env
        contours, _ = __import__("cv2").findContours(mask.astype(np.uint8), __import__(
            "cv2").RETR_EXTERNAL, __import__("cv2").CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            polygons.append(
                contour.reshape(-1, 2).astype(float).flatten().tolist())
        return polygons or [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]


def _clamp_unit(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _resolve_score_sequence(values: list[float] | None, *, count: int, default: float) -> list[float]:
    if values is None:
        return [float(default)] * count
    if len(values) != count:
        raise ValueError(f"Expected {count} score values, got {len(values)}")
    return [_clamp_unit(value) for value in values]


def _compose_instance_score(*, fg_score: float, boundary_score: float, merge_score: float) -> float:
    return _clamp_unit(0.5 * fg_score + 0.35 * merge_score + 0.15 * (1.0 - boundary_score))


def masks_to_results(
    image_id: int,
    masks: List[np.ndarray],
    *,
    fg_scores: list[float] | None = None,
    boundary_scores: list[float] | None = None,
    merge_scores: list[float] | None = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    fg_values = _resolve_score_sequence(
        fg_scores, count=len(masks), default=0.5)
    boundary_values = _resolve_score_sequence(
        boundary_scores, count=len(masks), default=0.5)
    merge_values = _resolve_score_sequence(
        merge_scores, count=len(masks), default=0.5)
    for index, mask in enumerate(masks):
        if int(mask.sum()) <= 0:
            continue
        ys, xs = np.nonzero(mask > 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        results.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "score": _compose_instance_score(
                    fg_score=fg_values[index],
                    boundary_score=boundary_values[index],
                    merge_score=merge_values[index],
                ),
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "segmentation": encode_binary_mask(mask.astype(np.uint8)),
            }
        )
    return results


def fragment_masks_from_merged(merged: np.ndarray, min_area: int) -> List[np.ndarray]:
    masks = []
    for label in [int(x) for x in np.unique(merged).tolist() if int(x) > 0]:
        mask = (merged == label).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            masks.append(mask)
    return masks


def _component_merge_score(
    *,
    merged_mask: np.ndarray,
    fragments: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    threshold: float,
) -> float:
    source_labels = {int(x) for x in np.unique(
        fragments[merged_mask]).tolist() if int(x) > 0}
    if len(source_labels) <= 1 or edge_index.numel() == 0:
        return 0.5
    label_order = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
    accepted_scores: list[float] = []
    fallback_scores: list[float] = []
    for (src, dst), score in zip(edge_index.t().tolist(), edge_scores.tolist()):
        label_src = label_order[int(src)]
        label_dst = label_order[int(dst)]
        if label_src not in source_labels or label_dst not in source_labels:
            continue
        score_value = _clamp_unit(float(score))
        fallback_scores.append(score_value)
        if score_value >= float(threshold):
            accepted_scores.append(score_value)
    if accepted_scores:
        return float(np.mean(accepted_scores))
    if fallback_scores:
        return float(np.mean(fallback_scores))
    return 0.5


def _build_export_records(
    *,
    merged: np.ndarray,
    fragments: np.ndarray,
    fg_prob: np.ndarray,
    boundary_prob: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    min_area: int,
    threshold: float,
) -> tuple[list[np.ndarray], list[float], list[float], list[float]]:
    masks: list[np.ndarray] = []
    fg_scores: list[float] = []
    boundary_scores: list[float] = []
    merge_scores: list[float] = []
    for label in [int(x) for x in np.unique(merged).tolist() if int(x) > 0]:
        mask = (merged == label).astype(np.uint8)
        if int(mask.sum()) < int(min_area):
            continue
        mask_bool = mask.astype(bool)
        masks.append(mask)
        fg_scores.append(_clamp_unit(
            float(fg_prob[mask_bool].mean()) if mask_bool.any() else 0.0))
        boundary_scores.append(_clamp_unit(
            float(boundary_prob[mask_bool].mean()) if mask_bool.any() else 0.0))
        merge_scores.append(
            _component_merge_score(
                merged_mask=mask_bool,
                fragments=fragments,
                edge_index=edge_index,
                edge_scores=edge_scores,
                threshold=threshold,
            )
        )
    return masks, fg_scores, boundary_scores, merge_scores


def _image_tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    return np.round(image * 255.0).astype(np.uint8)


def _prepare_overlay_dir(overlay_dir: Path) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for png_path in overlay_dir.glob("*.png"):
        png_path.unlink()


def evaluate_json(ann_file: Path, results_json: Path) -> Dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - exercised implicitly in dedicated test
        raise RuntimeError(
            "COCO evaluation requires pycocotools; install it in the active gisec environment."
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


def build_benchmark_payload(latencies_ms: list[float], device: torch.device) -> Dict[str, Any]:
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
        peak_memory_mb = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return {
        "status": "ok",
        "timed_images": int(lat.size),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p90": float(np.percentile(lat, 90)),
        "throughput_fps": float(lat.size / total_sec) if total_sec > 0 else None,
        "inference_peak_memory_mb": peak_memory_mb,
    }


def build_device(device_name: str, local_rank: int | None = None) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if local_rank is not None:
            return torch.device(f"cuda:{int(local_rank)}")
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
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    from gisec.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch

    dataset = ECCGraphDataset(dataset_root, split, image_size, train)
    sampler = None
    shuffle = train
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(train),
            drop_last=False,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_graph_batch,
    )


def build_model(device: torch.device, checkpoint: str | Path | None = None) -> GISECModel:
    model = GISECModel(base_channels=16).to(device)
    if checkpoint is not None:
        state_dict = torch.load(str(checkpoint), map_location=device)
        model.load_state_dict(state_dict)
    return model


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False,
                    indent=2, default=str) + "\n", encoding="utf-8")


def evaluate_and_export(
    *,
    model: GISECModel,
    loader: DataLoader,
    device: torch.device,
    prototype_cache,
    variant: str | VariantSpec,
    ann_file: Path | None,
    results_json: Path,
    min_area: int,
    edge_threshold: float,
    max_images: int | None = None,
    artifact_dir: Path | None = None,
    save_overlays: bool = False,
    overlay_limit: int = 0,
    save_graph_diagnostics: bool = False,
    diagnostics_limit: int = 0,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    variant_spec = get_variant_spec(variant)
    refiner = GraphRefiner(model)
    results: list[Dict[str, Any]] = []
    latencies_ms: list[float] = []
    diagnostics_path = None if artifact_dir is None else artifact_dir / \
        "graph_diagnostics.jsonl"
    overlay_dir = None if artifact_dir is None else artifact_dir / \
        "visualizations" / "overlay"
    overlay_budget = None if int(overlay_limit) <= 0 else int(overlay_limit)
    diagnostics_budget = None if int(
        diagnostics_limit) <= 0 else int(diagnostics_limit)
    overlays_written = 0
    diagnostics_written = 0
    if save_graph_diagnostics and diagnostics_path is not None and diagnostics_path.exists():
        diagnostics_path.unlink()
    if save_overlays and overlay_dir is not None:
        _prepare_overlay_dir(overlay_dir)
    model.eval()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_images is not None and batch_index >= int(max_images):
                break
            images = batch["images"].to(device)
            depths = batch["depths"].to(device)
            sync_cuda(device)
            start = time.perf_counter()
            outputs = model(images, query_depth=depths,
                            prototype_cache=prototype_cache)
            graph_batch = refiner.build_graph_batch(
                outputs=outputs,
                depth_map=depths,
                instance_map=None,
                prototype_cache=prototype_cache,
                variant=variant_spec,
            )
            edge_logits = refiner.score_edges(graph_batch, variant_spec)
            edge_scores = torch.sigmoid(edge_logits.detach()).cpu()
            merged = refiner.merge(
                graph_batch=graph_batch,
                edge_logits=edge_logits,
                threshold=edge_threshold,
                variant=variant_spec,
            ).cpu().numpy()
            sync_cuda(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            fg_prob = torch.sigmoid(outputs["fg_logits"].detach())[
                0, 0].cpu().numpy()
            boundary_prob = torch.sigmoid(outputs["boundary_logits"].detach())[
                0, 0].cpu().numpy()
            masks, fg_scores, boundary_scores, merge_scores = _build_export_records(
                merged=merged,
                fragments=graph_batch.fragments,
                fg_prob=fg_prob,
                boundary_prob=boundary_prob,
                edge_index=graph_batch.edge_index.cpu(),
                edge_scores=edge_scores,
                min_area=min_area,
                threshold=edge_threshold,
            )
            results.extend(
                masks_to_results(
                    int(batch["image_ids"][0]),
                    masks,
                    fg_scores=fg_scores,
                    boundary_scores=boundary_scores,
                    merge_scores=merge_scores,
                )
            )
            if save_graph_diagnostics and diagnostics_path is not None and (
                diagnostics_budget is None or diagnostics_written < diagnostics_budget
            ):
                graph_batch.diagnostics["num_merged"] = len(masks)
                diagnostic_row = {
                    "image_id": int(batch["image_ids"][0]),
                    "file_name": batch["file_names"][0],
                    "variant": variant_spec.name,
                    **graph_batch.diagnostics,
                    "graph_has_edges": int(graph_batch.edge_index.shape[1] > 0),
                    "graph_positive_edge_targets": 0.0
                    if graph_batch.edge_targets is None
                    else float(graph_batch.edge_targets.sum().item()),
                    "edge_score_mean": None if edge_scores.numel() == 0 else float(edge_scores.mean().item()),
                    "instance_score_mean": None if not masks else float(np.mean([item["score"] for item in results[-len(masks):]])),
                }
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(diagnostics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(diagnostic_row,
                                 ensure_ascii=False) + "\n")
                diagnostics_written += 1
            if save_overlays and overlay_dir is not None and (overlay_budget is None or overlays_written < overlay_budget):
                image_rgb = _image_tensor_to_rgb(batch["images"][0])
                stem = Path(batch["file_names"][0]).stem
                overlay_path = overlay_dir / \
                    f"{batch_index:04d}_{int(batch['image_ids'][0]):06d}_{stem}.png"
                render_fragment_merge_preview(
                    image=image_rgb,
                    fragments=graph_batch.fragments,
                    merged=merged,
                    output_path=overlay_path,
                )
                overlays_written += 1
    results_json.write_text(json.dumps(
        results, ensure_ascii=False) + "\n", encoding="utf-8")
    if ann_file is None or not ann_file.exists():
        return {"iteration": -1}, build_benchmark_payload(latencies_ms, device)
    return evaluate_json(ann_file, results_json), build_benchmark_payload(latencies_ms, device)


def prepare_prototype_cache(
    *,
    model: GISECModel,
    device: torch.device,
    prototype_root: str,
    image_size: int,
    contract_mode: str,
):
    bank = load_prototype_bank(
        prototype_root, image_size=image_size, contract_mode=contract_mode)
    return cache_to_device(model.build_prototype_cache(bank, device), device), bank


def resolve_checkpoint(output_dir: Path, checkpoint: str) -> Path:
    if checkpoint:
        return Path(checkpoint).resolve()
    for candidate in [output_dir / "model_best.pth", output_dir / "model_final.pth"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {output_dir}")
