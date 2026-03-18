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

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.datasets.prototype_bank import load_prototype_bank
from gisec.graph_refiner import GraphRefiner
from gisec.models.gisec_model import GISECModel
from gisec.models.prototype_cache import cache_to_device


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
        contours, _ = __import__("cv2").findContours(mask.astype(np.uint8), __import__("cv2").RETR_EXTERNAL, __import__("cv2").CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            polygons.append(contour.reshape(-1, 2).astype(float).flatten().tolist())
        return polygons or [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]


def masks_to_results(image_id: int, masks: List[np.ndarray]) -> List[Dict[str, Any]]:
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
    from gisec.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch

    dataset = ECCGraphDataset(dataset_root, split, image_size, train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


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
) -> tuple[Dict[str, Any], Dict[str, Any]]:
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
            sync_cuda(device)
            start = time.perf_counter()
            outputs = model(images, query_depth=depths, prototype_cache=prototype_cache)
            graph_batch = refiner.build_graph_batch(
                outputs=outputs,
                depth_map=depths,
                instance_map=None,
                prototype_cache=prototype_cache,
                variant=variant_spec,
            )
            edge_logits = refiner.score_edges(graph_batch, variant_spec)
            merged = refiner.merge(
                graph_batch=graph_batch,
                edge_logits=edge_logits,
                threshold=edge_threshold,
            ).cpu().numpy()
            sync_cuda(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            masks = fragment_masks_from_merged(merged, min_area=min_area)
            results.extend(masks_to_results(int(batch["image_ids"][0]), masks))
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
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
    bank = load_prototype_bank(prototype_root, image_size=image_size, contract_mode=contract_mode)
    return cache_to_device(model.build_prototype_cache(bank, device), device), bank


def resolve_checkpoint(output_dir: Path, checkpoint: str) -> Path:
    if checkpoint:
        return Path(checkpoint).resolve()
    for candidate in [output_dir / "model_best.pth", output_dir / "model_final.pth"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {output_dir}")
