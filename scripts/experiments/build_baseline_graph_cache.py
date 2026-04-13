#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.common.dataset import BaselineInstanceDataset  # noqa: E402
from baseline.common.instance_graph_cache import build_graph_cache_sample_from_masks  # noqa: E402
from baseline.mask2former.adapter import (  # noqa: E402
    build_mask2former_model,
    build_mask2former_processor,
    outputs_to_instance_masks as mask2former_outputs_to_instance_masks,
)
from baseline.mask_rcnn.adapter import outputs_to_instance_masks as maskrcnn_outputs_to_instance_masks  # noqa: E402
from baseline.mask_rcnn.adapter import sample_to_mask_rcnn_image  # noqa: E402
from baseline.mask_rcnn.train import _build_mask_rcnn_model  # noqa: E402
from gisec.datasets.prototype_bank import extract_query_part_key  # noqa: E402
from gisec.engine.runtime import build_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _available_part_keys(reference_root: str | None) -> list[str]:
    if reference_root is None:
        return []
    root = Path(reference_root).resolve()
    if not root.exists():
        return []
    return sorted([path.name for path in root.iterdir() if path.is_dir()], key=lambda item: (-len(item), item))


def _resolve_part_key(file_name: str, available_parts: list[str]) -> str | None:
    if not available_parts:
        return None
    try:
        return extract_query_part_key(str(file_name), available_parts)
    except KeyError:
        return None


def _aggregate_rows(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        return {
            "num_samples": 0,
            "avg_fragments": 0.0,
            "avg_edges": 0.0,
            "avg_fragment_gt_ratio": 0.0,
            "fragment_purity_mean": 0.0,
            "median_largest_fragment_ratio": 0.0,
            "same_instance_pairs_total": 0,
            "same_instance_pairs_covered": 0,
            "same_instance_recall": 0.0,
            "positive_edge_ratio": 0.0,
        }
    return {
        "num_samples": int(len(rows)),
        "avg_fragments": float(sum(int(row["num_fragments"]) for row in rows)) / float(len(rows)),
        "avg_edges": float(sum(int(row["num_edges"]) for row in rows)) / float(len(rows)),
        "avg_fragment_gt_ratio": float(sum(float(row["fragment_gt_count_ratio"]) for row in rows)) / float(len(rows)),
        "fragment_purity_mean": float(sum(float(row["fragment_purity_mean"]) for row in rows)) / float(len(rows)),
        "median_largest_fragment_ratio": float(median(float(row["largest_fragment_ratio"]) for row in rows)),
        "same_instance_pairs_total": int(sum(int(row["same_instance_pairs_total"]) for row in rows)),
        "same_instance_pairs_covered": int(sum(int(row["same_instance_pairs_covered"]) for row in rows)),
        "same_instance_recall": float(sum(float(row["same_instance_recall"]) for row in rows)) / float(len(rows)),
        "positive_edge_ratio": float(sum(float(row["positive_edge_ratio"]) for row in rows)) / float(len(rows)),
    }


def _infer_maskrcnn(
    *,
    model: torch.nn.Module,
    sample: dict,
    device: torch.device,
    input_mode: str,
    score_threshold: float,
) -> tuple[torch.Tensor, list, list[float]]:
    image = sample_to_mask_rcnn_image(sample, input_mode=input_mode).to(device)
    original_image_sizes = [tuple(int(v) for v in image.shape[-2:])]
    images, _ = model.transform([image], None)
    features = model.backbone(images.tensors)
    proposals, _ = model.rpn(images, features, None)
    detections, _ = model.roi_heads(features, proposals, images.image_sizes, None)
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
    masks, scores = maskrcnn_outputs_to_instance_masks(detections[0], score_threshold=float(score_threshold))
    feature_map = features["0"]
    return feature_map, masks, scores


def _infer_mask2former(
    *,
    model: torch.nn.Module,
    processor,
    sample: dict,
    device: torch.device,
    score_threshold: float,
    mask_threshold: float,
) -> tuple[torch.Tensor, list, list[float]]:
    image = sample["image"].float()
    encoded = processor(images=[image], return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)
    pixel_mask = encoded["pixel_mask"].to(device)
    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, output_hidden_states=True)
    masks, scores = mask2former_outputs_to_instance_masks(
        outputs,
        processor=processor,
        target_size=(int(image.shape[-2]), int(image.shape[-1])),
        score_threshold=float(score_threshold),
        mask_threshold=float(mask_threshold),
    )
    feature_map = outputs.pixel_decoder_hidden_states[-1]
    target_shape = (max(int(image.shape[-2]) // 4, 1), max(int(image.shape[-1]) // 4, 1))
    if tuple(int(v) for v in feature_map.shape[-2:]) != target_shape:
        feature_map = F.interpolate(feature_map, size=target_shape, mode="bilinear", align_corners=False)
    return feature_map, masks, scores


def main() -> None:
    args = parse_args()
    payload = _load_yaml(Path(args.config).resolve())
    common = dict(payload.get("common", {}))
    train_cfg = dict(payload.get("train", {}))
    model_cfg = dict(payload.get("model", {}))
    device = build_device(str(args.device or common.get("device", "cpu")))
    model_family = str(model_cfg.get("model_family") or ("mask2former" if "mask2former" in Path(args.config).stem else "mask_rcnn"))
    input_mode = str(model_cfg.get("input_mode", "rgb"))
    image_size = int(common.get("image_size", 1024))
    dataset_root = Path(args.dataset_root).resolve()
    cache_dir = Path(args.output_root).resolve() / str(args.split)
    cache_dir.mkdir(parents=True, exist_ok=True)
    available_parts = _available_part_keys(args.reference_root)

    if model_family == "mask_rcnn":
        model = _build_mask_rcnn_model(
            backbone_name=str(model_cfg.get("backbone_name", "resnet50_fpn")),
            pretrained_backbone=False,
            input_channels=4 if input_mode == "rgbd" else 3,
        ).to(device)
        state_dict = torch.load(Path(args.checkpoint).resolve(), map_location="cpu")
        model.load_state_dict(state_dict)
        infer = lambda sample: _infer_maskrcnn(
            model=model,
            sample=sample,
            device=device,
            input_mode=input_mode,
            score_threshold=float(model_cfg.get("score_threshold", 0.05)),
        )
    elif model_family == "mask2former":
        checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model = build_mask2former_model(
            image_size=image_size,
            pretrained_model_name=model_cfg.get("pretrained_model_name"),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            feature_size=int(model_cfg.get("feature_size", 64)),
            mask_feature_size=int(model_cfg.get("mask_feature_size", 64)),
            encoder_layers=int(model_cfg.get("encoder_layers", 2)),
            decoder_layers=int(model_cfg.get("decoder_layers", 2)),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
            num_queries=int(model_cfg.get("num_queries", 16)),
            train_num_points=int(model_cfg.get("train_num_points", 512)),
        ).to(device)
        model.load_state_dict(state_dict)
        processor = build_mask2former_processor()
        infer = lambda sample: _infer_mask2former(
            model=model,
            processor=processor,
            sample=sample,
            device=device,
            score_threshold=float(model_cfg.get("score_threshold", 0.5)),
            mask_threshold=float(model_cfg.get("mask_threshold", 0.5)),
        )
    else:
        raise ValueError(f"Unsupported model_family for graph cache export: {model_family}")

    model.eval()
    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split=str(args.split),
        image_size=image_size,
        include_depth=str(input_mode) != "rgb",
        include_annotations=False,
        include_instance_map=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(common.get("num_workers", 0)), collate_fn=lambda batch: batch[0])
    rows: list[dict[str, float | int]] = []
    start = time.perf_counter()
    with torch.no_grad():
        for sample_index, sample in enumerate(loader):
            if int(args.max_images) > 0 and sample_index >= int(args.max_images):
                break
            feature_map, masks, scores = infer(sample)
            part_key = _resolve_part_key(str(sample["file_name"]), available_parts)
            payload = build_graph_cache_sample_from_masks(
                image_id=int(sample["image_id"]),
                file_name=str(sample["file_name"]),
                feature_map=feature_map,
                masks=masks,
                scores=scores,
                depth_map=sample.get("depth"),
                instance_map=sample.get("instance_map"),
                part_key=part_key,
                variant="legacy_heuristic_graph_merge_baseline",
                boundary_threshold=float(model_cfg.get("boundary_threshold", 0.5)),
                purity_threshold=float(model_cfg.get("graph_purity_threshold", 0.8)),
                bridge_max_gap=float(model_cfg.get("graph_bridge_max_gap", 4.0)),
            )
            gt_count = 0
            if sample.get("instance_map") is not None:
                gt_values = [int(value) for value in torch.unique(sample["instance_map"]).tolist() if int(value) > 0]
                gt_count = len(gt_values)
            payload["summary"].update(
                {
                    "gt_count": int(gt_count),
                    "fragment_gt_count_ratio": 0.0 if gt_count <= 0 else float(payload["summary"]["num_fragments"]) / float(gt_count),
                }
            )
            torch.save(payload, cache_dir / f"{int(sample['image_id']):06d}.pt")
            rows.append(dict(payload["summary"]))
    manifest = {
        "dataset_root": str(dataset_root),
        "output_root": str(Path(args.output_root).resolve()),
        "split": str(args.split),
        "image_size": int(image_size),
        "model_family": model_family,
        "input_mode": input_mode,
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "reference_root": None if args.reference_root is None else str(Path(args.reference_root).resolve()),
        "elapsed_sec": float(time.perf_counter() - start),
        **_aggregate_rows(rows),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (cache_dir / "rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
