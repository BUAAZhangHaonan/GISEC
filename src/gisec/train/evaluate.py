from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from gisec.backbones.mask2former.adapter import build_mask2former_processor, outputs_to_instance_masks
from gisec.config.variants import get_gisec_variant_spec
from gisec.datasets.reference_bank import ReferenceBankSource
from gisec.engine.runtime import build_benchmark_payload, build_device, evaluate_json, write_json
from gisec.eval.boundary_metrics import compute_boundary_iou
from gisec.eval.coco_export import masks_to_coco_results
from gisec.eval.export import build_run_summary_payload
from gisec.metrics import compute_split_merge_counts
from gisec.models.gisec_model import GISECModel, prepare_gisec_input_batch
from gisec.train.data import _build_loader
from gisec.train.decode import _apply_local_rescue, _query_instances_from_outputs, _uses_baseline_decode
from gisec.train.model_builder import (
    _build_gisec_model,
    _build_pixel_mask,
    _extract_state_dict,
    _load_module_state_dict,
    _resolve_checkpoint_path,
    _run_backbone,
    _validate_runtime_checkpoint_variant,
)
from gisec.train.args import _model_payload


def _gisec_benchmark_payload(variant_name: str, depth_mode: str) -> dict[str, Any]:
    refine_mode = "none"
    if variant_name.endswith("_refine"):
        refine_mode = "local_refine"
    elif variant_name.endswith("_refine_ref"):
        refine_mode = "local_refine_ref"
    elif variant_name.endswith("_refine_ref_graph"):
        refine_mode = "local_refine_ref_graph"
    return {
        "model_family": "mask2former",
        "backbone_name": "swin_t",
        "resolution": 1024,
        "input_mode": str(depth_mode),
        "fusion_mode": str(depth_mode),
        "refine_mode": refine_mode,
        "inference_defaults_locked": True,
    }


def _evaluate_gisec(
    *,
    model: GISECModel,
    loader: DataLoader,
    device: torch.device,
    variant_name: str,
    reference_source: ReferenceBankSource | None,
    ann_file: Path,
    output_dir: Path,
    score_threshold: float,
    mask_threshold: float,
    crop_size: int,
    crop_pad: int,
    boundary_band_width: int,
    max_images: int,
    save_raw: bool,
    depth_mode: str,
    component_class_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    processor = build_mask2former_processor()
    results: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    boundary_rows: list[float] = []
    split_total = 0
    merge_total = 0
    refinement_invocations = 0
    graph_invocations = 0
    total_predictions = 0
    non_blocking = bool(device.type == "cuda")
    with torch.no_grad():
        for batch_index, samples in enumerate(loader):
            if int(max_images) > 0 and batch_index >= int(max_images):
                break
            images = torch.stack([sample["image"].float() for sample in samples], dim=0).to(
                device, non_blocking=non_blocking
            )
            depths = None
            if str(depth_mode) != "rgb":
                depths = torch.stack([sample["depth"].float() for sample in samples], dim=0).to(
                    device, non_blocking=non_blocking
                )
            pixel_values = prepare_gisec_input_batch(images=images, depths=depths, depth_mode=depth_mode)
            pixel_mask = _build_pixel_mask(pixel_values)
            start = time.perf_counter()
            outputs = _run_backbone(model=model, pixel_values=pixel_values, pixel_mask=pixel_mask)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            for sample_offset, sample in enumerate(samples):
                image_shape = (int(sample["image"].shape[-2]), int(sample["image"].shape[-1]))
                if _uses_baseline_decode(variant_name):
                    pred_masks, pred_scores = outputs_to_instance_masks(
                        outputs,
                        processor=processor,
                        target_size=image_shape,
                        score_threshold=float(score_threshold),
                        mask_threshold=float(mask_threshold),
                    )
                    predictions = [
                        {
                            "query_index": int(index),
                            "score": float(score),
                            "category_id": int(component_class_index),
                            "binary_mask": torch.from_numpy(mask.astype(np.float32)),
                            "mask_probs": torch.from_numpy(mask.astype(np.float32)),
                        }
                        for index, (mask, score) in enumerate(zip(pred_masks, pred_scores))
                    ]
                    refine_count = 0
                    graph_count = 0
                else:
                    predictions = _query_instances_from_outputs(
                        class_logits=outputs.class_queries_logits[sample_offset],
                        mask_logits=outputs.masks_queries_logits[sample_offset],
                        image_shape=image_shape,
                        score_threshold=float(score_threshold),
                        mask_threshold=float(mask_threshold),
                        component_class_index=int(component_class_index),
                    )
                    predictions, refine_count, graph_count = _apply_local_rescue(
                        model=model,
                        variant_name=variant_name,
                        sample=sample,
                        full_input=pixel_values[sample_offset],
                        feature_map=outputs.pixel_decoder_last_hidden_state[sample_offset],
                        predictions=predictions,
                        crop_size=int(crop_size),
                        crop_pad=int(crop_pad),
                        mask_threshold=float(mask_threshold),
                        boundary_band_width=int(boundary_band_width),
                        reference_source=reference_source,
                    )
                    pred_masks = [row["binary_mask"].detach().cpu().numpy().astype(np.uint8) for row in predictions]
                    pred_scores = [float(row["score"]) for row in predictions]
                refinement_invocations += int(refine_count)
                graph_invocations += int(graph_count)
                total_predictions += len(pred_masks)
                results.extend(
                    masks_to_coco_results(
                        image_id=int(sample["image_id"]),
                        masks=pred_masks,
                        scores=pred_scores,
                        category_id=int(component_class_index),
                    )
                )
                if save_raw:
                    raw_rows.extend(
                        [
                            {
                                "image_id": int(sample["image_id"]),
                                "query_index": int(row["query_index"]),
                                "score": float(row["score"]),
                            }
                            for row in predictions
                        ]
                    )
                gt_masks = [] if sample.get("masks") is None else [mask.cpu().numpy().astype(np.uint8) for mask in sample["masks"]]
                failure = compute_split_merge_counts(gt_masks=gt_masks, pred_masks=pred_masks)
                split_total += int(failure["split_gt_count"])
                merge_total += int(failure["merge_pred_count"])
                boundary_rows.append(
                    compute_boundary_iou(
                        pred_masks,
                        gt_masks,
                        image_shape=image_shape,
                    )
                )
    results_json = output_dir / "coco_instances_results.json"
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    if save_raw:
        write_json(output_dir / "coco_instances_results.raw.json", {"rows": raw_rows})
    metrics = evaluate_json(ann_file, results_json)
    metrics["boundary/IoU"] = float(np.mean(boundary_rows)) if boundary_rows else 0.0
    metrics["split_gt_count"] = int(split_total)
    metrics["merge_pred_count"] = int(merge_total)
    metrics["refinement_invocation_rate"] = 0.0 if total_predictions == 0 else float(refinement_invocations) / float(total_predictions)
    metrics["local_graph_invocation_rate"] = 0.0 if total_predictions == 0 else float(graph_invocations) / float(total_predictions)
    speed = build_benchmark_payload(latencies_ms, device)
    write_json(output_dir / "metrics.cocoeval.json", metrics)
    write_json(output_dir / "inference_speed.json", speed)
    return metrics, speed


def _run_checkpoint_inference(args: argparse.Namespace, *, save_raw: bool) -> None:
    if bool(args.dry_run):
        print(json.dumps(_model_payload(args), ensure_ascii=False))
        return
    variant_spec = get_gisec_variant_spec(args.variant)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir_arg = getattr(args, "checkpoint_dir", "")
    checkpoint_dir = output_dir if checkpoint_dir_arg in ("", None) else Path(str(checkpoint_dir_arg)).resolve()
    checkpoint_path = _resolve_checkpoint_path(checkpoint_dir, str(args.checkpoint))
    if checkpoint_path.parent.resolve() == output_dir.resolve():
        raise ValueError("eval/infer requires --checkpoint-dir to differ from --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_gisec_model(args).to(device)
    checkpoint_payload = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    _validate_runtime_checkpoint_variant(
        requested_variant=variant_spec.name,
        run_variant=getattr(args, "_run_metadata_variant", None),
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=str(checkpoint_path),
        context="eval" if not save_raw else "infer",
    )
    state_dict = _extract_state_dict(checkpoint_payload, prefix_backbone=True)
    _load_module_state_dict(
        model,
        state_dict,
        allow_partial=bool(getattr(args, "allow_partial_checkpoint_load", False)),
        context=f"checkpoint {checkpoint_path}",
    )
    reference_source = None
    if variant_spec.requires_reference_root:
        reference_source = ReferenceBankSource(
            root=Path(str(args.reference_root)).resolve(),
            image_size=int(args.crop_size),
            contract_mode="compat",
            max_views=int(args.reference_max_views),
            view_sampler=str(args.reference_view_sampler),
        )
    loader = _build_loader(
        dataset_root=str(args.dataset_root),
        split=str(args.split),
        image_size=int(args.image_size),
        batch_size=1,
        num_workers=int(args.num_workers),
        include_depth=str(args.depth_mode) != "rgb",
        train=False,
        use_cuda=bool(device.type == "cuda"),
    )
    component_class_index = int(loader.dataset.component_category_id)
    ann_file = Path(args.dataset_root).resolve() / "annotations" / f"instances_{args.split}.json"
    metrics, speed = _evaluate_gisec(
        model=model,
        loader=loader,
        device=device,
        variant_name=variant_spec.name,
        reference_source=reference_source,
        ann_file=ann_file,
        output_dir=output_dir,
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        crop_size=int(args.crop_size),
        crop_pad=int(args.crop_pad),
        boundary_band_width=int(args.boundary_band_width),
        max_images=int(args.max_images),
        save_raw=save_raw,
        depth_mode=str(args.depth_mode),
        component_class_index=component_class_index,
    )
    summary = build_run_summary_payload(
        model="mask2former",
        variant=variant_spec.name,
        modality=str(args.depth_mode),
        artifact_root=output_dir,
        metrics=metrics,
        inference_speed=speed,
        checkpoint=checkpoint_path,
        dataset_root=str(Path(args.dataset_root).resolve()),
        benchmark=_gisec_benchmark_payload(variant_spec.name, str(args.depth_mode)),
        decode_config={
            "score_threshold": float(args.score_threshold),
            "mask_threshold": float(args.mask_threshold),
        },
    )
    write_json(output_dir / "run_summary.json", summary)


def eval_gisec(args: argparse.Namespace) -> None:
    _run_checkpoint_inference(args, save_raw=False)


def infer_gisec(args: argparse.Namespace) -> None:
    _run_checkpoint_inference(args, save_raw=True)

