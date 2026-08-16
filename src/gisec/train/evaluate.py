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
from gisec.engine import build_benchmark_payload, build_device, write_json
from gisec.eval.boundary_metrics import compute_boundary_band_iou
from gisec.eval.coco_eval import evaluate_json
from gisec.eval.coco_export import masks_to_coco_results
from gisec.eval.export import build_run_summary_payload, gisec_benchmark_payload
from gisec.eval.split_merge import compute_split_merge_counts
from gisec.models.gisec_model import GISECModel, prepare_gisec_input_batch
from gisec.train.data import build_loader, build_reference_source
from gisec.train.decode import apply_local_rescue, query_instances_from_outputs
from gisec.train.model_builder import (
    build_gisec_model,
    build_pixel_mask,
    extract_state_dict,
    load_module_state_dict,
    resolve_checkpoint_path,
    run_backbone,
    validate_checkpoint_model_args,
    validate_runtime_checkpoint_variant,
)
from gisec.train.args import model_payload


def evaluate_gisec(
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
    graph_merge_threshold: float,
    crop_size: int,
    crop_pad: int,
    boundary_band_width: int,
    max_images: int,
    save_raw: bool,
    depth_mode: str,
    component_class_index: int,
    save_score_threshold: float | None = None,
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
    uses_baseline_decode = not get_gisec_variant_spec(
        variant_name).use_local_refine
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
            pixel_values = prepare_gisec_input_batch(
                images=images, depths=depths, depth_mode=depth_mode)
            pixel_mask = build_pixel_mask(pixel_values)
            # Latency covers the whole per-image pipeline: backbone forward,
            # candidate decode, local/reference/graph rescue, and mask
            # binarization. COCO export and metric computation stay outside
            # the timed region.
            start = time.perf_counter()
            outputs = run_backbone(
                model=model, pixel_values=pixel_values, pixel_mask=pixel_mask)
            batch_decoded: list[dict[str, Any]] = []
            for sample_offset, sample in enumerate(samples):
                image_shape = (
                    int(sample["image"].shape[-2]), int(sample["image"].shape[-1]))
                if uses_baseline_decode:
                    decoded_masks, decoded_scores = outputs_to_instance_masks(
                        outputs,
                        processor=processor,
                        target_size=image_shape,
                        score_threshold=float(score_threshold),
                        mask_threshold=float(mask_threshold),
                    )
                    if not decoded_masks and device.type == "cuda":
                        # An empty candidate set returns before any mask is
                        # copied to the host, so the backbone's async CUDA
                        # tail would still be running when the latency
                        # clock stops; a non-empty decode synchronizes
                        # implicitly through its mask transfer. Give the
                        # empty set the same treatment.
                        torch.cuda.synchronize(device)
                    predictions = [
                        {
                            "query_index": int(index),
                            "score": float(score),
                            "category_id": int(component_class_index),
                            "binary_mask": torch.from_numpy(mask.astype(np.float32)),
                            "mask_probs": torch.from_numpy(mask.astype(np.float32)),
                        }
                        for index, (mask, score) in enumerate(
                            zip(decoded_masks, decoded_scores))
                    ]
                    refine_count = 0
                    graph_count = 0
                else:
                    predictions = query_instances_from_outputs(
                        class_logits=outputs.class_queries_logits[sample_offset],
                        mask_logits=outputs.masks_queries_logits[sample_offset],
                        image_shape=image_shape,
                        score_threshold=float(score_threshold),
                        mask_threshold=float(mask_threshold),
                        component_class_index=int(component_class_index),
                    )
                    predictions, refine_count, graph_count = apply_local_rescue(
                        model=model,
                        variant_name=variant_name,
                        sample=sample,
                        full_input=pixel_values[sample_offset],
                        feature_map=outputs.pixel_decoder_last_hidden_state[sample_offset],
                        predictions=predictions,
                        crop_size=int(crop_size),
                        crop_pad=int(crop_pad),
                        mask_threshold=float(mask_threshold),
                        graph_merge_threshold=float(graph_merge_threshold),
                        boundary_band_width=int(boundary_band_width),
                        reference_source=reference_source,
                    )
                # The probability map is the single source of truth: derive
                # the binary mask by thresholding it here instead of trusting
                # a pre-binarized field whose bilinear paste may have left
                # fractional edge values for a uint8 cast to truncate.
                pred_masks = [
                    (row["mask_probs"] >= float(mask_threshold)).detach().cpu(
                    ).numpy().astype(np.uint8)
                    for row in predictions
                ]
                pred_scores = [float(row["score"]) for row in predictions]
                batch_decoded.append(
                    {
                        "sample": sample,
                        "image_shape": image_shape,
                        "predictions": predictions,
                        "pred_masks": pred_masks,
                        "pred_scores": pred_scores,
                        "refine_count": int(refine_count),
                        "graph_count": int(graph_count),
                    }
                )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            for decoded in batch_decoded:
                sample = decoded["sample"]
                predictions = decoded["predictions"]
                pred_masks = decoded["pred_masks"]
                pred_scores = decoded["pred_scores"]
                refinement_invocations += decoded["refine_count"]
                graph_invocations += decoded["graph_count"]
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
                gt_masks = [] if sample.get("masks") is None else [
                    mask.cpu().numpy().astype(np.uint8) for mask in sample["masks"]]
                failure = compute_split_merge_counts(
                    gt_masks=gt_masks, pred_masks=pred_masks)
                split_total += int(failure["split_gt_count"])
                merge_total += int(failure["merge_pred_count"])
                boundary_rows.append(
                    compute_boundary_band_iou(
                        pred_masks,
                        gt_masks,
                        image_shape=decoded["image_shape"],
                    )
                )
    if save_score_threshold is not None:
        # Infer saves only the high-confidence subset (--score-threshold);
        # the metrics below stay on the full standard-protocol candidate
        # set so they remain comparable with `gisec eval`.
        saved_results = [
            row for row in results
            if float(row["score"]) >= float(save_score_threshold)]
        saved_raw_rows = [
            row for row in raw_rows
            if float(row["score"]) >= float(save_score_threshold)]
    else:
        saved_results = results
        saved_raw_rows = raw_rows
    results_json = output_dir / "coco_instances_results.json"
    results_json.write_text(json.dumps(
        saved_results, ensure_ascii=False) + "\n", encoding="utf-8")
    if save_raw:
        write_json(output_dir / "coco_instances_results.raw.json",
                   {"rows": saved_raw_rows})
    metrics = evaluate_json(ann_file, results)
    metrics["boundary_band_iou"] = float(np.mean(boundary_rows)
                                         ) if boundary_rows else 0.0
    metrics["split_gt_count"] = int(split_total)
    metrics["merge_pred_count"] = int(merge_total)
    metrics["refinement_invocation_rate"] = 0.0 if total_predictions == 0 else float(
        refinement_invocations) / float(total_predictions)
    metrics["local_graph_invocation_rate"] = 0.0 if total_predictions == 0 else float(
        graph_invocations) / float(total_predictions)
    speed = build_benchmark_payload(
        latencies_ms, device, scope="full_pipeline")
    write_json(output_dir / "metrics.cocoeval.json", metrics)
    write_json(output_dir / "inference_speed.json", speed)
    return metrics, speed


def _run_checkpoint_inference(args: argparse.Namespace, *, save_raw: bool) -> None:
    if bool(args.dry_run):
        print(json.dumps(model_payload(args), ensure_ascii=False))
        return
    # Metrics always run on the standard COCO candidate set (score >=
    # eval_score_threshold, 0.05 by default, maxDets up to 100), so infer's
    # metrics land in the same protocol as eval's. --score-threshold only
    # filters the prediction artifacts infer saves to disk.
    score_threshold = float(
        getattr(args, "eval_score_threshold", args.score_threshold))
    save_score_threshold = float(args.score_threshold) if save_raw else None
    variant_spec = get_gisec_variant_spec(args.variant)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir_arg = getattr(args, "checkpoint_dir", "")
    checkpoint_dir = output_dir if checkpoint_dir_arg in (
        "", None) else Path(str(checkpoint_dir_arg)).resolve()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_dir, str(args.checkpoint))
    if checkpoint_path.parent.resolve() == output_dir.resolve():
        raise ValueError(
            "eval/infer requires --checkpoint-dir to differ from --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_gisec_model(args).to(device)
    checkpoint_payload = torch.load(
        str(checkpoint_path), map_location=device, weights_only=True)
    validate_checkpoint_model_args(
        payload=checkpoint_payload,
        args=args,
        context="eval" if not save_raw else "infer",
    )
    validate_runtime_checkpoint_variant(
        requested_variant=variant_spec.name,
        run_variant=getattr(args, "_run_metadata_variant", None),
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=str(checkpoint_path),
        context="eval" if not save_raw else "infer",
    )
    state_dict = extract_state_dict(checkpoint_payload, prefix_backbone=True)
    load_module_state_dict(
        model,
        state_dict,
        allow_partial=bool(
            getattr(args, "allow_partial_checkpoint_load", False)),
        context=f"checkpoint {checkpoint_path}",
    )
    reference_source = build_reference_source(args)
    loader = build_loader(
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
    ann_file = Path(args.dataset_root).resolve() / \
        "annotations" / f"instances_{args.split}.json"
    metrics, speed = evaluate_gisec(
        model=model,
        loader=loader,
        device=device,
        variant_name=variant_spec.name,
        reference_source=reference_source,
        ann_file=ann_file,
        output_dir=output_dir,
        score_threshold=score_threshold,
        mask_threshold=float(args.mask_threshold),
        graph_merge_threshold=float(args.graph_merge_threshold),
        crop_size=int(args.crop_size),
        crop_pad=int(args.crop_pad),
        boundary_band_width=int(args.boundary_band_width),
        max_images=int(args.max_images),
        save_raw=save_raw,
        depth_mode=str(args.depth_mode),
        component_class_index=component_class_index,
        save_score_threshold=save_score_threshold,
    )
    decode_config = {
        "eval_score_threshold": score_threshold,
        "mask_threshold": float(args.mask_threshold),
        "graph_merge_threshold": float(args.graph_merge_threshold),
    }
    if save_raw:
        decode_config["save_score_threshold"] = float(args.score_threshold)
    summary = build_run_summary_payload(
        model="mask2former",
        variant=variant_spec.name,
        modality=str(args.depth_mode),
        artifact_root=output_dir,
        metrics=metrics,
        inference_speed=speed,
        checkpoint=checkpoint_path,
        dataset_root=str(Path(args.dataset_root).resolve()),
        benchmark=gisec_benchmark_payload(
            variant_spec.name, str(args.depth_mode), int(args.image_size)),
        decode_config=decode_config,
    )
    write_json(output_dir / "run_summary.json", summary)


def eval_gisec(args: argparse.Namespace) -> None:
    _run_checkpoint_inference(args, save_raw=False)


def infer_gisec(args: argparse.Namespace) -> None:
    _run_checkpoint_inference(args, save_raw=True)
