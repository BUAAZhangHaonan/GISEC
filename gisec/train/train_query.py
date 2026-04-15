from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.common.coco_export import masks_to_coco_results
from gisec.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch
from gisec.datasets.prototype_bank import PrototypeBank, PrototypeBankSource
from gisec.engine.runtime import evaluate_json
from gisec.engine.query_factory import build_query_model
from gisec.engine.query_runtime import (
    UQRunSummary,
    build_query_graph_batch,
    merge_query_graph_instances,
    predict_instance_map,
    save_run_summary,
    score_query_graph_edges,
    summarize_instance_matching,
    summarize_mask_calibration,
    summarize_object_pathology,
    summarize_matches,
    write_json,
)
from gisec.train.query_targets import (
    build_core_heatmap_target,
    build_fg_target,
    build_instance_boundary_target,
    build_ownership_target,
)


DEFAULT_GRAPH_LOSS_WEIGHT = 0.5
DEFAULT_GRAPH_WARMUP_STEPS = 0


def _parse_optional_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _build_query_loader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    device_obj: torch.device,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> dict[str, object]:
    use_cuda = str(device_obj.type) == "cuda"
    loader_kwargs: dict[str, object] = {
        "batch_size": max(batch_size, 1),
        "shuffle": False,
        "num_workers": int(num_workers),
        "collate_fn": collate_graph_batch,
        "pin_memory": use_cuda if pin_memory is None else bool(pin_memory),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True if persistent_workers is None else bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = 2 if prefetch_factor is None else int(prefetch_factor)
    return loader_kwargs


def _build_alpha_targets_from_instance_maps(instance_maps: torch.Tensor) -> dict[str, torch.Tensor]:
    fg_targets = []
    boundary_targets = []
    core_targets = []
    ownership_targets = []
    for instance_map in instance_maps:
        instance_map_np = instance_map.cpu().numpy()
        fg_targets.append(torch.from_numpy(build_fg_target(instance_map_np)).float().unsqueeze(0))
        boundary_targets.append(torch.from_numpy(build_instance_boundary_target(instance_map_np)).float().unsqueeze(0))
        core_targets.append(torch.from_numpy(build_core_heatmap_target(instance_map_np)).float().unsqueeze(0))
        ownership_targets.append(torch.from_numpy(build_ownership_target(instance_map_np)).float())
    return {
        "fg": torch.stack(fg_targets, dim=0),
        "boundary": torch.stack(boundary_targets, dim=0),
        "core": torch.stack(core_targets, dim=0),
        "ownership": torch.stack(ownership_targets, dim=0),
    }


def _build_alpha_targets_from_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target_keys = {
        "fg": "fg_target",
        "boundary": "boundary_target",
        "core": "core_target",
        "ownership": "query_ownership_target",
    }
    if all(key in batch for key in target_keys.values()):
        return {name: batch[key] for name, key in target_keys.items()}
    return _build_alpha_targets_from_instance_maps(batch["instance_maps"])


def _dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _balanced_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, *, max_pos_weight: float = 64.0) -> torch.Tensor:
    pos = float(targets.sum().item())
    total = float(targets.numel())
    neg = max(total - pos, 0.0)
    if pos <= 0.0:
        pos_weight = torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
    else:
        pos_weight = torch.tensor(min(max(neg / pos, 1.0), max_pos_weight), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def _focal_heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating_factor = (1.0 - pt).pow(gamma)
    return (alpha_factor * modulating_factor * ce).mean()


def _compute_alpha_losses(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fg_bce = _balanced_bce_with_logits(outputs["fg_logits"], targets["fg"])
    fg_dice = _dice_loss_from_logits(outputs["fg_logits"], targets["fg"])
    loss_fg = fg_bce + fg_dice
    loss_boundary = _balanced_bce_with_logits(outputs["boundary_logits"], targets["boundary"])
    loss_core = _focal_heatmap_loss(outputs["core_heatmap"], targets["core"])
    fg_mask = targets["fg"].expand_as(outputs["ownership_offsets"]) > 0.5
    if fg_mask.any():
        loss_ownership = F.smooth_l1_loss(outputs["ownership_offsets"][fg_mask], targets["ownership"][fg_mask])
    else:
        loss_ownership = outputs["ownership_offsets"].sum() * 0.0
    return {
        "fg": loss_fg,
        "boundary": loss_boundary,
        "core": loss_core,
        "ownership": loss_ownership,
    }


def _reduce_alpha_losses(
    losses: dict[str, torch.Tensor],
    *,
    step_index: int,
    ownership_warmup_steps: int,
    loss_weights: dict[str, float],
) -> torch.Tensor:
    total = losses["fg"] * float(loss_weights["fg"])
    total = total + losses["boundary"] * float(loss_weights["boundary"])
    total = total + losses["core"] * float(loss_weights["core"])
    ownership_weight = 0.0 if int(step_index) < int(ownership_warmup_steps) else float(loss_weights["ownership"])
    total = total + losses["ownership"] * ownership_weight
    return total


def _build_alpha_optimizer(model: torch.nn.Module, *, lr: float, head_lr_multiplier: float) -> torch.optim.Optimizer:
    head_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in ("fg_head", "boundary_head", "core_head", "ownership_head", "graph_head", "edge_scorer")):
            head_params.append(param)
        else:
            base_params.append(param)
    return torch.optim.Adam(
        [
            {"params": base_params, "lr": float(lr)},
            {"params": head_params, "lr": float(lr) * float(head_lr_multiplier)},
        ]
    )


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _requires_prototype_source(model_id: str) -> bool:
    model_id = str(model_id)
    return model_id.startswith("query_ref_") or model_id.startswith("query_refgraph_")


def _prototype_source_enabled(model_id: str, prototype_root: str | Path | None) -> bool:
    return _requires_prototype_source(model_id) and bool(prototype_root)


def _graph_variant_enabled(model_id: str) -> bool:
    model_id = str(model_id)
    return model_id.startswith("query_graph_") or model_id.startswith("query_refgraph_")


def _model_supports_reference_bank(model: torch.nn.Module) -> bool:
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return False
    return "reference_bank" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


class _QueryPrototypeSource:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        prototype_root: str | Path,
        image_size: int,
        device_obj: torch.device,
        contract_mode: str = "compat",
        max_views: int = 0,
        view_sampler: str = "all",
        build_batch_size: int = 4,
    ) -> None:
        self.model = model
        self.device_obj = device_obj
        self.source = PrototypeBankSource(
            root=Path(prototype_root),
            image_size=image_size,
            contract_mode=contract_mode,
            max_views=max_views,
            view_sampler=view_sampler,
        )

    def resolve_for_file_name(self, file_name: str) -> PrototypeBank:
        return self.source.load_for_query(str(file_name))


def _maybe_prepare_prototype_source(
    *,
    model_id: str,
    prototype_root: str | Path | None,
    model: torch.nn.Module,
    image_size: int,
    device_obj: torch.device,
    contract_mode: str = "compat",
    max_views: int = 0,
    view_sampler: str = "all",
    build_batch_size: int = 4,
) -> _QueryPrototypeSource | None:
    if not _prototype_source_enabled(model_id, prototype_root):
        return None
    return _QueryPrototypeSource(
        model=model,
        prototype_root=Path(prototype_root),
        image_size=image_size,
        device_obj=device_obj,
        contract_mode=contract_mode,
        max_views=max_views,
        view_sampler=view_sampler,
        build_batch_size=build_batch_size,
    )


def _forward_query_batch(
    *,
    model: torch.nn.Module,
    images: torch.Tensor,
    depths: torch.Tensor,
    file_names: list[str],
    prototype_source: _QueryPrototypeSource | None,
) -> dict[str, torch.Tensor]:
    if prototype_source is None:
        return model(images, depths)
    if not _model_supports_reference_bank(model):
        raise ValueError(
            "reference query variants require a model.forward that accepts reference_bank"
        )

    grouped_indices: dict[Path, list[int]] = {}
    bank_by_root: dict[Path, PrototypeBank] = {}
    for index, file_name in enumerate(file_names):
        bank = prototype_source.resolve_for_file_name(file_name)
        root = bank.root.resolve()
        grouped_indices.setdefault(root, []).append(index)
        bank_by_root[root] = bank

    merged_tensor_outputs: dict[str, list[torch.Tensor | None]] = {}
    for root, indices in grouped_indices.items():
        subset_images = images[indices]
        subset_depths = depths[indices]
        subset_outputs = model(subset_images, subset_depths, reference_bank=bank_by_root[root])
        for key, value in subset_outputs.items():
            if not torch.is_tensor(value):
                continue
            slots = merged_tensor_outputs.setdefault(key, [None] * len(file_names))
            for local_index, sample_index in enumerate(indices):
                slots[sample_index] = value[local_index:local_index + 1]

    merged_outputs: dict[str, torch.Tensor] = {}
    for key, chunks in merged_tensor_outputs.items():
        if any(chunk is None for chunk in chunks):
            raise RuntimeError(f"Query model output '{key}' was not produced for every sample in the batch")
        merged_outputs[key] = torch.cat([chunk for chunk in chunks if chunk is not None], dim=0)
    return merged_outputs


def _graph_loss_weight(step_index: int, *, graph_loss_weight: float, graph_warmup_steps: int) -> float:
    if int(step_index) < int(graph_warmup_steps):
        return 0.0
    return float(graph_loss_weight)


def _compute_query_graph_step(
    *,
    model: torch.nn.Module,
    model_id: str,
    outputs: dict[str, torch.Tensor],
    depths: torch.Tensor,
    instance_maps: torch.Tensor,
    min_area: int,
) -> tuple[torch.Tensor, int, int, int, float, float, float]:
    graph_variant = _graph_variant_enabled(model_id)
    graph_loss_rows: list[torch.Tensor] = []
    graph_edge_count = 0
    graph_valid_edge_count = 0
    graph_positive_edge_targets = 0
    object_count = 0.0
    split_count = 0.0
    avg_cores_per_object = 0.0

    if not graph_variant:
        return (
            outputs["fg_logits"].new_zeros(()),
            0,
            0,
            0,
            object_count,
            split_count,
            avg_cores_per_object,
        )

    for batch_idx in range(int(outputs["fg_logits"].shape[0])):
        sample_outputs = {key: value[batch_idx:batch_idx + 1] for key, value in outputs.items()}
        graph_batch = build_query_graph_batch(
            outputs=sample_outputs,
            depth_map=depths[batch_idx:batch_idx + 1],
            instance_map=instance_maps[batch_idx:batch_idx + 1],
            prototype_cache=None,
            variant=model_id,
            fragment_fg_threshold=0.5,
            fragment_boundary_threshold=0.5,
            min_area=min_area,
        )
        if graph_batch.edge_targets is None or graph_batch.edge_targets.numel() == 0:
            continue
        edge_logits = score_query_graph_edges(model, graph_batch)
        valid_mask = torch.ones_like(graph_batch.edge_targets, dtype=torch.bool)
        if graph_batch.edge_ignore_mask is not None:
            valid_mask = ~graph_batch.edge_ignore_mask.to(dtype=torch.bool)
        if not bool(valid_mask.any()):
            continue
        graph_edge_count += int(graph_batch.edge_targets.numel())
        graph_valid_edge_count += int(valid_mask.sum().item())
        graph_positive_edge_targets += int(graph_batch.edge_targets[valid_mask].sum().item())
        graph_loss_rows.append(
            F.binary_cross_entropy_with_logits(
                edge_logits[valid_mask],
                graph_batch.edge_targets[valid_mask],
            )
        )
        pred_map = merge_query_graph_instances(
            graph_batch=graph_batch,
            edge_logits=edge_logits,
            threshold=0.5,
        )
        pred_count = len([int(x) for x in torch.unique(pred_map).tolist() if int(x) > 0])
        gt_count = len([int(x) for x in torch.unique(instance_maps[batch_idx]).tolist() if int(x) > 0])
        object_count += float(pred_count)
        split_count += float(max(pred_count - gt_count, 0))
        avg_cores_per_object += float(graph_batch.diagnostics.get("num_edges", 0)) / float(max(pred_count, 1))

    graph_loss = outputs["fg_logits"].new_zeros(())
    if graph_loss_rows:
        graph_loss = torch.stack(graph_loss_rows).mean()
    return (
        graph_loss,
        graph_edge_count,
        graph_valid_edge_count,
        graph_positive_edge_targets,
        object_count,
        split_count,
        avg_cores_per_object,
    )


def _predicted_masks_from_instance_map(instance_map: torch.Tensor) -> list[torch.Tensor]:
    labels = [int(x) for x in torch.unique(instance_map).tolist() if int(x) > 0]
    return [(instance_map == int(label)).to(torch.uint8) for label in labels]


def _classify_failure(gt_map: torch.Tensor, pred_map: torch.Tensor) -> str:
    gt_count = len([int(x) for x in torch.unique(gt_map).tolist() if int(x) > 0])
    pred_count = len([int(x) for x in torch.unique(pred_map).tolist() if int(x) > 0])
    pred_fg_ratio = float((pred_map > 0).float().mean().item())
    if pred_count == 0:
        return "empty"
    if pred_count == 1 and gt_count > 1 and pred_fg_ratio >= 0.15:
        return "oversized_blob"
    if pred_count < gt_count:
        return "severe_under_count"
    if pred_count > gt_count:
        return "severe_over_split"
    return "normal"


def _mask_scores_from_fg_prob(instance_map: torch.Tensor, fg_prob: torch.Tensor) -> list[float]:
    scores: list[float] = []
    for mask in _predicted_masks_from_instance_map(instance_map):
        mask_bool = mask.to(dtype=torch.bool)
        if mask_bool.any():
            scores.append(float(fg_prob[mask_bool].mean().item()))
        else:
            scores.append(0.0)
    return scores


def _write_subset_annotations(
    *,
    source_ann: Path,
    output_path: Path,
    selected_image_ids: list[int],
) -> Path:
    payload = json.loads(source_ann.read_text(encoding="utf-8"))
    selected = {int(image_id) for image_id in selected_image_ids}
    payload["images"] = [item for item in payload.get("images", []) if int(item.get("id", -1)) in selected]
    payload["annotations"] = [item for item in payload.get("annotations", []) if int(item.get("image_id", -1)) in selected]
    output_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def _load_query_model_state(
    *,
    model_id: str,
    device_obj: torch.device,
    checkpoint: Path | None,
) -> torch.nn.Module:
    model = build_query_model(model_id).to(device_obj)
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(Path(checkpoint), map_location=device_obj, weights_only=True)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return model


def _validate_eval_output_dir(output_dir: Path, checkpoint: Path) -> None:
    if output_dir.resolve() == checkpoint.resolve().parent:
        raise ValueError(
            "eval output_dir must differ from checkpoint directory to avoid in-place artifact writeback"
        )


def _run_uq_eval_outputs(
    *,
    dataset_root: Path,
    output_dir: Path,
    model_id: str,
    model: torch.nn.Module,
    prototype_source: _QueryPrototypeSource | None,
    val_loader: DataLoader,
    device_obj: torch.device,
    image_size: int,
    batch_size: int,
    max_train_steps: int,
    max_val_images: int,
    min_area: int,
    start_time: float,
    metrics_log_path: Path,
    checkpoint_path: Path | None,
) -> None:
    mask_rows: list[dict[str, float]] = []
    pathology_rows: list[dict[str, float]] = []
    match_rows: list[dict[str, float]] = []
    evaluated_image_ids: list[int] = []
    failure_counts = {
        "normal": 0,
        "empty": 0,
        "oversized_blob": 0,
        "severe_under_count": 0,
        "severe_over_split": 0,
    }
    coco_results: list[dict] = []
    model.eval()
    processed_count = 0
    for batch in val_loader:
        images = batch["images"].to(device_obj)
        depths = batch["depths"].to(device_obj)
        with torch.no_grad():
            outputs = _forward_query_batch(
                model=model,
                images=images,
                depths=depths,
                file_names=[str(file_name) for file_name in batch["file_names"]],
                prototype_source=prototype_source,
            )
        alpha_targets = _build_alpha_targets_from_batch(batch)
        for sample_idx in range(int(outputs["fg_logits"].shape[0])):
            if int(max_val_images) > 0 and processed_count >= int(max_val_images):
                break
            sample_outputs = {key: value[sample_idx:sample_idx + 1] for key, value in outputs.items()}
            graph_variant = _graph_variant_enabled(model_id)
            if graph_variant:
                graph_batch = build_query_graph_batch(
                    outputs=sample_outputs,
                    depth_map=depths[sample_idx:sample_idx + 1],
                    instance_map=batch["instance_maps"][sample_idx:sample_idx + 1],
                    prototype_cache=None,
                    variant=model_id,
                    fragment_fg_threshold=0.5,
                    fragment_boundary_threshold=0.5,
                    min_area=min_area,
                )
                edge_logits = score_query_graph_edges(model, graph_batch)
                pred_map = merge_query_graph_instances(
                    graph_batch=graph_batch,
                    edge_logits=edge_logits,
                    threshold=0.5,
                )
                pred_count = len([int(x) for x in torch.unique(pred_map).tolist() if int(x) > 0])
                gt_count = len([int(x) for x in torch.unique(batch["instance_maps"][sample_idx]).tolist() if int(x) > 0])
                stats = {
                    "object_count": float(pred_count),
                    "split_count": float(max(pred_count - gt_count, 0)),
                    "avg_cores_per_object": float(graph_batch.diagnostics.get("num_edges", 0)) / float(max(pred_count, 1)),
                }
            else:
                pred_map, stats = predict_instance_map(
                    fg_logits=outputs["fg_logits"][sample_idx, 0].detach().cpu(),
                    boundary_logits=outputs["boundary_logits"][sample_idx, 0].detach().cpu(),
                    core_heatmap=outputs["core_heatmap"][sample_idx, 0].detach().cpu(),
                    ownership_offsets=outputs["ownership_offsets"][sample_idx].detach().cpu(),
                    min_area=min_area,
                )
            fg_prob = torch.sigmoid(outputs["fg_logits"][sample_idx, 0].detach().cpu())
            boundary_prob = torch.sigmoid(outputs["boundary_logits"][sample_idx, 0].detach().cpu())
            mask_rows.append(
                {
                    "pred_fg_rate": float((fg_prob >= 0.5).float().mean().item()),
                    "pred_boundary_rate": float((boundary_prob >= 0.5).float().mean().item()),
                    "target_fg_rate": float(alpha_targets["fg"][sample_idx, 0].float().mean().item()),
                    "target_boundary_rate": float(alpha_targets["boundary"][sample_idx, 0].float().mean().item()),
                }
            )
            pathology_rows.append(stats)
            gt_instance_map = batch["instance_maps"][sample_idx]
            match = summarize_instance_matching(gt_instance_map, pred_map)
            match_rows.append(match)
            failure_counts[_classify_failure(gt_instance_map, pred_map)] += 1
            pred_masks = [mask.cpu().numpy().astype("uint8") for mask in _predicted_masks_from_instance_map(pred_map)]
            mask_scores = _mask_scores_from_fg_prob(pred_map, fg_prob)
            image_id = int(batch["image_ids"][sample_idx])
            coco_results.extend(
                masks_to_coco_results(
                    image_id=image_id,
                    masks=pred_masks,
                    scores=mask_scores,
                )
            )
            evaluated_image_ids.append(image_id)
            _append_jsonl(
                metrics_log_path,
                {
                    "mode": "eval",
                    "image_id": image_id,
                    **stats,
                    **match,
                },
            )
            processed_count += 1
        if int(max_val_images) > 0 and processed_count >= int(max_val_images):
            break

    results_json = output_dir / "coco_instances_results.json"
    results_json.write_text(json.dumps(coco_results, ensure_ascii=False) + "\n", encoding="utf-8")
    ann_file = dataset_root / "annotations" / "instances_val.json"
    subset_ann_file = _write_subset_annotations(
        source_ann=ann_file,
        output_path=output_dir / "instances_val.subset.json",
        selected_image_ids=evaluated_image_ids,
    )
    metrics = evaluate_json(subset_ann_file, results_json)
    write_json(output_dir / "metrics.cocoeval.json", metrics)
    write_json(output_dir / "mask_calibration_summary.json", summarize_mask_calibration(mask_rows))
    write_json(output_dir / "object_pathology_summary.json", summarize_object_pathology(pathology_rows))
    write_json(output_dir / "match_diagnostics_summary.json", summarize_matches(match_rows))
    write_json(
        output_dir / "failure_summary.json",
        {
            "total_images": int(sum(failure_counts.values())),
            "counts": failure_counts,
        },
    )
    save_run_summary(
        output_dir / "run_summary.json",
        UQRunSummary(
            variant=model_id,
            model_id=model_id,
            checkpoint_path=None if checkpoint_path is None else str(Path(checkpoint_path)),
            split_mode="object_first",
            use_reference=_requires_prototype_source(model_id),
            use_graph_rescue=_graph_variant_enabled(model_id),
            dataset_root=str(dataset_root),
            output_dir=str(output_dir),
            image_size=int(image_size),
            batch_size=int(batch_size),
            max_train_steps=int(max_train_steps),
            max_val_images=int(max_val_images),
            metrics=metrics,
            inference_speed={},
            params_trainable=int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
            wall_time_sec=float(time.perf_counter() - start_time),
            results_json=str(results_json),
        ),
    )


def run_uq_minibatch(
    *,
    dataset_root: Path,
    output_dir: Path,
    model_id: str,
    checkpoint: Path | None = None,
    prototype_root: Path | None = None,
    device: str = "cpu",
    image_size: int = 64,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    max_train_steps: int = 1,
    max_val_images: int = 1,
    min_area: int = 8,
    lr: float = 1.0e-4,
    head_lr_multiplier: float = 10.0,
    fg_loss_weight: float = 1.0,
    boundary_loss_weight: float = 0.5,
    core_loss_weight: float = 4.0,
    ownership_loss_weight: float = 0.25,
    ownership_warmup_steps: int = 16,
    graph_loss_weight: float = DEFAULT_GRAPH_LOSS_WEIGHT,
    graph_warmup_steps: int = DEFAULT_GRAPH_WARMUP_STEPS,
) -> None:
    start_time = time.perf_counter()
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    train_dataset = ECCGraphDataset(str(dataset_root), "train", image_size, train=True)
    val_dataset = ECCGraphDataset(str(dataset_root), "val", image_size, train=False)
    loader_kwargs = _build_query_loader_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        device_obj=device_obj,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    model = _load_query_model_state(model_id=model_id, device_obj=device_obj, checkpoint=checkpoint)
    if _requires_prototype_source(model_id) and not prototype_root:
        raise ValueError("reference query variants require prototype_root")
    if _requires_prototype_source(model_id) and not _model_supports_reference_bank(model):
        raise ValueError("reference query variants require a model.forward that accepts reference_bank")
    if _graph_variant_enabled(model_id) and getattr(model, "forward_graph", None) is None and getattr(model, "graph_head", None) is None:
        raise ValueError("graph query variants require a learned graph scorer")
    prototype_source = _maybe_prepare_prototype_source(
        model_id=model_id,
        prototype_root=prototype_root,
        model=model,
        image_size=image_size,
        device_obj=device_obj,
    )
    optimizer = _build_alpha_optimizer(model, lr=lr, head_lr_multiplier=head_lr_multiplier)
    loss_weights = {
        "fg": float(fg_loss_weight),
        "boundary": float(boundary_loss_weight),
        "core": float(core_loss_weight),
        "ownership": float(ownership_loss_weight),
    }
    metrics_log_path = output_dir / "metrics_log.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()

    train_steps = 0
    model.train()
    graph_variant = _graph_variant_enabled(model_id)
    for batch in train_loader:
        images = batch["images"].to(device_obj)
        depths = batch["depths"].to(device_obj)
        file_names = [str(file_name) for file_name in batch["file_names"]]
        alpha_targets = {
            key: value.to(device_obj)
            for key, value in _build_alpha_targets_from_batch(batch).items()
        }

        outputs = _forward_query_batch(
            model=model,
            images=images,
            depths=depths,
            file_names=file_names,
            prototype_source=prototype_source,
        )
        losses = _compute_alpha_losses(outputs, alpha_targets)
        loss_fg = losses["fg"]
        loss_boundary = losses["boundary"]
        loss_core = losses["core"]
        loss_ownership = losses["ownership"]
        graph_loss = outputs["fg_logits"].new_zeros(())
        graph_edge_count = 0
        graph_valid_edge_count = 0
        graph_positive_edge_targets = 0
        object_count = 0.0
        split_count = 0.0
        avg_cores_per_object = 0.0
        if graph_variant:
            (
                graph_loss,
                graph_edge_count,
                graph_valid_edge_count,
                graph_positive_edge_targets,
                object_count,
                split_count,
                avg_cores_per_object,
            ) = _compute_query_graph_step(
                model=model,
                model_id=model_id,
                outputs=outputs,
                depths=depths,
                instance_maps=batch["instance_maps"],
                min_area=min_area,
            )
        loss = _reduce_alpha_losses(
            losses,
            step_index=train_steps + 1,
            ownership_warmup_steps=ownership_warmup_steps,
            loss_weights=loss_weights,
        )
        if graph_variant:
            loss = loss + _graph_loss_weight(
                train_steps + 1,
                graph_loss_weight=graph_loss_weight,
                graph_warmup_steps=graph_warmup_steps,
            ) * graph_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if not graph_variant:
            for batch_idx in range(images.shape[0]):
                pred_map, stats = predict_instance_map(
                    fg_logits=outputs["fg_logits"][batch_idx, 0].detach().cpu(),
                    boundary_logits=outputs["boundary_logits"][batch_idx, 0].detach().cpu(),
                    core_heatmap=outputs["core_heatmap"][batch_idx, 0].detach().cpu(),
                    ownership_offsets=outputs["ownership_offsets"][batch_idx].detach().cpu(),
                    min_area=min_area,
                )
                object_count += stats["object_count"]
                split_count += stats["split_count"]
                avg_cores_per_object += stats["avg_cores_per_object"]

        row = {
            "mode": "train",
            "step": train_steps + 1,
            "loss": float(loss.detach().cpu()),
            "loss_fg": float(loss_fg.detach().cpu()),
            "loss_boundary": float(loss_boundary.detach().cpu()),
            "loss_core": float(loss_core.detach().cpu()),
            "loss_ownership": float(loss_ownership.detach().cpu()),
            "graph_loss": float(graph_loss.detach().cpu()),
            "graph_edge_count": int(graph_edge_count),
            "graph_valid_edge_count": int(graph_valid_edge_count),
            "graph_positive_edge_targets": int(graph_positive_edge_targets),
            "object_count": float(object_count / max(images.shape[0], 1)),
            "split_count": float(split_count / max(images.shape[0], 1)),
            "avg_cores_per_object": float(avg_cores_per_object / max(images.shape[0], 1)),
        }
        _append_jsonl(metrics_log_path, row)
        train_steps += 1
        if train_steps >= int(max_train_steps):
            break

    best_checkpoint_path = output_dir / "model_best.pth"
    final_checkpoint_path = output_dir / "model_final.pth"
    torch.save({"model": model.state_dict()}, best_checkpoint_path)
    torch.save({"model": model.state_dict()}, final_checkpoint_path)
    _run_uq_eval_outputs(
        dataset_root=dataset_root,
        output_dir=output_dir,
        model_id=model_id,
        model=model,
        prototype_source=prototype_source,
        val_loader=val_loader,
        device_obj=device_obj,
        image_size=image_size,
        batch_size=batch_size,
        max_train_steps=max_train_steps,
        max_val_images=max_val_images,
        min_area=min_area,
        start_time=start_time,
        metrics_log_path=metrics_log_path,
        checkpoint_path=final_checkpoint_path,
    )


def run_uq_eval(
    *,
    dataset_root: Path,
    output_dir: Path,
    model_id: str,
    checkpoint: Path,
    prototype_root: Path | None = None,
    device: str = "cpu",
    image_size: int = 64,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    max_val_images: int = 1,
    min_area: int = 8,
) -> None:
    start_time = time.perf_counter()
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    _validate_eval_output_dir(output_dir, checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    val_dataset = ECCGraphDataset(str(dataset_root), "val", image_size, train=False)
    val_loader_kwargs = _build_query_loader_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        device_obj=device_obj,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    model = _load_query_model_state(model_id=model_id, device_obj=device_obj, checkpoint=checkpoint)
    if _requires_prototype_source(model_id) and not prototype_root:
        raise ValueError("reference query variants require prototype_root")
    if _requires_prototype_source(model_id) and not _model_supports_reference_bank(model):
        raise ValueError("reference query variants require a model.forward that accepts reference_bank")
    if _graph_variant_enabled(model_id) and getattr(model, "forward_graph", None) is None and getattr(model, "graph_head", None) is None:
        raise ValueError("graph query variants require a learned graph scorer")
    prototype_source = _maybe_prepare_prototype_source(
        model_id=model_id,
        prototype_root=prototype_root,
        model=model,
        image_size=image_size,
        device_obj=device_obj,
    )
    metrics_log_path = output_dir / "metrics_log.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()
    _run_uq_eval_outputs(
        dataset_root=dataset_root,
        output_dir=output_dir,
        model_id=model_id,
        model=model,
        prototype_source=prototype_source,
        val_loader=val_loader,
        device_obj=device_obj,
        image_size=image_size,
        batch_size=batch_size,
        max_train_steps=0,
        max_val_images=max_val_images,
        min_area=min_area,
        start_time=start_time,
        metrics_log_path=metrics_log_path,
        checkpoint_path=checkpoint,
    )
