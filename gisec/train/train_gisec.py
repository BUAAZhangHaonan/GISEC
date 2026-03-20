from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gisec.config.io import extract_argparse_defaults, load_yaml_config, merge_config_dicts
from gisec.config.variants import get_variant_spec, variant_names
from gisec.engine.runtime import (
    PrototypeCacheSource,
    RunContext,
    RunSummary,
    build_device,
    build_loader,
    build_model,
    evaluate_and_export,
    prepare_prototype_source,
    read_git_revision,
    resolve_checkpoint,
    write_json,
)
from gisec.graph_refiner import GraphRefiner
from gisec.utils.logging import JsonlMetricLogger, setup_logger, write_metrics_csv

MODEL_CONFIG_DEFAULTS = {
    "base_channels": 16,
    "graph_hidden_dim": 64,
    "norm_layer": "group",
}


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    backend: str | None
    dist_url: str | None
    world_size: int
    rank: int
    local_rank: int


def resolve_distributed_context(
    *,
    launcher: str,
    device_name: str,
    dist_backend: str,
    dist_url: str = "env://",
    local_rank_arg: int = 0,
) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(local_rank_arg)))
    enabled = launcher == "torchrun" and world_size > 1
    if not enabled:
        return DistributedContext(
            enabled=False,
            backend=None,
            dist_url=None,
            world_size=1,
            rank=0,
            local_rank=0,
        )
    backend = str(dist_backend)
    if device_name == "cpu":
        backend = "gloo"
    return DistributedContext(
        enabled=True,
        backend=backend,
        dist_url=str(dist_url),
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )


def _is_main_process(context: DistributedContext) -> bool:
    return (not context.enabled) or context.rank == 0


def _setup_distributed(context: DistributedContext) -> None:
    if not context.enabled or dist.is_initialized():
        return
    dist.init_process_group(
        backend=str(context.backend),
        init_method=str(context.dist_url),
        rank=int(context.rank),
        world_size=int(context.world_size),
    )


def _cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--dataset-root")
    parser.add_argument("--prototype-root")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--variant", choices=list(variant_names()), default="G5")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--fragment-fg-threshold", type=float, default=0.5)
    parser.add_argument("--fragment-boundary-threshold", type=float, default=0.5)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--contract-mode",
                        choices=["compat", "strict"], default="compat")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-limit", type=int, default=8)
    parser.add_argument("--save-graph-diagnostics", action="store_true")
    parser.add_argument("--diagnostics-limit", type=int, default=64)
    parser.add_argument("--launcher", choices=["none", "torchrun"], default="none")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--sync-bn", action="store_true")
    parser.add_argument("--reference-max-views", type=int, default=0)
    parser.add_argument(
        "--reference-view-sampler",
        choices=["all", "uniform", "pose_farthest"],
        default="all",
    )
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--graph-hidden-dim", type=int, default=64)
    parser.add_argument("--norm-layer", choices=["group", "batch"], default="group")
    return parser


def _config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", action="append", default=[])
    return parser


def _load_parser_defaults(argv: list[str] | None, *, mode: str) -> dict[str, object]:
    config_args, _ = _config_parser().parse_known_args(argv)
    config_paths = list(getattr(config_args, "config", []) or [])
    if not config_paths:
        return {}
    config = merge_config_dicts(load_yaml_config(path) for path in config_paths)
    return extract_argparse_defaults(config, mode=mode)


def _validate_required_args(parser: argparse.ArgumentParser, args: argparse.Namespace, names: list[str]) -> None:
    missing = [name for name in names if getattr(args, name, None) in (None, "")]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))


def _model_config_from_args(args: argparse.Namespace) -> dict[str, int | str]:
    return {
        "base_channels": int(args.base_channels),
        "graph_hidden_dim": int(args.graph_hidden_dim),
        "norm_layer": str(args.norm_layer),
    }


def _load_model_config_sidecar(checkpoint_path: Path | None, output_dir: Path | None) -> dict[str, int | str] | None:
    candidates: list[Path] = []
    if checkpoint_path is not None:
        candidates.append(checkpoint_path.parent / "model_config.json")
    if output_dir is not None:
        candidates.append(output_dir / "model_config.json")
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def resolve_model_config(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, int | str]:
    resolved = dict(MODEL_CONFIG_DEFAULTS)
    sidecar = _load_model_config_sidecar(checkpoint_path, output_dir)
    if sidecar is not None:
        resolved.update(sidecar)
    arg_config = _model_config_from_args(args)
    for key, default_value in MODEL_CONFIG_DEFAULTS.items():
        if checkpoint_path is None or arg_config[key] != default_value:
            resolved[key] = arg_config[key]
    return resolved


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = _load_parser_defaults(argv, mode="train")
    parser = argparse.ArgumentParser(parents=[_common_parser()])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fg-dice-weight", type=float, default=0.5)
    parser.add_argument("--boundary-pos-weight", type=float, default=4.0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    _validate_required_args(parser, args, ["dataset_root", "prototype_root", "output_dir"])
    return args


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = _load_parser_defaults(argv, mode="eval")
    parser = argparse.ArgumentParser(parents=[_common_parser()])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--results-json", type=str, default="")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    _validate_required_args(parser, args, ["dataset_root", "prototype_root", "output_dir"])
    return args


def parse_infer_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = _load_parser_defaults(argv, mode="infer")
    parser = argparse.ArgumentParser(parents=[_common_parser()])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--results-json", type=str, default="")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    _validate_required_args(parser, args, ["dataset_root", "prototype_root", "output_dir"])
    return args


def relation_target_key(variant: str | object) -> str:
    variant_spec = get_variant_spec(variant)
    return "ownership_target" if variant_spec.use_ownership_supervision else "affinity_target"


def relation_target_from_batch(batch: dict[str, torch.Tensor], variant: str | object) -> torch.Tensor:
    return batch[relation_target_key(variant)]


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + float(eps)) / (union + float(eps))
    return 1.0 - dice.mean()


def balanced_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    pos_weight = torch.as_tensor(float(positive_weight), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def forward_with_reference_routing(
    *,
    model: torch.nn.Module,
    images: torch.Tensor,
    depths: torch.Tensor,
    file_names: list[str],
    prototype_source: PrototypeCacheSource,
) -> tuple[dict[str, torch.Tensor], list[object]]:
    outputs_by_sample: list[dict[str, torch.Tensor]] = []
    prototype_caches: list[object] = []
    for batch_index, file_name in enumerate(file_names):
        prototype_cache, _bank = prototype_source.resolve_for_query(file_name)
        outputs_by_sample.append(
            model(
                images[batch_index: batch_index + 1],
                query_depth=depths[batch_index: batch_index + 1],
                prototype_cache=prototype_cache,
            )
        )
        prototype_caches.append(prototype_cache)
    merged_outputs = {
        key: torch.cat([sample[key] for sample in outputs_by_sample], dim=0)
        for key in outputs_by_sample[0]
    }
    return merged_outputs, prototype_caches


def train_main(args: argparse.Namespace) -> None:
    dist_context = resolve_distributed_context(
        launcher=args.launcher,
        device_name=args.device,
        dist_backend=args.dist_backend,
        dist_url=args.dist_url,
        local_rank_arg=args.local_rank,
    )
    _setup_distributed(dist_context)
    output_dir = Path(args.output_dir).resolve()
    if _is_main_process(dist_context):
        output_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(args.device, local_rank=dist_context.local_rank if dist_context.enabled else None)
    if dist_context.enabled and device.type == "cuda":
        torch.cuda.set_device(device)
    use_cuda = device.type == "cuda"
    variant_spec = get_variant_spec(args.variant)
    run_context = RunContext(
        dataset_root=str(Path(args.dataset_root).resolve()),
        prototype_root=str(Path(args.prototype_root).resolve()),
        split="val",
        image_size=int(args.image_size),
        batch=int(args.batch),
        num_workers=int(args.num_workers),
        min_area=int(args.min_area),
        fragment_fg_threshold=float(args.fragment_fg_threshold),
        fragment_boundary_threshold=float(args.fragment_boundary_threshold),
        edge_threshold=float(args.edge_threshold),
        contract_mode=args.contract_mode,
        device=str(device),
        code_revision=read_git_revision(Path(__file__).resolve().parents[2]),
    )
    log_path = output_dir / ("run.log" if _is_main_process(dist_context) else f"run.rank{dist_context.rank}.log")
    metrics_log_path = output_dir / (
        "metrics_log.jsonl" if _is_main_process(dist_context) else f"metrics_log.rank{dist_context.rank}.jsonl"
    )
    logger = setup_logger("gisec", log_path)
    metric_logger = JsonlMetricLogger(metrics_log_path)

    train_loader = build_loader(
        dataset_root=args.dataset_root,
        split="train",
        image_size=args.image_size,
        train=True,
        batch_size=args.batch,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
        distributed=dist_context.enabled,
        rank=dist_context.rank,
        world_size=dist_context.world_size,
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

    model_config = resolve_model_config(args, output_dir=output_dir)
    if _is_main_process(dist_context):
        write_json(output_dir / "model_config.json", model_config)
    model = build_model(device, **model_config)
    if dist_context.enabled and args.sync_bn and use_cuda:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if dist_context.enabled:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    model_for_graph = _unwrap_model(model)
    refiner = GraphRefiner(model_for_graph)
    prototype_source = prepare_prototype_source(
        model=model_for_graph,
        device=device,
        prototype_root=args.prototype_root,
        image_size=args.image_size,
        contract_mode=args.contract_mode,
        max_views=int(args.reference_max_views),
        view_sampler=str(args.reference_view_sampler),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    ann_file = Path(args.dataset_root) / "annotations" / "instances_val.json"
    params_trainable = sum(int(p.numel())
                           for p in model.parameters() if p.requires_grad)
    if _is_main_process(dist_context):
        (output_dir / "params_trainable.txt").write_text(str(params_trainable) +
                                                         "\n", encoding="utf-8")

    metrics_path = output_dir / "metrics.json"
    if _is_main_process(dist_context) and metrics_path.exists():
        metrics_path.unlink()

    best_ap = -1.0
    best_ckpt = output_dir / "model_best.pth"
    start = time.time()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            if dist_context.enabled and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            for step, batch in enumerate(train_loader, start=1):
                file_names = list(batch["file_names"])
                images = batch["images"].to(device)
                depths = batch["depths"].to(device)
                fg_target = batch["fg_target"].to(device)
                boundary_target = batch["boundary_target"].to(device)
                relation_target = relation_target_from_batch(batch, variant_spec).to(device)
                instance_maps = batch["instance_maps"].to(device)
                prototype_source.clear()

                with torch.cuda.amp.autocast(enabled=use_cuda):
                    outputs, prototype_caches = forward_with_reference_routing(
                        model=model,
                        images=images,
                        depths=depths,
                        file_names=file_names,
                        prototype_source=prototype_source,
                    )
                    loss_fg_bce = F.binary_cross_entropy_with_logits(outputs["fg_logits"], fg_target)
                    loss_fg_dice = dice_loss_with_logits(outputs["fg_logits"], fg_target)
                    loss_fg = loss_fg_bce + float(args.fg_dice_weight) * loss_fg_dice
                    loss_boundary = balanced_bce_with_logits(
                        outputs["boundary_logits"],
                        boundary_target,
                        positive_weight=float(args.boundary_pos_weight),
                    )
                    relation_pred = outputs["ownership_offsets"]
                    if variant_spec.use_ownership_supervision:
                        fg_mask = fg_target.expand_as(relation_target) > 0.5
                        if fg_mask.any():
                            loss_relation = F.smooth_l1_loss(
                                relation_pred[fg_mask], relation_target[fg_mask]
                            )
                        else:
                            loss_relation = relation_pred.sum() * 0.0
                    else:
                        loss_relation = F.binary_cross_entropy_with_logits(relation_pred, relation_target)
                    loss = loss_fg + loss_boundary + 0.5 * loss_relation
                    graph_edge_count = 0
                    graph_positive_edge_targets = 0.0
                    graph_has_edges = False
                    graph_loss_value = 0.0

                    if variant_spec.use_learned_edge_scorer:
                        graph_losses = []
                        for batch_idx in range(images.shape[0]):
                            graph_batch = refiner.build_graph_batch(
                                outputs={key: value[batch_idx: batch_idx + 1] for key, value in outputs.items()},
                                depth_map=depths[batch_idx: batch_idx + 1],
                                instance_map=instance_maps[batch_idx],
                                prototype_cache=prototype_caches[batch_idx],
                                variant=variant_spec,
                                fragment_fg_threshold=float(args.fragment_fg_threshold),
                                fragment_boundary_threshold=float(args.fragment_boundary_threshold),
                                min_area=int(args.min_area),
                            )
                            graph_edge_count += int(
                                graph_batch.diagnostics.get("num_edges", int(graph_batch.edge_index.shape[1]))
                            )
                            graph_has_edges = graph_has_edges or bool(graph_batch.edge_index.shape[1] > 0)
                            if graph_batch.edge_targets is not None:
                                valid_mask = (
                                    torch.ones_like(graph_batch.edge_targets, dtype=torch.bool)
                                    if graph_batch.edge_ignore_mask is None
                                    else ~graph_batch.edge_ignore_mask
                                )
                                graph_positive_edge_targets += float(
                                    graph_batch.edge_targets[valid_mask].sum().item()
                                )
                            if graph_batch.edge_targets is None or graph_batch.edge_targets.numel() == 0:
                                continue
                            valid_mask = (
                                torch.ones_like(graph_batch.edge_targets, dtype=torch.bool)
                                if graph_batch.edge_ignore_mask is None
                                else ~graph_batch.edge_ignore_mask
                            )
                            if not bool(valid_mask.any()):
                                continue
                            edge_logits = refiner.score_edges(graph_batch, variant_spec)
                            graph_losses.append(
                                F.binary_cross_entropy_with_logits(
                                    edge_logits[valid_mask], graph_batch.edge_targets[valid_mask]
                                )
                            )
                        if graph_losses:
                            graph_loss = torch.stack(graph_losses).mean()
                            graph_loss_value = float(graph_loss.detach().cpu())
                            loss = loss + 0.5 * graph_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                metric_row = {
                    "mode": "train",
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "loss_fg": float(loss_fg.detach().cpu()),
                    "loss_fg_bce": float(loss_fg_bce.detach().cpu()),
                    "loss_fg_dice": float(loss_fg_dice.detach().cpu()),
                    "loss_boundary": float(loss_boundary.detach().cpu()),
                    "loss_relation": float(loss_relation.detach().cpu()),
                    "relation_target": relation_target_key(variant_spec),
                    "graph_has_edges": int(graph_has_edges),
                    "graph_edge_count": int(graph_edge_count),
                    "graph_positive_edge_targets": float(graph_positive_edge_targets),
                    "graph_loss": float(graph_loss_value),
                }
                if _is_main_process(dist_context):
                    metric_logger.append(metric_row)

                if step % 20 == 0 and _is_main_process(dist_context):
                    logger.info(
                        "epoch=%s step=%s loss=%.4f loss_fg=%.4f loss_fg_bce=%.4f loss_fg_dice=%.4f loss_boundary=%.4f loss_relation=%.4f relation_target=%s graph_edges=%s graph_has_edges=%s graph_pos_targets=%.1f graph_loss=%.4f",
                        epoch,
                        step,
                        float(loss.detach().cpu()),
                        float(loss_fg.detach().cpu()),
                        float(loss_fg_bce.detach().cpu()),
                        float(loss_fg_dice.detach().cpu()),
                        float(loss_boundary.detach().cpu()),
                        float(loss_relation.detach().cpu()),
                        relation_target_key(variant_spec),
                        int(graph_edge_count),
                        int(graph_has_edges),
                        float(graph_positive_edge_targets),
                        float(graph_loss_value),
                    )
                if int(args.max_train_steps) > 0 and step >= int(args.max_train_steps):
                    break

            if _is_main_process(dist_context):
                epoch_results = output_dir / f"epoch_{epoch:04d}_results.json"
                metrics, _benchmark = evaluate_and_export(
                    model=model_for_graph,
                    loader=val_loader,
                    device=device,
                    prototype_source=prototype_source,
                    variant=variant_spec,
                    ann_file=ann_file,
                    results_json=epoch_results,
                    min_area=args.min_area,
                    fragment_fg_threshold=args.fragment_fg_threshold,
                    fragment_boundary_threshold=args.fragment_boundary_threshold,
                    edge_threshold=args.edge_threshold,
                    max_images=int(args.max_val_images) if int(
                        args.max_val_images) > 0 else None,
                )
                metrics["iteration"] = epoch
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                metric_logger.append({"mode": "eval", **metrics})
                segm_ap = float(metrics.get("segm/AP", 0.0))
                if segm_ap >= best_ap:
                    best_ap = segm_ap
                    torch.save(model_for_graph.state_dict(), best_ckpt)
                logger.info("epoch=%s best_ap=%.4f", epoch, best_ap)
            if dist_context.enabled:
                dist.barrier()

        if _is_main_process(dist_context):
            final_ckpt = output_dir / "model_final.pth"
            torch.save(model_for_graph.state_dict(), final_ckpt)
            final_results = output_dir / "coco_instances_results.json"
            final_metrics, inference_speed = evaluate_and_export(
                model=model_for_graph,
                loader=val_loader,
                device=device,
                prototype_source=prototype_source,
                variant=variant_spec,
                ann_file=ann_file,
                results_json=final_results,
                min_area=args.min_area,
                fragment_fg_threshold=args.fragment_fg_threshold,
                fragment_boundary_threshold=args.fragment_boundary_threshold,
                edge_threshold=args.edge_threshold,
                max_images=int(args.max_val_images) if int(
                    args.max_val_images) > 0 else None,
                artifact_dir=output_dir,
                save_overlays=bool(args.save_overlays),
                overlay_limit=int(args.overlay_limit),
                save_graph_diagnostics=bool(args.save_graph_diagnostics),
                diagnostics_limit=int(args.diagnostics_limit),
            )
            final_metrics["iteration"] = int(args.epochs)
            wall_time_sec = int(time.time() - start)
            write_json(output_dir / "metrics.cocoeval.json", final_metrics)
            write_json(output_dir / "inference_speed.json", inference_speed)
            write_metrics_csv(output_dir / "metrics_log.csv", metric_logger.rows)
            peak_memory_mb = 0.0
            if device.type == "cuda" and torch.cuda.is_available():
                peak_memory_mb = float(
                    torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            (output_dir /
             "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
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
                        dataset_root=run_context.dataset_root,
                        prototype_root=run_context.prototype_root,
                        split=run_context.split,
                        image_size=run_context.image_size,
                        batch=run_context.batch,
                        num_workers=run_context.num_workers,
                        min_area=run_context.min_area,
                        fragment_fg_threshold=run_context.fragment_fg_threshold,
                        fragment_boundary_threshold=run_context.fragment_boundary_threshold,
                        edge_threshold=run_context.edge_threshold,
                        device=run_context.device,
                        code_revision=run_context.code_revision,
                        params_trainable=params_trainable,
                        training_peak_memory_mb=peak_memory_mb,
                        wall_time_sec=wall_time_sec,
                    )
                ),
            )
            write_json(output_dir / "prototype_bank_manifest.json", prototype_source.describe())
            (output_dir / "last_checkpoint").write_text(final_ckpt.name + "\n", encoding="utf-8")
            (output_dir / "wall_time_sec.txt").write_text(str(wall_time_sec) + "\n", encoding="utf-8")
            logger.info("final_best_ap=%.4f training_peak_memory_mb=%.4f",
                        best_ap, peak_memory_mb)
        if dist_context.enabled:
            dist.barrier()
    finally:
        _cleanup_distributed(dist_context)


def _run_eval_like(args: argparse.Namespace, *, compute_metrics: bool) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(args.device)
    use_cuda = device.type == "cuda"
    variant_spec = get_variant_spec(args.variant)
    run_context = RunContext(
        dataset_root=str(Path(args.dataset_root).resolve()),
        prototype_root=str(Path(args.prototype_root).resolve()),
        split=args.split,
        image_size=int(args.image_size),
        batch=1,
        num_workers=int(args.num_workers),
        min_area=int(args.min_area),
        fragment_fg_threshold=float(args.fragment_fg_threshold),
        fragment_boundary_threshold=float(args.fragment_boundary_threshold),
        edge_threshold=float(args.edge_threshold),
        contract_mode=args.contract_mode,
        device=str(device),
        code_revision=read_git_revision(Path(__file__).resolve().parents[2]),
    )
    loader = build_loader(
        dataset_root=args.dataset_root,
        split=args.split,
        image_size=args.image_size,
        train=False,
        batch_size=1,
        num_workers=args.num_workers,
        use_cuda=use_cuda,
    )
    checkpoint_path = resolve_checkpoint(output_dir, args.checkpoint)
    model_config = resolve_model_config(args, checkpoint_path=checkpoint_path, output_dir=output_dir)
    write_json(output_dir / "model_config.json", model_config)
    model = build_model(device, checkpoint_path, **model_config)
    prototype_source = prepare_prototype_source(
        model=model,
        device=device,
        prototype_root=args.prototype_root,
        image_size=args.image_size,
        contract_mode=args.contract_mode,
        max_views=int(args.reference_max_views),
        view_sampler=str(args.reference_view_sampler),
    )
    results_json = Path(args.results_json).resolve(
    ) if args.results_json else output_dir / "coco_instances_results.json"
    ann_file = None
    if compute_metrics:
        ann_candidate = Path(args.dataset_root) / \
            "annotations" / f"instances_{args.split}.json"
        if ann_candidate.exists():
            ann_file = ann_candidate
    metrics, inference_speed = evaluate_and_export(
        model=model,
        loader=loader,
        device=device,
        prototype_source=prototype_source,
        variant=variant_spec,
        ann_file=ann_file,
        results_json=results_json,
        min_area=args.min_area,
        fragment_fg_threshold=args.fragment_fg_threshold,
        fragment_boundary_threshold=args.fragment_boundary_threshold,
        edge_threshold=args.edge_threshold,
        max_images=int(args.max_images) if int(args.max_images) > 0 else None,
        artifact_dir=output_dir,
        save_overlays=bool(args.save_overlays),
        overlay_limit=int(args.overlay_limit),
        save_graph_diagnostics=bool(args.save_graph_diagnostics),
        diagnostics_limit=int(args.diagnostics_limit),
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
                dataset_root=run_context.dataset_root,
                prototype_root=run_context.prototype_root,
                split=run_context.split,
                image_size=run_context.image_size,
                batch=run_context.batch,
                num_workers=run_context.num_workers,
                min_area=run_context.min_area,
                fragment_fg_threshold=run_context.fragment_fg_threshold,
                fragment_boundary_threshold=run_context.fragment_boundary_threshold,
                edge_threshold=run_context.edge_threshold,
                device=run_context.device,
                code_revision=run_context.code_revision,
            )
        ),
    )
    write_json(output_dir / "prototype_bank_manifest.json", prototype_source.describe())


def eval_main(args: argparse.Namespace) -> None:
    _run_eval_like(args, compute_metrics=True)


def infer_main(args: argparse.Namespace) -> None:
    _run_eval_like(args, compute_metrics=args.split != "test")


def main(argv: list[str] | None = None) -> None:
    train_main(parse_train_args(argv))


if __name__ == "__main__":
    main()
