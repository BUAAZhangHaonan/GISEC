from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gisec.config.io import extract_argparse_defaults, load_yaml_config, merge_config_dicts
from gisec.config.variants import VariantSpec, get_variant_spec, variant_names
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
    resolve_num_workers,
    sync_cuda,
    write_json,
)
from gisec.graph_refiner import GraphRefiner
from gisec.utils.logging import JsonlMetricLogger, setup_logger, write_metrics_csv

MODEL_CONFIG_DEFAULTS = {
    "base_channels": 16,
    "graph_hidden_dim": 64,
    "norm_layer": "group",
    "prototype_slot_count": 6,
    "prototype_topk": 2,
    "fg_prior": 0.093,
    "boundary_prior": 0.024,
    "reference_conditioning_mode": "full",
    "reference_routing_mode": "soft_topk",
    "reference_skip_margin": 0.0,
}

REFERENCE_CONDITIONING_ALIASES = {
    "false": "off",
    "off": "off",
    "none": "off",
    "0": "off",
    "true": "full",
    "on": "full",
    "1": "full",
    "full": "full",
    "bottleneck_only": "bottleneck_only",
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
    parser.add_argument("--num-workers", type=int, default=None)
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
    parser.add_argument("--prototype-slot-count", type=int, default=6)
    parser.add_argument("--prototype-topk", type=int, default=2)
    parser.add_argument("--fg-prior", type=float, default=0.093)
    parser.add_argument("--boundary-prior", type=float, default=0.024)
    parser.add_argument(
        "--reference-conditioning-mode",
        choices=["full", "bottleneck_only", "off"],
        default="full",
    )
    parser.add_argument(
        "--reference-routing-mode",
        choices=["soft_topk", "hard_top1"],
        default="soft_topk",
    )
    parser.add_argument("--reference-skip-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--profile-start-step", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--profile-output-dir", type=str, default="")
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
        "prototype_slot_count": int(args.prototype_slot_count),
        "prototype_topk": int(args.prototype_topk),
        "fg_prior": float(args.fg_prior),
        "boundary_prior": float(args.boundary_prior),
        "reference_conditioning_mode": normalize_reference_conditioning_mode(args.reference_conditioning_mode),
        "reference_routing_mode": str(args.reference_routing_mode),
        "reference_skip_margin": float(args.reference_skip_margin),
    }


def _load_model_config_sidecar(checkpoint_path: Path | None, output_dir: Path | None) -> dict[str, int | str] | None:
    candidates: list[Path] = []
    if checkpoint_path is not None:
        candidates.append(checkpoint_path.parent / "model_config.json")
    if output_dir is not None:
        candidates.append(output_dir / "model_config.json")
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if "reference_conditioning_mode" in payload:
                payload["reference_conditioning_mode"] = normalize_reference_conditioning_mode(
                    payload["reference_conditioning_mode"]
                )
            return payload
    return None


def normalize_reference_conditioning_mode(value: object) -> str:
    if isinstance(value, bool):
        return "full" if value else "off"
    text = str(value).strip()
    normalized = REFERENCE_CONDITIONING_ALIASES.get(text.lower())
    if normalized is not None:
        return normalized
    return text


def _scalar_is_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        return bool(torch.isfinite(detached).all().item())
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _non_finite_scalar_names(values: dict[str, object]) -> list[str]:
    return [name for name, value in values.items() if not _scalar_is_finite(value)]


def _epoch_artifact_dir(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"epoch_{int(epoch):04d}_artifacts"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _profile_output_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if str(getattr(args, "profile_output_dir", "")).strip():
        return Path(str(args.profile_output_dir)).resolve()
    return output_dir / "profile"


def _profile_step_active(
    *,
    args: argparse.Namespace,
    dist_context: DistributedContext,
    step: int,
) -> bool:
    if not _is_main_process(dist_context):
        return False
    profile_steps = int(getattr(args, "profile_steps", 0))
    if profile_steps <= 0:
        return False
    profile_start_step = max(1, int(getattr(args, "profile_start_step", 1)))
    profile_end_step = profile_start_step + profile_steps - 1
    return profile_start_step <= int(step) <= profile_end_step


def _profile_sync(device: torch.device, *, enabled: bool) -> None:
    if enabled:
        sync_cuda(device)


def _summarize_step_profiles(
    rows: list[dict[str, Any]],
    *,
    epoch_eval_timings: list[float],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "profiled_steps": int(len(rows)),
        "epoch_eval_count": int(len(epoch_eval_timings)),
        "epoch_eval_wall_time_sec": float(np.median(epoch_eval_timings)) if epoch_eval_timings else 0.0,
    }
    if not rows:
        return summary
    numeric_keys = [key for key in rows[0] if key.endswith("_sec") or key in {"forward_call_count", "unique_prototype_roots", "graph_edge_count", "gpu_peak_memory_mb"}]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        summary[f"median_{key}"] = float(np.median(np.asarray(values, dtype=np.float64)))
        summary[f"mean_{key}"] = float(np.mean(np.asarray(values, dtype=np.float64)))
    cycle_values = [
        float(row["data_wait_sec"]) + float(row["step_total_sec"])
        for row in rows
        if "data_wait_sec" in row and "step_total_sec" in row
    ]
    if cycle_values:
        summary["median_cycle_sec"] = float(np.median(np.asarray(cycle_values, dtype=np.float64)))
        summary["mean_cycle_sec"] = float(np.mean(np.asarray(cycle_values, dtype=np.float64)))
    return summary


def _normalize_args_in_place(args: argparse.Namespace) -> argparse.Namespace:
    args.reference_conditioning_mode = normalize_reference_conditioning_mode(
        getattr(args, "reference_conditioning_mode", "full")
    )
    return args


def _reference_mode_was_explicit(argv: list[str] | None, *, mode: str) -> bool:
    argv_list = list(argv or [])
    if "--reference-conditioning-mode" in argv_list:
        return True
    defaults = _load_parser_defaults(argv_list, mode=mode)
    return "reference_conditioning_mode" in defaults


def _apply_reference_mode_for_variant(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    argv: list[str] | None,
    mode: str,
) -> argparse.Namespace:
    variant_spec = get_variant_spec(args.variant)
    explicit_reference_mode = _reference_mode_was_explicit(argv, mode=mode)
    if not explicit_reference_mode:
        args.reference_conditioning_mode = "full" if variant_spec.use_reference_conditioning else "off"
    args.reference_conditioning_mode = normalize_reference_conditioning_mode(args.reference_conditioning_mode)
    if (not variant_spec.use_reference_conditioning) and str(args.reference_conditioning_mode) != "off":
        parser.error(
            f"legacy variant {variant_spec.name} does not allow reference conditioning; "
            "use --reference-conditioning-mode off"
        )
    return args


def _prototype_source_enabled(variant: str | VariantSpec, reference_conditioning_mode: str) -> bool:
    variant_spec = get_variant_spec(variant)
    return bool(
        variant_spec.use_reference_conditioning
        and normalize_reference_conditioning_mode(reference_conditioning_mode) != "off"
    )


def _maybe_prepare_prototype_source(
    *,
    model: object,
    device: torch.device | str,
    args: argparse.Namespace,
    dataset_root: str | Path,
) -> PrototypeCacheSource | None:
    if not _prototype_source_enabled(args.variant, args.reference_conditioning_mode):
        return None
    return prepare_prototype_source(
        model=model,
        device=device,
        prototype_root=args.prototype_root,
        dataset_root=str(dataset_root),
        image_size=args.image_size,
        contract_mode=args.contract_mode,
        max_views=int(args.reference_max_views),
        view_sampler=str(args.reference_view_sampler),
    )


def _cli_arg_value(argv: list[str] | None, flag: str) -> str | None:
    if argv is None:
        return None
    value = None
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            value = argv[index + 1]
    return value


def _variant_requires_prototype_source(variant: str | VariantSpec) -> bool:
    return bool(get_variant_spec(variant).use_reference_conditioning)


def _apply_variant_reference_policy(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    argv: list[str] | None,
    defaults: dict[str, object],
) -> argparse.Namespace:
    args = _normalize_args_in_place(args)
    variant_spec = get_variant_spec(args.variant)
    if variant_spec.use_reference_conditioning:
        return args
    cli_mode = _cli_arg_value(argv, "--reference-conditioning-mode")
    config_mode = defaults.get("reference_conditioning_mode")
    explicit_mode = cli_mode if cli_mode not in {None, ""} else config_mode
    if explicit_mode not in {None, ""} and normalize_reference_conditioning_mode(explicit_mode) != "off":
        parser.error(
            f"--reference-conditioning-mode {normalize_reference_conditioning_mode(explicit_mode)} "
            f"is not allowed for no-reference variant {variant_spec.name}"
        )
    args.reference_conditioning_mode = "off"
    return args


def set_random_seed(seed: int, *, use_cuda: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    parser.add_argument("--fg-pos-weight", type=float, default=9.0)
    parser.add_argument("--fg-dice-weight", type=float, default=0.5)
    parser.add_argument("--boundary-pos-weight", type=float, default=40.0)
    parser.add_argument("--graph-warmup-steps", type=int, default=16)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args = _apply_variant_reference_policy(parser, args, argv=argv, defaults=defaults)
    required_args = ["dataset_root", "output_dir"]
    if _prototype_source_enabled(args.variant, args.reference_conditioning_mode):
        required_args.append("prototype_root")
    _validate_required_args(parser, args, required_args)
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
    args = _apply_variant_reference_policy(parser, args, argv=argv, defaults=defaults)
    required_args = ["dataset_root", "output_dir"]
    if _prototype_source_enabled(args.variant, args.reference_conditioning_mode):
        required_args.append("prototype_root")
    _validate_required_args(parser, args, required_args)
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
    args = _apply_variant_reference_policy(parser, args, argv=argv, defaults=defaults)
    required_args = ["dataset_root", "output_dir"]
    if _prototype_source_enabled(args.variant, args.reference_conditioning_mode):
        required_args.append("prototype_root")
    _validate_required_args(parser, args, required_args)
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


def _prob_quantile(probs: torch.Tensor, q: float) -> float:
    return float(torch.quantile(probs.float().flatten(), float(q)).cpu())


def forward_with_reference_routing(
    *,
    model: torch.nn.Module,
    images: torch.Tensor,
    depths: torch.Tensor,
    file_names: list[str],
    prototype_source: PrototypeCacheSource | None,
    reference_conditioning_mode: str = "full",
    reference_routing_mode: str = "soft_topk",
    reference_skip_margin: float = 0.0,
    return_reference_routing: bool = True,
) -> tuple[dict[str, torch.Tensor], list[object], dict[str, int]]:
    reference_conditioning_mode = normalize_reference_conditioning_mode(reference_conditioning_mode)
    batch_size = int(images.shape[0])
    if batch_size == 0:
        return {}, [], {"forward_call_count": 0, "unique_prototype_roots": 0}
    if prototype_source is None or str(reference_conditioning_mode) == "off":
        outputs = model(
            images,
            query_depth=depths,
            prototype_cache=None,
            reference_conditioning_mode=reference_conditioning_mode,
            reference_routing_mode=reference_routing_mode,
            reference_skip_margin=reference_skip_margin,
            return_reference_routing=return_reference_routing,
        )
        return outputs, [None] * batch_size, {"forward_call_count": 1, "unique_prototype_roots": 0}

    prototype_caches: list[object] = []
    group_indices: dict[str, list[int]] = {}
    group_caches: dict[str, object] = {}
    group_order: list[str] = []
    for batch_index, file_name in enumerate(file_names):
        prototype_cache, bank = prototype_source.resolve_for_query(file_name)
        cache_key = str(bank.root)
        if cache_key not in group_indices:
            group_indices[cache_key] = []
            group_caches[cache_key] = prototype_cache
            group_order.append(cache_key)
        group_indices[cache_key].append(int(batch_index))
        prototype_caches.append(prototype_cache)

    grouped_outputs: dict[str, list[Any]] = {}
    forward_call_count = 0
    for cache_key in group_order:
        indices = group_indices[cache_key]
        grouped_result = model(
            images[indices],
            query_depth=depths[indices],
            prototype_cache=group_caches[cache_key],
            reference_conditioning_mode=reference_conditioning_mode,
            reference_routing_mode=reference_routing_mode,
            reference_skip_margin=reference_skip_margin,
            return_reference_routing=return_reference_routing,
        )
        forward_call_count += 1
        for key, value in grouped_result.items():
            slots = grouped_outputs.setdefault(key, [None] * batch_size)
            if isinstance(value, torch.Tensor):
                for offset, batch_index in enumerate(indices):
                    slots[batch_index] = value[offset: offset + 1]
            elif isinstance(value, (list, tuple)) and len(value) == len(indices):
                for batch_index, item in zip(indices, value):
                    slots[batch_index] = item
            else:
                for batch_index in indices:
                    slots[batch_index] = value

    merged_outputs: dict[str, Any] = {}
    for key, values in grouped_outputs.items():
        first_value = values[0]
        if isinstance(first_value, torch.Tensor):
            merged_outputs[key] = torch.cat([value for value in values if isinstance(value, torch.Tensor)], dim=0)
        else:
            merged_outputs[key] = values
    return merged_outputs, prototype_caches, {
        "forward_call_count": int(forward_call_count),
        "unique_prototype_roots": int(len(group_order)),
    }


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
    set_random_seed(int(args.seed), use_cuda=use_cuda)
    variant_spec = get_variant_spec(args.variant)
    resolved_num_workers = resolve_num_workers(args.num_workers)
    run_context = RunContext(
        dataset_root=str(Path(args.dataset_root).resolve()),
        prototype_root="" if getattr(args, "prototype_root", "") in (None, "") else str(Path(args.prototype_root).resolve()),
        split="val",
        image_size=int(args.image_size),
        batch=int(args.batch),
        num_workers=int(resolved_num_workers),
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
    prototype_source = _maybe_prepare_prototype_source(
        model=model_for_graph,
        device=device,
        args=args,
        dataset_root=args.dataset_root,
    )
    reference_part_keys = None
    if prototype_source is not None and not prototype_source.source.is_single_bank:
        reference_part_keys = list(prototype_source.source.available_parts)

    train_loader = build_loader(
        dataset_root=args.dataset_root,
        split="train",
        image_size=args.image_size,
        train=True,
        batch_size=args.batch,
        num_workers=resolved_num_workers,
        use_cuda=use_cuda,
        distributed=dist_context.enabled,
        rank=dist_context.rank,
        world_size=dist_context.world_size,
        reference_part_keys=reference_part_keys,
    )
    val_loader = build_loader(
        dataset_root=args.dataset_root,
        split="val",
        image_size=args.image_size,
        train=False,
        batch_size=1,
        num_workers=resolved_num_workers,
        use_cuda=use_cuda,
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
    profile_rows: list[dict[str, Any]] = []
    epoch_eval_timings: list[float] = []
    profile_jsonl_path: Path | None = None
    profile_summary_path: Path | None = None
    if _is_main_process(dist_context) and int(getattr(args, "profile_steps", 0)) > 0:
        profile_dir = _profile_output_dir(args, output_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_jsonl_path = profile_dir / "step_profile.jsonl"
        profile_summary_path = profile_dir / "step_profile_summary.json"
        for path in [profile_jsonl_path, profile_summary_path]:
            if path.exists():
                path.unlink()

    best_ap = -1.0
    best_ckpt = output_dir / "model_best.pth"
    start = time.time()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            if prototype_source is not None:
                prototype_source.clear()
            if dist_context.enabled and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            epoch_graph_loss_values: list[float] = []
            epoch_non_finite_event_count = 0
            train_iter = iter(train_loader)
            step = 0
            while True:
                next_step = step + 1
                profile_active = _profile_step_active(args=args, dist_context=dist_context, step=next_step)
                data_wait_start = time.perf_counter() if profile_active else 0.0
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
                step = next_step
                step_profile: dict[str, Any] | None = None
                if profile_active:
                    _profile_sync(device, enabled=True)
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(device)
                    step_profile = {
                        "epoch": int(epoch),
                        "step": int(step),
                        "data_wait_sec": float(time.perf_counter() - data_wait_start),
                    }
                    step_total_start = time.perf_counter()
                h2d_start = time.perf_counter() if profile_active else 0.0
                file_names = list(batch["file_names"])
                images = batch["images"].to(device, non_blocking=use_cuda)
                depths = batch["depths"].to(device, non_blocking=use_cuda)
                fg_target = batch["fg_target"].to(device, non_blocking=use_cuda)
                boundary_target = batch["boundary_target"].to(device, non_blocking=use_cuda)
                relation_target = relation_target_from_batch(batch, variant_spec).to(device, non_blocking=use_cuda)
                instance_maps = batch["instance_maps"].to(device, non_blocking=use_cuda)
                if step_profile is not None:
                    _profile_sync(device, enabled=True)
                    step_profile["h2d_and_prep_sec"] = float(time.perf_counter() - h2d_start)

                routing_stats = {"forward_call_count": 0, "unique_prototype_roots": 0}
                model_forward_sec = 0.0
                dense_loss_sec = 0.0
                graph_build_sec = 0.0
                graph_score_and_loss_sec = 0.0
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    forward_start = time.perf_counter() if profile_active else 0.0
                    outputs, prototype_caches, routing_stats = forward_with_reference_routing(
                        model=model,
                        images=images,
                        depths=depths,
                        file_names=file_names,
                        prototype_source=prototype_source,
                        reference_conditioning_mode=str(args.reference_conditioning_mode),
                        reference_routing_mode=str(args.reference_routing_mode),
                        reference_skip_margin=float(args.reference_skip_margin),
                        return_reference_routing=False,
                    )
                    if step_profile is not None:
                        _profile_sync(device, enabled=True)
                        model_forward_sec = float(time.perf_counter() - forward_start)
                    dense_loss_start = time.perf_counter() if profile_active else 0.0
                    loss_fg_bce = balanced_bce_with_logits(
                        outputs["fg_logits"],
                        fg_target,
                        positive_weight=float(args.fg_pos_weight),
                    )
                    loss_fg_dice = dice_loss_with_logits(outputs["fg_logits"], fg_target)
                    loss_fg = loss_fg_bce + float(args.fg_dice_weight) * loss_fg_dice
                    loss_boundary = balanced_bce_with_logits(
                        outputs["boundary_logits"],
                        boundary_target,
                        positive_weight=float(args.boundary_pos_weight),
                    )
                    relation_pred = outputs["ownership_offsets"]
                    fg_probs = torch.sigmoid(outputs["fg_logits"].detach())
                    boundary_probs = torch.sigmoid(outputs["boundary_logits"].detach())
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
                    graph_loss_tensor: torch.Tensor | None = None
                    if step_profile is not None:
                        _profile_sync(device, enabled=True)
                        dense_loss_sec = float(time.perf_counter() - dense_loss_start)

                    if variant_spec.use_learned_edge_scorer:
                        graph_losses = []
                        for batch_idx in range(images.shape[0]):
                            graph_build_start = time.perf_counter() if profile_active else 0.0
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
                            if step_profile is not None:
                                graph_build_sec += float(time.perf_counter() - graph_build_start)
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
                            graph_score_start = time.perf_counter() if profile_active else 0.0
                            edge_logits = refiner.score_edges(graph_batch, variant_spec)
                            graph_losses.append(
                                F.binary_cross_entropy_with_logits(
                                    edge_logits[valid_mask], graph_batch.edge_targets[valid_mask]
                                )
                            )
                            if step_profile is not None:
                                _profile_sync(device, enabled=True)
                                graph_score_and_loss_sec += float(time.perf_counter() - graph_score_start)
                        if graph_losses:
                            graph_loss = torch.stack(graph_losses).mean()
                            graph_loss_tensor = graph_loss
                            graph_loss_weight = 0.0 if step <= int(args.graph_warmup_steps) else 0.5
                            graph_loss_value = float((graph_loss * graph_loss_weight).detach().cpu())
                            loss = loss + graph_loss_weight * graph_loss

                non_finite_scalars = _non_finite_scalar_names(
                    {
                        "loss": loss,
                        "loss_fg": loss_fg,
                        "loss_boundary": loss_boundary,
                        "loss_relation": loss_relation,
                        "graph_loss": graph_loss_tensor if graph_loss_tensor is not None else graph_loss_value,
                    }
                )
                if non_finite_scalars:
                    epoch_non_finite_event_count += 1
                    raise RuntimeError(
                        f"Non-finite training scalars detected at epoch={epoch} step={step}: "
                        + ", ".join(non_finite_scalars)
                    )

                epoch_graph_loss_values.append(float(graph_loss_value))

                backward_start = time.perf_counter() if profile_active else 0.0
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if step_profile is not None:
                    _profile_sync(device, enabled=True)
                    backward_sec = float(time.perf_counter() - backward_start)
                else:
                    backward_sec = 0.0
                optimizer_start = time.perf_counter() if profile_active else 0.0
                scaler.step(optimizer)
                scaler.update()
                if step_profile is not None:
                    _profile_sync(device, enabled=True)
                    optimizer_step_sec = float(time.perf_counter() - optimizer_start)
                else:
                    optimizer_step_sec = 0.0

                metric_io_start = time.perf_counter() if profile_active else 0.0
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
                    "pred_fg_rate": float((fg_probs >= 0.5).float().mean().cpu()),
                    "pred_boundary_rate": float((boundary_probs >= 0.5).float().mean().cpu()),
                    "target_fg_rate": float(fg_target.mean().cpu()),
                    "target_boundary_rate": float(boundary_target.mean().cpu()),
                    "fg_prob_p50": _prob_quantile(fg_probs, 0.50),
                    "fg_prob_p90": _prob_quantile(fg_probs, 0.90),
                    "fg_prob_p95": _prob_quantile(fg_probs, 0.95),
                    "boundary_prob_p50": _prob_quantile(boundary_probs, 0.50),
                    "boundary_prob_p90": _prob_quantile(boundary_probs, 0.90),
                    "boundary_prob_p95": _prob_quantile(boundary_probs, 0.95),
                    "relation_target": relation_target_key(variant_spec),
                    "graph_has_edges": int(graph_has_edges),
                    "graph_edge_count": int(graph_edge_count),
                    "graph_positive_edge_targets": float(graph_positive_edge_targets),
                    "graph_loss": float(graph_loss_value),
                }
                if _is_main_process(dist_context):
                    metric_logger.append(metric_row)
                if step_profile is not None:
                    step_profile.update(
                        {
                            "model_forward_sec": float(model_forward_sec),
                            "dense_loss_sec": float(dense_loss_sec),
                            "graph_build_sec": float(graph_build_sec),
                            "graph_score_and_loss_sec": float(graph_score_and_loss_sec),
                            "backward_sec": float(backward_sec),
                            "optimizer_step_sec": float(optimizer_step_sec),
                            "forward_call_count": int(routing_stats["forward_call_count"]),
                            "unique_prototype_roots": int(routing_stats["unique_prototype_roots"]),
                            "graph_edge_count": int(graph_edge_count),
                            "gpu_peak_memory_mb": (
                                float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                                if use_cuda and torch.cuda.is_available()
                                else 0.0
                            ),
                        }
                    )

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
                if step_profile is not None:
                    step_profile["step_metric_io_sec"] = float(time.perf_counter() - metric_io_start)
                    step_profile["step_total_sec"] = float(time.perf_counter() - step_total_start)
                    profile_rows.append(step_profile)
                    if profile_jsonl_path is not None:
                        _append_jsonl(profile_jsonl_path, step_profile)
                    logger.info(
                        "profile epoch=%s step=%s data_wait=%.4fs h2d=%.4fs forward=%.4fs dense_loss=%.4fs graph_build=%.4fs graph_score=%.4fs backward=%.4fs optimizer=%.4fs metric_io=%.4fs total=%.4fs forward_calls=%s unique_roots=%s graph_edges=%s",
                        epoch,
                        step,
                        float(step_profile["data_wait_sec"]),
                        float(step_profile["h2d_and_prep_sec"]),
                        float(step_profile["model_forward_sec"]),
                        float(step_profile["dense_loss_sec"]),
                        float(step_profile["graph_build_sec"]),
                        float(step_profile["graph_score_and_loss_sec"]),
                        float(step_profile["backward_sec"]),
                        float(step_profile["optimizer_step_sec"]),
                        float(step_profile["step_metric_io_sec"]),
                        float(step_profile["step_total_sec"]),
                        int(step_profile["forward_call_count"]),
                        int(step_profile["unique_prototype_roots"]),
                        int(step_profile["graph_edge_count"]),
                    )
                if int(args.max_train_steps) > 0 and step >= int(args.max_train_steps):
                    break

            if _is_main_process(dist_context):
                epoch_results = output_dir / f"epoch_{epoch:04d}_results.json"
                epoch_artifact_dir = (
                    _epoch_artifact_dir(output_dir, epoch)
                    if bool(args.save_graph_diagnostics)
                    else None
                )
                if prototype_source is not None:
                    prototype_source.clear()
                eval_start = time.perf_counter()
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
                    artifact_dir=epoch_artifact_dir,
                    save_graph_diagnostics=bool(args.save_graph_diagnostics),
                )
                eval_wall_time_sec = float(time.perf_counter() - eval_start)
                epoch_eval_timings.append(eval_wall_time_sec)
                metrics["iteration"] = epoch
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                metric_logger.append({"mode": "eval", **metrics})
                epoch_summary_row = {
                    "mode": "epoch_eval",
                    "epoch": int(epoch),
                    "graph_loss_mean": 0.0 if not epoch_graph_loss_values else float(np.mean(epoch_graph_loss_values)),
                    "graph_loss_max": 0.0 if not epoch_graph_loss_values else float(np.max(epoch_graph_loss_values)),
                    "num_merged_mean": 0.0,
                    "num_merged_std": 0.0,
                    "num_merged_min": 0.0,
                    "num_merged_max": 0.0,
                    "gt_count_mean": 0.0,
                    "pred_count_mean": 0.0,
                    "non_finite_event_count": int(epoch_non_finite_event_count),
                    "eval_wall_time_sec": float(eval_wall_time_sec),
                }
                if epoch_artifact_dir is not None:
                    graph_readiness_path = epoch_artifact_dir / "graph_readiness_summary.json"
                    match_summary_path = epoch_artifact_dir / "match_diagnostics_summary.json"
                    if graph_readiness_path.exists():
                        graph_readiness_summary = _read_json(graph_readiness_path)
                        for key in ["num_merged_mean", "num_merged_std", "num_merged_min", "num_merged_max"]:
                            if key in graph_readiness_summary:
                                epoch_summary_row[key] = float(graph_readiness_summary[key])
                    if match_summary_path.exists():
                        match_summary = _read_json(match_summary_path)
                        for key in ["gt_count_mean", "pred_count_mean"]:
                            if key in match_summary:
                                epoch_summary_row[key] = float(match_summary[key])
                metric_logger.append(epoch_summary_row)
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
            if prototype_source is not None:
                write_json(output_dir / "prototype_bank_manifest.json", prototype_source.describe())
            (output_dir / "last_checkpoint").write_text(final_ckpt.name + "\n", encoding="utf-8")
            (output_dir / "wall_time_sec.txt").write_text(str(wall_time_sec) + "\n", encoding="utf-8")
            logger.info("final_best_ap=%.4f training_peak_memory_mb=%.4f",
                        best_ap, peak_memory_mb)
            if profile_summary_path is not None:
                write_json(
                    profile_summary_path,
                    _summarize_step_profiles(
                        profile_rows,
                        epoch_eval_timings=epoch_eval_timings,
                    ),
                )
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
    resolved_num_workers = resolve_num_workers(args.num_workers)
    run_context = RunContext(
        dataset_root=str(Path(args.dataset_root).resolve()),
        prototype_root="" if getattr(args, "prototype_root", "") in (None, "") else str(Path(args.prototype_root).resolve()),
        split=args.split,
        image_size=int(args.image_size),
        batch=1,
        num_workers=int(resolved_num_workers),
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
        num_workers=resolved_num_workers,
        use_cuda=use_cuda,
    )
    checkpoint_path = resolve_checkpoint(output_dir, args.checkpoint)
    model_config = resolve_model_config(args, checkpoint_path=checkpoint_path, output_dir=output_dir)
    write_json(output_dir / "model_config.json", model_config)
    model = build_model(device, checkpoint_path, **model_config)
    prototype_source = _maybe_prepare_prototype_source(
        model=model,
        device=device,
        args=args,
        dataset_root=args.dataset_root,
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
    if prototype_source is not None:
        write_json(output_dir / "prototype_bank_manifest.json", prototype_source.describe())


def eval_main(args: argparse.Namespace) -> None:
    _run_eval_like(args, compute_metrics=True)


def infer_main(args: argparse.Namespace) -> None:
    _run_eval_like(args, compute_metrics=args.split != "test")


def main(argv: list[str] | None = None) -> None:
    train_main(parse_train_args(argv))


if __name__ == "__main__":
    main()
