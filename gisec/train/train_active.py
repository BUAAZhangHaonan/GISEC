from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from baseline.common.boundary_metrics import compute_boundary_iou
from baseline.common.coco_export import masks_to_coco_results
from baseline.common.dataset import BaselineInstanceDataset
from baseline.common.export import build_run_summary_payload
from baseline.mask2former.adapter import (
    build_mask2former_model,
    build_mask2former_processor,
    outputs_to_instance_masks,
)
from gisec.active.config import active_variant_names, get_active_variant_spec
from gisec.active.metrics import compute_split_merge_counts
from gisec.active.model import (
    ActiveInstanceModel,
    boundary_target_from_mask,
    crop_and_resize,
    expand_bbox,
    mask_bbox,
    paste_mask_from_crop,
    prepare_active_input_batch,
    prepare_reference_depth,
)
from gisec.active.runtime import select_refinement_instances
from gisec.cli._routing import explicit_cli_variant, resolve_run_directory_variant
from gisec.config.io import extract_argparse_defaults, load_yaml_config, merge_config_dicts
from gisec.datasets.prototype_bank import PrototypeBank, PrototypeBankSource, extract_query_part_key
from gisec.engine.runtime import build_benchmark_payload, build_device, evaluate_json, write_json


MODEL_DEFAULTS = {
    "variant": "base_rgb_1024",
    "image_size": 1024,
    "batch": 1,
    "num_workers": 4,
    "device": "cuda",
    "epochs": 20,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "score_threshold": 0.5,
    "mask_threshold": 0.5,
    "pretrained_model_name": "facebook/mask2former-swin-tiny-coco-instance",
    "hidden_dim": 64,
    "feature_size": 64,
    "mask_feature_size": 64,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "num_attention_heads": 4,
    "num_queries": 16,
    "train_num_points": 512,
    "refiner_hidden_dim": 32,
    "graph_hidden_dim": 64,
    "crop_size": 256,
    "crop_pad": 16,
    "boundary_band_width": 4,
    "max_train_steps": 0,
    "max_val_images": 0,
    "eval_every_epochs": 1,
    "log_every_steps": 50,
    "reference_max_views": 16,
    "reference_view_sampler": "pose_farthest",
    "prototype_root": "",
    "init_checkpoint": "",
    "resume_checkpoint": "",
    "checkpoint": "",
    "split": "val",
    "dry_run": False,
    "resume_save_every_epochs": 1,
}

ACTIVE_DEPTH_MODES = ("rgb", "rgbd_concat", "rgbd_concat_valid_mask")


def _config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", action="append", default=[])
    return parser


def _load_parser_defaults(argv: list[str] | None, *, mode: str) -> dict[str, Any]:
    config_args, _ = _config_parser().parse_known_args(argv)
    config_paths = list(getattr(config_args, "config", []) or [])
    if not config_paths:
        return {}
    config = merge_config_dicts(load_yaml_config(path) for path in config_paths)
    defaults = extract_argparse_defaults(config, mode=mode)
    for key in [
        "variant",
        "score_threshold",
        "mask_threshold",
        "pretrained_model_name",
        "hidden_dim",
        "feature_size",
        "mask_feature_size",
        "encoder_layers",
        "decoder_layers",
        "num_attention_heads",
        "num_queries",
        "train_num_points",
        "depth_mode",
        "refiner_hidden_dim",
        "graph_hidden_dim",
        "crop_size",
        "crop_pad",
        "boundary_band_width",
        "reference_max_views",
        "reference_view_sampler",
    ]:
        model_key = f"model_{key}"
        if model_key in defaults and key not in defaults:
            defaults[key] = defaults[model_key]
    return defaults


def _resolved_active_variant_default(argv: list[str] | None) -> str:
    run_variant = resolve_run_directory_variant(list(argv or []))
    if run_variant in set(active_variant_names()):
        return str(run_variant)
    return str(MODEL_DEFAULTS["variant"])


def _validate_variant_source_consistency(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None,
) -> None:
    cli_variant = explicit_cli_variant(list(argv or []))
    run_variant = resolve_run_directory_variant(list(argv or []))
    if run_variant in {None, "", "__legacy__"}:
        return
    if cli_variant not in {None, ""} and str(cli_variant) != str(run_variant):
        parser.error(
            f"--variant {cli_variant} conflicts with run metadata variant {run_variant}"
        )
    if str(args.variant) != str(run_variant):
        parser.error(
            f"parsed active variant {args.variant} does not match run metadata variant {run_variant}"
        )


def _annotate_variant_sources(args: argparse.Namespace, argv: list[str] | None) -> None:
    setattr(args, "_explicit_variant", explicit_cli_variant(list(argv or [])))
    setattr(args, "_run_metadata_variant", resolve_run_directory_variant(list(argv or [])))


def _common_parser(*, mode: str, argv: list[str] | None) -> argparse.ArgumentParser:
    defaults = _load_parser_defaults(argv, mode=mode)
    parser = argparse.ArgumentParser(parents=[_config_parser()])
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--prototype-root", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument(
        "--variant",
        choices=list(active_variant_names()),
        default=_resolved_active_variant_default(argv),
    )
    parser.add_argument("--depth-mode", choices=list(ACTIVE_DEPTH_MODES), default="")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model-name", type=str, default=MODEL_DEFAULTS["pretrained_model_name"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--mask-feature-size", type=int, default=64)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--train-num-points", type=int, default=512)
    parser.add_argument("--refiner-hidden-dim", type=int, default=32)
    parser.add_argument("--graph-hidden-dim", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--crop-pad", type=int, default=16)
    parser.add_argument("--boundary-band-width", type=int, default=4)
    parser.add_argument("--reference-max-views", type=int, default=16)
    parser.add_argument("--reference-view-sampler", choices=["all", "uniform", "pose_farthest"], default="pose_farthest")
    parser.add_argument("--allow-partial-checkpoint-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    if mode == "train":
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--learning-rate", type=float, default=1.0e-4)
        parser.add_argument("--weight-decay", type=float, default=1.0e-4)
        parser.add_argument("--max-train-steps", type=int, default=0)
        parser.add_argument("--max-val-images", type=int, default=0)
        parser.add_argument("--eval-every-epochs", type=int, default=1)
        parser.add_argument("--log-every-steps", type=int, default=int(MODEL_DEFAULTS["log_every_steps"]))
        parser.add_argument("--resume-save-every-epochs", type=int, default=int(MODEL_DEFAULTS["resume_save_every_epochs"]))
        parser.add_argument("--allow-unsafe-resume", action="store_true")
    else:
        parser.add_argument("--checkpoint", type=str, default="")
        parser.add_argument("--split", choices=["train", "val"], default="val")
        parser.add_argument("--max-images", type=int, default=0)
    parser.set_defaults(**defaults)
    return parser


def _validate_required_args(parser: argparse.ArgumentParser, args: argparse.Namespace, required: list[str]) -> None:
    missing = [name for name in required if getattr(args, name, None) in (None, "")]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))


def _resolve_checkpoint_path(checkpoint_dir: Path, checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute():
        return checkpoint_path.resolve()
    return (checkpoint_dir / checkpoint_path).resolve()


def _validate_variant_requirements(parser: argparse.ArgumentParser, args: argparse.Namespace, *, is_eval: bool) -> None:
    variant_spec = get_active_variant_spec(args.variant)
    depth_mode = str(getattr(args, "depth_mode", "") or variant_spec.depth_mode)
    if variant_spec.depth_mode == "rgb":
        if depth_mode != "rgb":
            parser.error(f"--depth-mode {depth_mode} is not allowed for active variant {variant_spec.name}")
    elif depth_mode not in {"rgbd_concat", "rgbd_concat_valid_mask"}:
        parser.error(f"--depth-mode {depth_mode} is not allowed for active variant {variant_spec.name}")
    args.depth_mode = depth_mode
    if variant_spec.requires_prototype_root and getattr(args, "prototype_root", "") in ("", None):
        parser.error(f"--prototype-root is required for active variant {variant_spec.name}")
    if (not is_eval) and variant_spec.use_local_refine and getattr(args, "init_checkpoint", "") in ("", None):
        parser.error(f"--init-checkpoint is required for active variant {variant_spec.name}")
    if is_eval and getattr(args, "checkpoint", "") in ("", None):
        parser.error("--checkpoint is required for eval/infer")


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser(mode="train", argv=argv)
    args = parser.parse_args(argv)
    _annotate_variant_sources(args, argv)
    _validate_variant_source_consistency(parser, args, argv)
    _validate_required_args(parser, args, ["dataset_root", "output_dir"])
    _validate_variant_requirements(parser, args, is_eval=False)
    return args


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser(mode="eval", argv=argv)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    args = parser.parse_args(argv)
    _annotate_variant_sources(args, argv)
    _validate_variant_source_consistency(parser, args, argv)
    _validate_required_args(parser, args, ["dataset_root", "output_dir"])
    _validate_variant_requirements(parser, args, is_eval=True)
    return args


def parse_infer_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser(mode="infer", argv=argv)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    args = parser.parse_args(argv)
    _annotate_variant_sources(args, argv)
    _validate_variant_source_consistency(parser, args, argv)
    _validate_required_args(parser, args, ["dataset_root", "output_dir"])
    _validate_variant_requirements(parser, args, is_eval=True)
    return args


def _model_payload(args: argparse.Namespace) -> dict[str, Any]:
    variant_spec = get_active_variant_spec(args.variant)
    return {
        "variant": variant_spec.name,
        "depth_mode": str(args.depth_mode),
        "use_local_refine": variant_spec.use_local_refine,
        "use_reference_rescue": variant_spec.use_reference_rescue,
        "use_graph_rescue": variant_spec.use_graph_rescue,
        "requires_prototype_root": variant_spec.requires_prototype_root,
        "image_size": int(args.image_size),
        "batch": int(args.batch),
        "num_workers": int(args.num_workers),
        "device": str(args.device),
        "score_threshold": float(args.score_threshold),
        "mask_threshold": float(args.mask_threshold),
        "pretrained_model_name": None if str(args.pretrained_model_name).strip().lower() in {"", "none"} else str(args.pretrained_model_name),
        "hidden_dim": int(args.hidden_dim),
        "feature_size": int(args.feature_size),
        "mask_feature_size": int(args.mask_feature_size),
        "encoder_layers": int(args.encoder_layers),
        "decoder_layers": int(args.decoder_layers),
        "num_attention_heads": int(args.num_attention_heads),
        "num_queries": int(args.num_queries),
        "train_num_points": int(args.train_num_points),
        "refiner_hidden_dim": int(args.refiner_hidden_dim),
        "graph_hidden_dim": int(args.graph_hidden_dim),
        "crop_size": int(args.crop_size),
        "crop_pad": int(args.crop_pad),
        "boundary_band_width": int(args.boundary_band_width),
        "reference_max_views": int(args.reference_max_views),
        "reference_view_sampler": str(args.reference_view_sampler),
        "prototype_root": str(getattr(args, "prototype_root", "")),
        "init_checkpoint": str(getattr(args, "init_checkpoint", "")),
        "resume_checkpoint": str(getattr(args, "resume_checkpoint", "")),
        "checkpoint": str(getattr(args, "checkpoint", "")),
        "split": str(getattr(args, "split", "val")),
        "max_train_steps": int(getattr(args, "max_train_steps", 0)),
        "max_val_images": int(getattr(args, "max_val_images", 0)),
        "max_images": int(getattr(args, "max_images", 0)),
        "epochs": int(getattr(args, "epochs", 0)),
        "log_every_steps": int(getattr(args, "log_every_steps", MODEL_DEFAULTS["log_every_steps"])),
        "learning_rate": float(getattr(args, "learning_rate", 0.0)),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "eval_every_epochs": int(getattr(args, "eval_every_epochs", 1)),
        "allow_partial_checkpoint_load": bool(getattr(args, "allow_partial_checkpoint_load", False)),
        "resume_save_every_epochs": int(getattr(args, "resume_save_every_epochs", MODEL_DEFAULTS["resume_save_every_epochs"])),
    }


def _checkpoint_payload(model: nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "variant": str(args.variant),
        "depth_mode": str(args.depth_mode),
        "model": _model_payload(args),
    }


def _serialize_train_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[str(key)] = value
        else:
            payload[str(key)] = str(value)
    return payload


def _capture_rng_state() -> dict[str, Any]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    payload: dict[str, Any] = {
        "python": [
            int(python_state[0]),
            [int(value) for value in python_state[1]],
            python_state[2],
        ],
        "numpy": [
            str(numpy_state[0]),
            [int(value) for value in numpy_state[1].tolist()],
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch_cpu": [int(value) for value in torch.get_rng_state().tolist()],
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = [
            [int(value) for value in state.cpu().tolist()]
            for state in torch.cuda.get_rng_state_all()
        ]
    return payload


def _restore_rng_state(payload: dict[str, Any] | None) -> None:
    if not isinstance(payload, dict):
        return
    python_state = payload.get("python")
    if python_state is not None:
        if isinstance(python_state, tuple):
            random.setstate(python_state)
        elif isinstance(python_state, list) and len(python_state) == 3:
            random.setstate(
                (
                    int(python_state[0]),
                    tuple(int(value) for value in python_state[1]),
                    python_state[2],
                )
            )
    numpy_state = payload.get("numpy")
    if numpy_state is not None:
        if isinstance(numpy_state, tuple):
            np.random.set_state(numpy_state)
        elif isinstance(numpy_state, list) and len(numpy_state) == 5:
            np.random.set_state(
                (
                    str(numpy_state[0]),
                    np.array(numpy_state[1], dtype=np.uint32),
                    int(numpy_state[2]),
                    int(numpy_state[3]),
                    float(numpy_state[4]),
                )
            )
    torch_cpu_state = payload.get("torch_cpu")
    if torch_cpu_state is not None:
        if isinstance(torch_cpu_state, torch.Tensor):
            torch.set_rng_state(torch_cpu_state)
        else:
            torch.set_rng_state(torch.tensor(torch_cpu_state, dtype=torch.uint8))
    torch_cuda_state = payload.get("torch_cuda")
    if torch_cuda_state is not None and torch.cuda.is_available():
        if torch_cuda_state and isinstance(torch_cuda_state[0], torch.Tensor):
            torch.cuda.set_rng_state_all(torch_cuda_state)
        else:
            torch.cuda.set_rng_state_all(
                [torch.tensor(state, dtype=torch.uint8) for state in torch_cuda_state]
            )


def _resume_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    completed_epoch: int,
    global_step: int,
    best_metric: float,
    running_step_time_total: float,
) -> dict[str, Any]:
    return {
        **_checkpoint_payload(model, args),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "completed_epoch": int(completed_epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "running_step_time_total": float(running_step_time_total),
        "train_args": _serialize_train_args(args),
    }


def _resume_metadata_payload() -> dict[str, Any]:
    return {
        "rng_state": _capture_rng_state(),
    }


def _load_resume_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    allow_unsafe_resume: bool = False,
) -> tuple[int, int, float, float]:
    resume_checkpoint = Path(str(args.resume_checkpoint)).resolve()
    if not resume_checkpoint.exists():
        raise FileNotFoundError(resume_checkpoint)
    _validate_resume_checkpoint_allowed(resume_checkpoint)
    try:
        payload = torch.load(str(resume_checkpoint), map_location="cpu", weights_only=True)
    except Exception as safe_exc:
        if not bool(allow_unsafe_resume):
            raise RuntimeError(
                f"resume checkpoint {resume_checkpoint} is not weights-only safe; "
                "rerun with --allow-unsafe-resume for legacy payloads"
            ) from safe_exc
        payload = torch.load(str(resume_checkpoint), map_location="cpu", weights_only=False)
    _validate_checkpoint_variant(
        expected_variant=str(args.variant),
        checkpoint_payload=payload,
        checkpoint_path=resume_checkpoint,
    )
    _load_module_state_dict(
        model,
        _extract_state_dict(payload),
        allow_partial=False,
        context=f"resume checkpoint {resume_checkpoint}",
    )
    optimizer.load_state_dict(dict(payload.get("optimizer_state_dict", {})))
    scaler_state = payload.get("scaler_state_dict")
    if isinstance(scaler_state, dict):
        scaler.load_state_dict(scaler_state)
    metadata_path = _resume_metadata_path(resume_checkpoint)
    metadata = _read_json(metadata_path) if metadata_path.exists() else None
    _restore_rng_state(
        metadata.get("rng_state") if isinstance(metadata, dict) and metadata.get("rng_state") is not None else payload.get("rng_state")
    )
    completed_epoch = int(payload.get("completed_epoch", 0))
    global_step = int(payload.get("global_step", 0))
    best_metric = float(payload.get("best_metric", float("-inf")))
    running_step_time_total = float(payload.get("running_step_time_total", 0.0))
    return completed_epoch, global_step, best_metric, running_step_time_total


def _extract_state_dict(payload: dict[str, Any], *, prefix_backbone: bool = False) -> dict[str, Any]:
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = dict(payload["state_dict"])
    else:
        state_dict = dict(payload)
    if prefix_backbone:
        if state_dict and not any(key.startswith("backbone.") for key in state_dict):
            state_dict = {f"backbone.{key}": value for key, value in state_dict.items()}
    return state_dict


def _filter_compatible_state_dict(source_state: dict[str, Any], target_state: dict[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key, value in source_state.items():
        if key not in target_state:
            continue
        target_value = target_state[key]
        if hasattr(value, "shape") and hasattr(target_value, "shape"):
            if tuple(value.shape) != tuple(target_value.shape):
                continue
        filtered[key] = value
    return filtered


def _checkpoint_variant(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    candidates = [
        payload.get("variant"),
        payload.get("config", {}).get("variant") if isinstance(payload.get("config"), dict) else None,
        payload.get("model", {}).get("variant") if isinstance(payload.get("model"), dict) else None,
    ]
    for candidate in candidates:
        if candidate not in {"", None}:
            return str(candidate)
    return None


def _state_dict_mismatch_report(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    missing_keys = sorted(key for key in target_state if key not in source_state)
    unexpected_keys = sorted(key for key in source_state if key not in target_state)
    shape_mismatches: list[str] = []
    for key in sorted(set(source_state).intersection(target_state)):
        source_value = source_state[key]
        target_value = target_state[key]
        if hasattr(source_value, "shape") and hasattr(target_value, "shape"):
            if tuple(source_value.shape) != tuple(target_value.shape):
                shape_mismatches.append(
                    f"{key}: checkpoint {tuple(source_value.shape)} != model {tuple(target_value.shape)}"
                )
    return missing_keys, unexpected_keys, shape_mismatches


def _load_module_state_dict(
    module: nn.Module,
    source_state: dict[str, Any],
    *,
    allow_partial: bool,
    context: str,
) -> None:
    target_state = module.state_dict()
    missing_keys, unexpected_keys, shape_mismatches = _state_dict_mismatch_report(
        source_state,
        target_state,
    )
    if not allow_partial and (missing_keys or unexpected_keys or shape_mismatches):
        raise RuntimeError(
            f"{context} does not match the requested model; "
            f"missing_keys={missing_keys}; "
            f"unexpected_keys={unexpected_keys}; "
            f"shape_mismatches={shape_mismatches}"
        )
    if allow_partial:
        module.load_state_dict(
            _filter_compatible_state_dict(source_state, target_state),
            strict=False,
        )
        return
    module.load_state_dict(source_state, strict=True)


def _backbone_state_dict(source_state: dict[str, Any]) -> dict[str, Any]:
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in source_state.items()
        if key.startswith("backbone.")
    }
    return backbone_state or dict(source_state)


def _validate_checkpoint_variant(
    *,
    expected_variant: str,
    checkpoint_payload: Any,
    checkpoint_path: str | Path,
) -> None:
    checkpoint_variant = _checkpoint_variant(checkpoint_payload)
    if checkpoint_variant in {None, "", str(expected_variant)}:
        return
    raise RuntimeError(
        f"Checkpoint {Path(checkpoint_path).resolve()} declares variant {checkpoint_variant}, "
        f"but the requested active variant is {expected_variant}."
    )


def _validate_runtime_checkpoint_variant(
    *,
    requested_variant: str,
    run_variant: str | None,
    checkpoint_payload: Any,
    checkpoint_path: str | Path,
    context: str,
) -> None:
    checkpoint_variant = _checkpoint_variant(checkpoint_payload)
    if checkpoint_variant in {None, ""}:
        return
    if str(checkpoint_variant) != str(requested_variant):
        raise RuntimeError(
            f"{context} checkpoint {Path(checkpoint_path).resolve()} declares variant {checkpoint_variant}, "
            f"but the requested active variant is {requested_variant}."
        )
    if run_variant not in {None, "", "__legacy__"} and str(checkpoint_variant) != str(run_variant):
        raise RuntimeError(
            f"{context} checkpoint {Path(checkpoint_path).resolve()} declares variant {checkpoint_variant}, "
            f"but run metadata resolves to {run_variant}."
        )


def _resolve_input_channels(depth_mode: str) -> int:
    if str(depth_mode) == "rgb":
        return 3
    if str(depth_mode) == "rgbd_concat":
        return 4
    if str(depth_mode) == "rgbd_concat_valid_mask":
        return 5
    raise ValueError(f"Unsupported active depth_mode: {depth_mode}")


def _build_active_model(args: argparse.Namespace) -> ActiveInstanceModel:
    variant_spec = get_active_variant_spec(args.variant)
    depth_mode = str(getattr(args, "depth_mode", "") or variant_spec.depth_mode)
    input_channels = _resolve_input_channels(depth_mode)
    backbone = build_mask2former_model(
        image_size=int(args.image_size),
        pretrained_model_name=None if str(args.pretrained_model_name).strip().lower() in {"", "none"} else str(args.pretrained_model_name),
        input_channels=int(input_channels),
        hidden_dim=int(args.hidden_dim),
        feature_size=int(args.feature_size),
        mask_feature_size=int(args.mask_feature_size),
        encoder_layers=int(args.encoder_layers),
        decoder_layers=int(args.decoder_layers),
        num_attention_heads=int(args.num_attention_heads),
        num_queries=int(args.num_queries),
        train_num_points=int(args.train_num_points),
    )
    feature_channels = int(getattr(backbone.config, "hidden_dim", int(args.feature_size)))
    return ActiveInstanceModel(
        backbone=backbone,
        feature_channels=feature_channels,
        refine_feature_channels=16,
        query_channels=int(input_channels),
        use_local_refine=variant_spec.use_local_refine,
        use_reference_rescue=variant_spec.use_reference_rescue,
        use_graph_rescue=variant_spec.use_graph_rescue,
        refiner_hidden_dim=int(args.refiner_hidden_dim),
        graph_hidden_dim=int(args.graph_hidden_dim),
    )


def _configure_model_for_stage(model: nn.Module, args: argparse.Namespace) -> None:
    variant_spec = get_active_variant_spec(args.variant)
    if not variant_spec.use_local_refine:
        return
    for param in model.backbone.parameters():
        param.requires_grad = False
    init_checkpoint = Path(str(args.init_checkpoint)).resolve()
    if not init_checkpoint.exists():
        raise FileNotFoundError(init_checkpoint)
    checkpoint_payload = torch.load(str(init_checkpoint), map_location="cpu")
    state_dict = _extract_state_dict(checkpoint_payload, prefix_backbone=True)
    _load_module_state_dict(
        model.backbone,
        _backbone_state_dict(state_dict),
        allow_partial=bool(getattr(args, "allow_partial_checkpoint_load", False)),
        context=f"init checkpoint {init_checkpoint}",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class NonFiniteActiveTrainingError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _collect_non_finite_paths(value: object, *, prefix: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value.detach()).all().item()):
            failures.append(prefix)
        return failures
    if isinstance(value, dict):
        for key, item in value.items():
            failures.extend(_collect_non_finite_paths(item, prefix=f"{prefix}.{key}"))
        return failures
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            failures.extend(_collect_non_finite_paths(item, prefix=f"{prefix}[{index}]"))
        return failures
    if isinstance(value, (float, int)):
        if not math.isfinite(float(value)):
            failures.append(prefix)
    return failures


def _assert_finite_tensor(name: str, value: torch.Tensor | None) -> None:
    if value is None:
        return
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise NonFiniteActiveTrainingError(f"Non-finite tensor detected in {name}")


def _run_state_path(output_dir: Path) -> Path:
    return output_dir / "run_state.json"


def _write_run_state(
    output_dir: Path,
    *,
    status: str,
    allow_resume: bool,
    failure_reason: str | None,
    last_finite_step: int,
    last_finite_checkpoint: str | None,
) -> None:
    write_json(
        _run_state_path(output_dir),
        {
            "status": str(status),
            "allow_resume": bool(allow_resume),
            "failure_reason": None if failure_reason in (None, "") else str(failure_reason),
            "last_finite_step": int(last_finite_step),
            "last_finite_checkpoint": "" if not last_finite_checkpoint else str(last_finite_checkpoint),
            "pid": int(os.getpid()),
            "updated_at": float(time.time()),
        },
    )


def _is_process_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stage_lock_path(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}.train.lock")


def _acquire_stage_lock(output_dir: Path) -> Path:
    lock_path = _stage_lock_path(output_dir)
    payload = {
        "pid": int(os.getpid()),
        "created_at": float(time.time()),
        "output_dir": str(output_dir),
    }
    try:
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        existing_pid = None
        try:
            existing = _read_json(lock_path)
            existing_pid = int(existing.get("pid", -1))
        except Exception:
            existing_pid = None
        if existing_pid is not None and _is_process_alive(existing_pid):
            raise RuntimeError(f"Training stage lock is already held by pid={existing_pid}: {lock_path}") from exc
        raise RuntimeError(
            "Training stage lock already exists and must be removed manually before retrying: "
            f"{lock_path}"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return lock_path


def _release_stage_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _validate_resume_checkpoint_allowed(resume_checkpoint: Path) -> dict[str, Any]:
    run_state_path = resume_checkpoint.resolve().parent / "run_state.json"
    if not run_state_path.exists():
        raise RuntimeError(
            f"resume checkpoint requires sibling run_state.json with status=running and allow_resume=true: {resume_checkpoint}"
        )
    payload = _read_json(run_state_path)
    if str(payload.get("status", "")) != "running" or not bool(payload.get("allow_resume", False)):
        raise RuntimeError(
            f"resume checkpoint is not resumable according to run_state.json: {resume_checkpoint}"
        )
    return payload


def _save_torch_payload(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _save_json_payload(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _resume_metadata_path(resume_checkpoint: Path) -> Path:
    return resume_checkpoint.with_suffix(".meta.json")


def _validate_resume_payload_finite(payload: dict[str, Any]) -> None:
    failures = _collect_non_finite_paths(payload.get("state_dict", {}), prefix="state_dict")
    failures.extend(_collect_non_finite_paths(payload.get("optimizer_state_dict", {}), prefix="optimizer_state_dict"))
    failures.extend(_collect_non_finite_paths(payload.get("scaler_state_dict", {}), prefix="scaler_state_dict"))
    if failures:
        preview = ", ".join(failures[:8])
        raise NonFiniteActiveTrainingError(f"Refusing to save non-finite resume payload: {preview}")


def _validate_checkpoint_payload_finite(payload: dict[str, Any]) -> None:
    failures = _collect_non_finite_paths(payload.get("state_dict", {}), prefix="state_dict")
    if failures:
        preview = ", ".join(failures[:8])
        raise NonFiniteActiveTrainingError(f"Refusing to save non-finite checkpoint payload: {preview}")


def _active_log_line(payload: dict[str, Any]) -> str:
    ordered_keys = [
        "mode",
        "epoch",
        "global_step",
        "epoch_step",
        "epoch_steps_total",
        "loss_total",
        "loss_backbone_total",
        "loss_local_total",
        "lr",
        "step_time_sec",
        "step_time_running_avg_sec",
        "elapsed_sec",
        "eta_sec",
        "local_refine_sec",
        "local_reference_sec",
        "local_graph_sec",
        "epoch_train_sec",
        "eval_sec",
        "best_updated",
        "checkpoint_path",
        "reason",
        "metric",
        "best_metric",
        "wall_time_sec",
        "final_checkpoint_path",
        "best_checkpoint_path",
    ]
    parts: list[str] = []
    for key in ordered_keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    for key, value in payload.items():
        if key in ordered_keys:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    return "[active-train] " + " ".join(parts)


def _emit_active_log(metrics_log_path: Path, payload: dict[str, Any]) -> None:
    _append_jsonl(metrics_log_path, payload)
    print(_active_log_line(payload), flush=True)


def _backward_active_loss(
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss: torch.Tensor,
) -> bool:
    optimizer.zero_grad(set_to_none=True)
    if not bool(loss.requires_grad):
        return False
    scaler.scale(loss).backward()
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    if model is not None:
        grad_failures = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if not bool(torch.isfinite(param.grad.detach()).all().item()):
                grad_failures.append(name)
        if grad_failures:
            preview = ", ".join(grad_failures[:8])
            raise NonFiniteActiveTrainingError(f"Non-finite gradients detected after backward: {preview}")
    scaler.step(optimizer)
    scaler.update()
    optimizer_failures = _collect_non_finite_paths(optimizer.state_dict(), prefix="optimizer_state_dict")
    scaler_failures = _collect_non_finite_paths(scaler.state_dict(), prefix="scaler_state_dict")
    if optimizer_failures or scaler_failures:
        preview = ", ".join((optimizer_failures + scaler_failures)[:8])
        raise NonFiniteActiveTrainingError(f"Non-finite optimizer or scaler state detected after step: {preview}")
    return True


def _move_active_tensor_to_device(tensor: Any, device: torch.device, *, non_blocking: bool) -> Any:
    return tensor.to(device, non_blocking=non_blocking)


def _build_loader(
    *,
    dataset_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    include_depth: bool,
    train: bool,
    use_cuda: bool,
) -> DataLoader:
    dataset = BaselineInstanceDataset(
        dataset_root=dataset_root,
        split=split,
        image_size=image_size,
        include_depth=include_depth,
        include_annotations=True,
        include_instance_map=True,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": max(int(batch_size), 1),
        "shuffle": bool(train),
        "num_workers": int(num_workers),
        "collate_fn": lambda batch: batch,
        "pin_memory": bool(use_cuda),
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def _build_pixel_mask(pixel_values: torch.Tensor) -> torch.Tensor:
    return torch.ones(
        (int(pixel_values.shape[0]), int(pixel_values.shape[-2]), int(pixel_values.shape[-1])),
        dtype=torch.long,
        device=pixel_values.device,
    )


def _build_label_targets(
    samples: list[dict[str, Any]],
    *,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mask_labels = []
    class_labels = []
    for sample in samples:
        masks = sample.get("masks")
        labels = sample.get("labels")
        if masks is None or labels is None:
            mask_labels.append(torch.zeros((0, 1, 1), dtype=torch.float32, device=device))
            class_labels.append(torch.zeros((0,), dtype=torch.long, device=device))
            continue
        mask_labels.append(_move_active_tensor_to_device(masks.float(), device, non_blocking=non_blocking))
        class_labels.append(_move_active_tensor_to_device(labels.long(), device, non_blocking=non_blocking))
    return mask_labels, class_labels


def _upscale_mask_logits(mask_logits: torch.Tensor, *, image_shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        mask_logits.unsqueeze(0),
        size=(int(image_shape[0]), int(image_shape[1])),
        mode="bilinear",
        align_corners=False,
    )[0]


def _query_instances_from_outputs(
    *,
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    image_shape: tuple[int, int],
    score_threshold: float,
    mask_threshold: float,
) -> list[dict[str, Any]]:
    class_prob = torch.softmax(class_logits.float(), dim=-1)
    if int(class_prob.shape[-1]) < 2:
        return []
    fg_prob = class_prob[:, :-1]
    component_class_index = max(0, min(1, int(fg_prob.shape[1]) - 1))
    scores, class_ids = fg_prob.max(dim=-1)
    upsampled_mask_logits = _upscale_mask_logits(mask_logits, image_shape=image_shape)
    mask_probs = torch.sigmoid(upsampled_mask_logits)
    rows: list[dict[str, Any]] = []
    for query_index in range(int(mask_probs.shape[0])):
        predicted_class = int(class_ids[query_index].item())
        score = float(fg_prob[query_index, component_class_index].item())
        if predicted_class != component_class_index or score < float(score_threshold):
            continue
        category_id = 1
        binary = mask_probs[query_index] >= float(mask_threshold)
        if int(binary.sum().item()) <= 0:
            continue
        rows.append(
            {
                "query_index": int(query_index),
                "score": score,
                "category_id": category_id,
                "mask_probs": mask_probs[query_index],
                "binary_mask": binary.float(),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def _mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_binary = pred_mask > 0.5
    gt_binary = gt_mask > 0.5
    intersection = float((pred_binary & gt_binary).sum().item())
    union = float((pred_binary | gt_binary).sum().item())
    if union <= 0.0:
        return 0.0
    return intersection / union


def _match_query_predictions_to_gt(
    *,
    predictions: list[dict[str, Any]],
    gt_masks: torch.Tensor,
) -> list[tuple[int, int, float]]:
    if not predictions or int(gt_masks.shape[0]) == 0:
        return []
    cost = np.ones((len(predictions), int(gt_masks.shape[0])), dtype=np.float32)
    for pred_index, prediction in enumerate(predictions):
        for gt_index in range(int(gt_masks.shape[0])):
            cost[pred_index, gt_index] = 1.0 - float(
                _mask_iou(prediction["binary_mask"], gt_masks[gt_index])
            )
    pred_indices, gt_indices = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        iou = 1.0 - float(cost[pred_index, gt_index])
        if iou <= 0.0:
            continue
        matches.append((int(pred_index), int(gt_index), float(iou)))
    return matches


def _mask_iou_matrix(pred_masks: torch.Tensor, gt_masks: torch.Tensor, *, mask_threshold: float) -> torch.Tensor:
    if pred_masks.numel() == 0 or gt_masks.numel() == 0:
        return pred_masks.new_zeros((int(pred_masks.shape[0]), int(gt_masks.shape[0])))
    pred_binary = (pred_masks >= float(mask_threshold)).reshape(int(pred_masks.shape[0]), -1).float()
    gt_binary = (gt_masks >= 0.5).reshape(int(gt_masks.shape[0]), -1).float()
    intersections = pred_binary @ gt_binary.t()
    pred_area = pred_binary.sum(dim=1, keepdim=True)
    gt_area = gt_binary.sum(dim=1).unsqueeze(0)
    unions = (pred_area + gt_area - intersections).clamp_min(1.0)
    return intersections / unions


def _match_predicted_queries_to_instances(
    *,
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    image_shape: tuple[int, int],
    mask_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    if int(gt_masks.shape[0]) == 0:
        return []
    mask_probs = torch.sigmoid(_upscale_mask_logits(mask_logits, image_shape=image_shape))
    iou = _mask_iou_matrix(mask_probs, gt_masks.float().to(mask_probs.device), mask_threshold=float(mask_threshold))
    if iou.numel() == 0:
        return []
    query_indices, gt_indices = linear_sum_assignment((1.0 - iou).detach().cpu().numpy())
    class_prob = torch.softmax(class_logits.float(), dim=-1)
    fg_prob = class_prob[:, :-1] if int(class_prob.shape[-1]) >= 2 else class_prob
    component_class_index = 0 if int(fg_prob.shape[-1]) <= 1 else min(1, int(fg_prob.shape[-1]) - 1)
    query_scores = fg_prob[:, component_class_index] if fg_prob.numel() > 0 else mask_probs.new_ones((int(mask_probs.shape[0]),))
    matches: list[dict[str, Any]] = []
    for query_index, gt_index in zip(query_indices.tolist(), gt_indices.tolist()):
        match_iou = float(iou[int(query_index), int(gt_index)].item())
        if match_iou <= 0.0:
            continue
        matches.append(
            {
                "query_index": int(query_index),
                "gt_index": int(gt_index),
                "iou": match_iou,
                "score": float(query_scores[int(query_index)].item()),
                "mask_probs": mask_probs[int(query_index)],
                "binary_mask": (mask_probs[int(query_index)] >= float(mask_threshold)).float(),
            }
        )
    matches.sort(key=lambda item: item["iou"], reverse=True)
    return matches


def _uses_baseline_decode(variant_name: str) -> bool:
    variant_spec = get_active_variant_spec(variant_name)
    return not bool(variant_spec.use_local_refine)


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    sx = float(target_shape[1]) / float(max(source_shape[1], 1))
    sy = float(target_shape[0]) / float(max(source_shape[0], 1))
    x, y, w, h = bbox
    tx = int(round(float(x) * sx))
    ty = int(round(float(y) * sy))
    tw = max(1, int(round(float(w) * sx)))
    th = max(1, int(round(float(h) * sy)))
    tx = min(max(tx, 0), max(target_shape[1] - 1, 0))
    ty = min(max(ty, 0), max(target_shape[0] - 1, 0))
    tw = min(tw, max(target_shape[1] - tx, 1))
    th = min(th, max(target_shape[0] - ty, 1))
    return (tx, ty, tw, th)


def _prepare_reference_tensors(
    *,
    sample: dict[str, Any],
    source: PrototypeBankSource | None,
    crop_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if source is None:
        return None, None, None
    bank = source.load_for_query(str(sample["file_name"]))
    return _reference_tensors_from_bank(bank=bank, crop_size=crop_size, device=device)


def _reference_tensors_from_bank(
    *,
    bank: Any,
    crop_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_ref_depth = prepare_reference_depth(
        depth=bank.depths.float(),
        mask=bank.masks.float(),
    )
    return (
        F.interpolate(bank.images.float().to(device), size=(crop_size, crop_size), mode="bilinear", align_corners=False).unsqueeze(0),
        F.interpolate(normalized_ref_depth.to(device), size=(crop_size, crop_size), mode="bilinear", align_corners=False).unsqueeze(0),
        F.interpolate(bank.masks.float().to(device), size=(crop_size, crop_size), mode="nearest").unsqueeze(0),
    )


def _reference_match_aux_examples(
    *,
    file_name: str,
    source: PrototypeBankSource | None,
) -> list[tuple[PrototypeBank, float]]:
    if source is None or source.is_single_bank:
        return []
    positive_part_key = extract_query_part_key(str(file_name), source.available_parts)
    negative_candidates = [
        part_key for part_key in source.available_parts
        if str(part_key) != str(positive_part_key)
    ]
    if not negative_candidates:
        return []
    negative_bank = source.load_for_part(random.choice(negative_candidates))
    return [
        (source.load_for_part(positive_part_key), 1.0),
        (negative_bank, 0.0),
    ]


def _reference_match_examples(
    *,
    sample: dict[str, Any],
    source: PrototypeBankSource | None,
    crop_size: int,
    device: torch.device,
) -> list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], float]]:
    if source is None:
        return []
    positive_bank = source.load_for_query(str(sample["file_name"]))
    examples = [
        (_reference_tensors_from_bank(bank=positive_bank, crop_size=crop_size, device=device), 1.0)
    ]
    if source.is_single_bank:
        return examples
    for bank, target in _reference_match_aux_examples(
        file_name=str(sample["file_name"]),
        source=source,
    ):
        if float(target) <= 0.0:
            examples.append(
                (_reference_tensors_from_bank(bank=bank, crop_size=crop_size, device=device), target)
            )
            break
    return examples


def _connected_components(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask_u8, dtype=np.int32)
    return labels.astype(np.int32)


def _build_local_graph_inputs(
    *,
    component_map: np.ndarray,
    feature_crop: torch.Tensor,
    mask_prob_crop: torch.Tensor,
    depth_crop: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = [int(x) for x in np.unique(component_map).tolist() if int(x) > 0]
    if len(labels) <= 1:
        return (
            torch.zeros((0, feature_crop.shape[0] + 4), dtype=feature_crop.dtype, device=feature_crop.device),
            torch.zeros((2, 0), dtype=torch.long, device=feature_crop.device),
            torch.zeros((0, 4), dtype=feature_crop.dtype, device=feature_crop.device),
        )
    node_features = []
    geometry_rows: dict[int, tuple[float, float, float, float]] = {}
    height, width = component_map.shape
    depth_map = None if depth_crop is None else depth_crop[0]
    for label in labels:
        mask_np = component_map == int(label)
        mask_t = torch.from_numpy(mask_np).to(feature_crop.device)
        denom = mask_t.sum().clamp_min(1).float()
        pooled = (feature_crop * mask_t.unsqueeze(0)).sum(dim=(1, 2)) / denom
        ys, xs = np.nonzero(mask_np)
        centroid_x = float(xs.mean()) / float(max(width, 1))
        centroid_y = float(ys.mean()) / float(max(height, 1))
        area_ratio = float(mask_np.mean())
        mean_prob = float(mask_prob_crop[mask_t].mean().item()) if bool(mask_t.any()) else 0.0
        node_features.append(torch.cat([pooled, feature_crop.new_tensor([area_ratio, centroid_x, centroid_y, mean_prob])], dim=0))
        depth_mean = float(depth_map[mask_t].mean().item()) if depth_map is not None and bool(mask_t.any()) else 0.0
        geometry_rows[int(label)] = (centroid_x, centroid_y, area_ratio, depth_mean)
    edge_index = []
    edge_features = []
    for src_index, src_label in enumerate(labels):
        for dst_index, dst_label in enumerate(labels[src_index + 1 :], start=src_index + 1):
            sx, sy, sa, sd = geometry_rows[int(src_label)]
            dx, dy, da, dd = geometry_rows[int(dst_label)]
            edge_index.append([src_index, dst_index])
            edge_features.append(
                feature_crop.new_tensor(
                    [
                        float(math.hypot(sx - dx, sy - dy)),
                        abs(float(sa - da)),
                        abs(float(sd - dd)),
                        abs(float(node_features[src_index][-1].item()) - float(node_features[dst_index][-1].item())),
                    ]
                )
            )
    return (
        torch.stack(node_features, dim=0),
        torch.tensor(edge_index, dtype=torch.long, device=feature_crop.device).t().contiguous(),
        torch.stack(edge_features, dim=0),
    )


def _graph_rescue_edge_targets(
    *,
    component_map: np.ndarray,
    instance_mask_crops: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = [int(x) for x in np.unique(component_map).tolist() if int(x) > 0]
    if len(labels) <= 1 or edge_index.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.float32, device=edge_index.device)
        return empty, torch.zeros((0,), dtype=torch.bool, device=edge_index.device)
    owners: dict[int, int] = {}
    instance_masks = instance_mask_crops.float()
    for label in labels:
        component_mask = torch.from_numpy(component_map == int(label)).to(instance_masks.device)
        overlap_scores = []
        for instance_index in range(int(instance_masks.shape[0])):
            overlap_scores.append(float((instance_masks[instance_index] * component_mask.float()).sum().item()))
        best_overlap = max(overlap_scores, default=0.0)
        owners[int(label)] = 0 if best_overlap <= 0.0 else int(np.argmax(overlap_scores)) + 1
    targets: list[float] = []
    valid_mask: list[bool] = []
    for src_index, dst_index in edge_index.t().tolist():
        src_owner = owners[labels[int(src_index)]]
        dst_owner = owners[labels[int(dst_index)]]
        if src_owner == 0 and dst_owner == 0:
            targets.append(0.0)
            valid_mask.append(False)
            continue
        targets.append(1.0 if src_owner > 0 and src_owner == dst_owner else 0.0)
        valid_mask.append(True)
    if not targets:
        empty = torch.zeros((0,), dtype=torch.float32, device=edge_index.device)
        return empty, torch.zeros((0,), dtype=torch.bool, device=edge_index.device)
    return (
        torch.tensor(targets, dtype=torch.float32, device=edge_index.device),
        torch.tensor(valid_mask, dtype=torch.bool, device=edge_index.device),
    )


def _merge_local_components(
    *,
    component_map: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    labels = [int(x) for x in np.unique(component_map).tolist() if int(x) > 0]
    if len(labels) <= 1 or edge_index.numel() == 0:
        return component_map
    parent = {label: label for label in labels}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for (src_index, dst_index), score in zip(edge_index.t().tolist(), edge_scores.tolist()):
        if float(score) >= float(threshold):
            union(labels[int(src_index)], labels[int(dst_index)])
    remapped = np.zeros_like(component_map, dtype=np.int32)
    root_to_new: dict[int, int] = {}
    next_label = 1
    for label in labels:
        root = find(label)
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        remapped[component_map == int(label)] = root_to_new[root]
    return remapped


def _apply_local_rescue(
    *,
    model: ActiveInstanceModel,
    variant_name: str,
    sample: dict[str, Any],
    full_input: torch.Tensor,
    feature_map: torch.Tensor,
    predictions: list[dict[str, Any]],
    crop_size: int,
    crop_pad: int,
    mask_threshold: float,
    boundary_band_width: int,
    prototype_source: PrototypeBankSource | None,
) -> tuple[list[dict[str, Any]], int, int]:
    variant_spec = get_active_variant_spec(variant_name)
    if not variant_spec.use_local_refine or model.refiner is None or not predictions:
        return predictions, 0, 0
    binary_masks = torch.stack([row["binary_mask"] for row in predictions], dim=0)
    mask_probs = torch.stack([row["mask_probs"] for row in predictions], dim=0)
    scores = torch.tensor([float(row["score"]) for row in predictions], dtype=torch.float32, device=mask_probs.device)
    selected = select_refinement_instances(
        mask_probs=mask_probs,
        binary_masks=binary_masks,
        instance_scores=scores,
        boundary_band_width=int(boundary_band_width),
    )
    if not selected:
        return predictions, 0, 0
    image_shape = (int(full_input.shape[-2]), int(full_input.shape[-1]))
    feature_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
    refinement_invocations = 0
    graph_invocations = 0
    updated = list(predictions)
    for index in selected:
        row = dict(updated[int(index)])
        bbox = expand_bbox(
            bbox=mask_bbox(row["binary_mask"]),
            image_shape=image_shape,
            pad=int(crop_pad),
        )
        feature_bbox = _scale_bbox(bbox, source_shape=image_shape, target_shape=feature_shape)
        query_crop = crop_and_resize(full_input, bbox=bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
        coarse_mask_crop = crop_and_resize(row["mask_probs"].unsqueeze(0), bbox=bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
        projected_feature_map = _project_local_features_float32(model, feature_map.unsqueeze(0))[0]
        feature_crop = crop_and_resize(projected_feature_map, bbox=feature_bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
        reference_rgb, reference_depth, reference_mask = _prepare_reference_tensors(
            sample=sample,
            source=prototype_source if variant_spec.use_reference_rescue else None,
            crop_size=int(crop_size),
            device=full_input.device,
        )
        refined = _run_local_refiner_float32(
            model=model,
            query_crop=query_crop,
            coarse_mask_prob=coarse_mask_crop,
            feature_crop=feature_crop,
            reference_rgb=reference_rgb,
            reference_depth=reference_depth,
            reference_mask=reference_mask,
        )
        refined_prob = torch.sigmoid(refined["refined_mask_logits"][0, 0])
        refined_binary = (refined_prob >= float(mask_threshold)).float()
        refinement_invocations += 1
        if variant_spec.use_graph_rescue and model.graph_head is not None:
            component_map = _connected_components(refined_binary.detach().cpu().numpy())
            if int(component_map.max()) > 1:
                node_features, edge_index, edge_features = _build_local_graph_inputs(
                    component_map=component_map,
                    feature_crop=refined["crop_features"][0],
                    mask_prob_crop=refined_prob,
                    depth_crop=None if query_crop.shape[1] <= 3 else query_crop[0, 3:4],
                )
                if edge_index.numel() > 0:
                    edge_logits = model.graph_head(
                        node_features=node_features,
                        edge_index=edge_index,
                        edge_features=edge_features,
                    )
                    edge_scores = torch.sigmoid(edge_logits.detach().cpu())
                    merged = _merge_local_components(
                        component_map=component_map,
                        edge_index=edge_index.detach().cpu(),
                        edge_scores=edge_scores,
                        threshold=0.5,
                    )
                    refined_binary = torch.from_numpy((merged > 0).astype(np.float32)).to(refined_binary.device)
                    graph_invocations += 1
        row["mask_probs"] = paste_mask_from_crop(refined_prob, bbox=bbox, image_shape=image_shape)
        row["binary_mask"] = paste_mask_from_crop(refined_binary, bbox=bbox, image_shape=image_shape)
        updated[int(index)] = row
    return updated, refinement_invocations, graph_invocations


def _run_backbone(
    *,
    model: ActiveInstanceModel,
    pixel_values: torch.Tensor,
    pixel_mask: torch.Tensor,
    mask_labels: list[torch.Tensor] | None = None,
    class_labels: list[torch.Tensor] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "output_hidden_states": True,
    }
    if mask_labels is not None and class_labels is not None:
        kwargs["mask_labels"] = mask_labels
        kwargs["class_labels"] = class_labels
    return model.backbone(**kwargs)


def _active_benchmark_payload(variant_name: str, depth_mode: str) -> dict[str, Any]:
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


def _graph_rescue_training_loss(
    *,
    graph_head: nn.Module,
    crop_features: torch.Tensor,
    coarse_mask_prob: torch.Tensor,
    depth_crop: torch.Tensor | None,
    instance_mask_crops: torch.Tensor,
) -> torch.Tensor:
    coarse_prob = coarse_mask_prob.detach().float()
    component_map = _connected_components((coarse_prob >= 0.5).cpu().numpy())
    if int(component_map.max()) <= 1:
        return crop_features.sum() * 0.0
    node_features, edge_index, edge_features = _build_local_graph_inputs(
        component_map=component_map,
        feature_crop=crop_features,
        mask_prob_crop=coarse_prob,
        depth_crop=depth_crop,
    )
    if edge_index.numel() == 0:
        return crop_features.sum() * 0.0
    edge_targets, valid_edge_mask = _graph_rescue_edge_targets(
        component_map=component_map,
        instance_mask_crops=instance_mask_crops,
        edge_index=edge_index,
    )
    if edge_targets.numel() == 0 or not bool(valid_edge_mask.any()):
        return crop_features.sum() * 0.0
    edge_logits = graph_head(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )
    return F.binary_cross_entropy_with_logits(edge_logits[valid_edge_mask], edge_targets[valid_edge_mask])


def _evaluate_active(
    *,
    model: ActiveInstanceModel,
    loader: DataLoader,
    device: torch.device,
    variant_name: str,
    prototype_source: PrototypeBankSource | None,
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_spec = get_active_variant_spec(variant_name)
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
            images = _move_active_tensor_to_device(
                torch.stack([sample["image"].float() for sample in samples], dim=0),
                device,
                non_blocking=non_blocking,
            )
            depths = None
            if str(depth_mode) != "rgb":
                depths = _move_active_tensor_to_device(
                    torch.stack([sample["depth"].float() for sample in samples], dim=0),
                    device,
                    non_blocking=non_blocking,
                )
            pixel_values = prepare_active_input_batch(images=images, depths=depths, depth_mode=depth_mode)
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
                            "category_id": 1,
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
                        prototype_source=prototype_source,
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
                        category_id=1,
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


def _expand_reference_batch(
    reference_tensors: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    *,
    batch_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    rgb, depth, mask = reference_tensors
    if rgb is None or depth is None or mask is None:
        return None, None, None
    reference_batch_size = int(rgb.shape[0])
    if reference_batch_size not in {1, int(batch_size)}:
        raise ValueError(
            f"Reference batch size must be 1 or match query batch size, got {reference_batch_size} vs {int(batch_size)}"
        )
    return rgb, depth, mask


def _project_local_features_float32(model: ActiveInstanceModel, feature_map: torch.Tensor) -> torch.Tensor:
    with autocast(device_type=feature_map.device.type, enabled=False):
        projected = model.feature_proj(feature_map.float())
    _assert_finite_tensor("feature_proj", projected)
    return projected


def _run_local_refiner_float32(
    *,
    model: ActiveInstanceModel,
    query_crop: torch.Tensor,
    coarse_mask_prob: torch.Tensor,
    feature_crop: torch.Tensor,
    reference_rgb: torch.Tensor | None = None,
    reference_depth: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    with autocast(device_type=query_crop.device.type, enabled=False):
        refined = model.refiner(
            query_crop=query_crop.float(),
            coarse_mask_prob=coarse_mask_prob.float(),
            feature_crop=feature_crop.float(),
            reference_rgb=None if reference_rgb is None else reference_rgb.float(),
            reference_depth=None if reference_depth is None else reference_depth.float(),
            reference_mask=None if reference_mask is None else reference_mask.float(),
        )
    _assert_finite_tensor("refined_mask_logits", refined.get("refined_mask_logits"))
    _assert_finite_tensor("refined_boundary_logits", refined.get("refined_boundary_logits"))
    _assert_finite_tensor("crop_features", refined.get("crop_features"))
    _assert_finite_tensor("reference_match_logits", refined.get("reference_match_logits"))
    _assert_finite_tensor("reference_top_weights", refined.get("reference_top_weights"))
    return refined


def _train_local_modules_with_metrics(
    *,
    model: ActiveInstanceModel,
    samples: list[dict[str, Any]],
    pixel_values: torch.Tensor,
    backbone_outputs: Any,
    variant_name: str,
    prototype_source: PrototypeBankSource | None,
    crop_size: int,
    crop_pad: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    variant_spec = get_active_variant_spec(variant_name)
    if not variant_spec.use_local_refine or model.refiner is None:
        zero = pixel_values.sum() * 0.0
        return zero, {
            "loss_local_total": 0.0,
            "loss_local_mask": 0.0,
            "loss_local_boundary": 0.0,
            "loss_local_reference_positive": 0.0,
            "loss_local_reference_negative": 0.0,
            "loss_local_graph": 0.0,
            "local_refine_sec": 0.0,
            "local_reference_sec": 0.0,
            "local_graph_sec": 0.0,
        }
    feature_map = _project_local_features_float32(model, backbone_outputs.pixel_decoder_last_hidden_state)
    loss_sum = pixel_values.sum() * 0.0
    loss_count = 0
    component_totals = {
        "loss_local_mask": 0.0,
        "loss_local_boundary": 0.0,
        "loss_local_reference_positive": 0.0,
        "loss_local_reference_negative": 0.0,
        "loss_local_graph": 0.0,
        "local_refine_sec": 0.0,
        "local_reference_sec": 0.0,
        "local_graph_sec": 0.0,
    }
    for sample_index, sample in enumerate(samples):
        masks = sample.get("masks")
        if masks is None:
            continue
        gt_masks = masks.float().to(pixel_values.device)
        image_shape = (int(sample["image"].shape[-2]), int(sample["image"].shape[-1]))
        feature_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
        predictions = _query_instances_from_outputs(
            class_logits=backbone_outputs.class_queries_logits[sample_index].detach(),
            mask_logits=backbone_outputs.masks_queries_logits[sample_index].detach(),
            image_shape=image_shape,
            score_threshold=0.0,
            mask_threshold=0.5,
        )
        matches = _match_query_predictions_to_gt(predictions=predictions, gt_masks=gt_masks)
        positive_reference = _prepare_reference_tensors(
            sample=sample,
            source=prototype_source if variant_spec.use_reference_rescue else None,
            crop_size=int(crop_size),
            device=pixel_values.device,
        )
        reference_examples = _reference_match_examples(
            sample=sample,
            source=prototype_source if variant_spec.use_reference_rescue else None,
            crop_size=int(crop_size),
            device=pixel_values.device,
        )
        if not matches:
            continue
        query_crops: list[torch.Tensor] = []
        feature_crops: list[torch.Tensor] = []
        coarse_masks: list[torch.Tensor] = []
        gt_crops: list[torch.Tensor] = []
        match_rows: list[dict[str, Any]] = []
        for prediction_index, gt_index, _iou in matches:
            prediction = predictions[int(prediction_index)]
            instance_mask = gt_masks[int(gt_index)]
            bbox = expand_bbox(
                bbox=mask_bbox(prediction["binary_mask"]),
                image_shape=image_shape,
                pad=int(crop_pad),
            )
            feature_bbox = _scale_bbox(bbox, source_shape=image_shape, target_shape=feature_shape)
            gt_crop = crop_and_resize(instance_mask.unsqueeze(0), bbox=bbox, output_size=int(crop_size), mode="nearest")[0]
            query_crop = crop_and_resize(pixel_values[sample_index], bbox=bbox, output_size=int(crop_size), mode="bilinear")
            feature_crop = crop_and_resize(feature_map[sample_index], bbox=feature_bbox, output_size=int(crop_size), mode="bilinear")
            coarse_mask = crop_and_resize(
                prediction["mask_probs"].unsqueeze(0),
                bbox=bbox,
                output_size=int(crop_size),
                mode="bilinear",
            )
            query_crops.append(query_crop)
            feature_crops.append(feature_crop)
            coarse_masks.append(coarse_mask)
            gt_crops.append(gt_crop)
            match_rows.append(
                {
                    "bbox": bbox,
                    "query_crop": query_crop,
                }
            )
        batch_size = len(match_rows)
        query_crop_batch = torch.stack(query_crops, dim=0)
        feature_crop_batch = torch.stack(feature_crops, dim=0)
        coarse_mask_batch = torch.stack(coarse_masks, dim=0)
        gt_crop_batch = torch.stack(gt_crops, dim=0)
        gt_boundary_batch = torch.stack(
            [boundary_target_from_mask(gt_crop) for gt_crop in gt_crop_batch],
            dim=0,
        )
        reference_rgb, reference_depth, reference_mask = _expand_reference_batch(
            positive_reference,
            batch_size=batch_size,
        )
        refine_start = time.perf_counter()
        refined = _run_local_refiner_float32(
            model=model,
            query_crop=query_crop_batch,
            coarse_mask_prob=coarse_mask_batch,
            feature_crop=feature_crop_batch,
            reference_rgb=reference_rgb,
            reference_depth=reference_depth,
            reference_mask=reference_mask,
        )
        component_totals["local_refine_sec"] += float(time.perf_counter() - refine_start)
        loss_mask = F.binary_cross_entropy_with_logits(refined["refined_mask_logits"][:, 0], gt_crop_batch)
        loss_boundary = F.binary_cross_entropy_with_logits(refined["refined_boundary_logits"][:, 0], gt_boundary_batch)
        _assert_finite_tensor("loss_local_mask", loss_mask)
        _assert_finite_tensor("loss_local_boundary", loss_boundary)
        sample_loss_sum = (loss_mask + 0.5 * loss_boundary) * float(batch_size)
        component_totals["loss_local_mask"] += float(loss_mask.detach().cpu()) * float(batch_size)
        component_totals["loss_local_boundary"] += float((0.5 * loss_boundary).detach().cpu()) * float(batch_size)
        if variant_spec.use_reference_rescue and refined["reference_match_logits"] is not None and len(reference_examples) > 1:
            reference_start = time.perf_counter()
            positive_target = torch.ones_like(refined["reference_match_logits"])
            positive_loss = 0.05 * F.binary_cross_entropy_with_logits(
                refined["reference_match_logits"],
                positive_target,
            )
            _assert_finite_tensor("loss_local_reference_positive", positive_loss)
            negative_rgb, negative_depth, negative_mask = _expand_reference_batch(
                reference_examples[1][0],
                batch_size=batch_size,
            )
            negative_refined = _run_local_refiner_float32(
                model=model,
                query_crop=query_crop_batch,
                coarse_mask_prob=coarse_mask_batch,
                feature_crop=feature_crop_batch,
                reference_rgb=negative_rgb,
                reference_depth=negative_depth,
                reference_mask=negative_mask,
            )
            negative_target = torch.zeros_like(negative_refined["reference_match_logits"])
            negative_loss = 0.05 * F.binary_cross_entropy_with_logits(
                negative_refined["reference_match_logits"],
                negative_target,
            )
            _assert_finite_tensor("loss_local_reference_negative", negative_loss)
            component_totals["local_reference_sec"] += float(time.perf_counter() - reference_start)
            sample_loss_sum = sample_loss_sum + (positive_loss + negative_loss) * float(batch_size)
            component_totals["loss_local_reference_positive"] += float(positive_loss.detach().cpu()) * float(batch_size)
            component_totals["loss_local_reference_negative"] += float(negative_loss.detach().cpu()) * float(batch_size)
        if variant_spec.use_graph_rescue and model.graph_head is not None:
            graph_start = time.perf_counter()
            graph_losses: list[torch.Tensor] = []
            for match_index, match in enumerate(match_rows):
                gt_instance_crops = torch.stack(
                    [
                        crop_and_resize(mask.unsqueeze(0), bbox=match["bbox"], output_size=int(crop_size), mode="nearest")[0]
                        for mask in gt_masks
                    ],
                    dim=0,
                )
                graph_losses.append(
                    0.1 * _graph_rescue_training_loss(
                        graph_head=model.graph_head,
                        crop_features=refined["crop_features"][match_index],
                        coarse_mask_prob=coarse_mask_batch[match_index, 0],
                        depth_crop=None if query_crop_batch.shape[1] <= 3 else query_crop_batch[match_index, 3:4],
                        instance_mask_crops=gt_instance_crops,
                    )
                )
            component_totals["local_graph_sec"] += float(time.perf_counter() - graph_start)
            if graph_losses:
                graph_loss_sum = torch.stack(graph_losses).sum()
                _assert_finite_tensor("loss_local_graph", graph_loss_sum)
                sample_loss_sum = sample_loss_sum + graph_loss_sum
                component_totals["loss_local_graph"] += float(graph_loss_sum.detach().cpu())
        loss_sum = loss_sum + sample_loss_sum
        loss_count += int(batch_size)
    if loss_count <= 0:
        zero = pixel_values.sum() * 0.0
        return zero, {
            "loss_local_total": 0.0,
            **component_totals,
        }
    local_loss = loss_sum / float(loss_count)
    _assert_finite_tensor("loss_local_total", local_loss)
    return local_loss, {
        "loss_local_total": float(local_loss.detach().cpu()),
        **{
            key: float(value / loss_count)
            for key, value in component_totals.items()
            if key.startswith("loss_")
        },
        "local_refine_sec": float(component_totals["local_refine_sec"]),
        "local_reference_sec": float(component_totals["local_reference_sec"]),
        "local_graph_sec": float(component_totals["local_graph_sec"]),
    }


def _train_local_modules(
    *,
    model: ActiveInstanceModel,
    samples: list[dict[str, Any]],
    pixel_values: torch.Tensor,
    backbone_outputs: Any,
    variant_name: str,
    prototype_source: PrototypeBankSource | None,
    crop_size: int,
    crop_pad: int,
) -> torch.Tensor:
    return _train_local_modules_with_metrics(
        model=model,
        samples=samples,
        pixel_values=pixel_values,
        backbone_outputs=backbone_outputs,
        variant_name=variant_name,
        prototype_source=prototype_source,
        crop_size=crop_size,
        crop_pad=crop_pad,
    )[0]


def train_active(args: argparse.Namespace) -> None:
    payload = _model_payload(args)
    if bool(args.dry_run):
        print(json.dumps(payload, ensure_ascii=False))
        return
    variant_spec = get_active_variant_spec(args.variant)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_stage_lock(output_dir)
    include_depth = str(args.depth_mode) != "rgb"
    last_finite_step = 0
    last_finite_checkpoint = ""
    try:
        train_loader = _build_loader(
            dataset_root=str(args.dataset_root),
            split="train",
            image_size=int(args.image_size),
            batch_size=int(args.batch),
            num_workers=int(args.num_workers),
            include_depth=include_depth,
            train=True,
            use_cuda=bool(device.type == "cuda"),
        )
        val_loader = _build_loader(
            dataset_root=str(args.dataset_root),
            split="val",
            image_size=int(args.image_size),
            batch_size=1,
            num_workers=int(args.num_workers),
            include_depth=include_depth,
            train=False,
            use_cuda=bool(device.type == "cuda"),
        )
        model = _build_active_model(args).to(device)
        _configure_model_for_stage(model, args)
        prototype_source = None
        if variant_spec.requires_prototype_root:
            prototype_source = PrototypeBankSource(
                root=Path(str(args.prototype_root)).resolve(),
                image_size=int(args.crop_size),
                contract_mode="compat",
                max_views=int(args.reference_max_views),
                view_sampler=str(args.reference_view_sampler),
            )
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
        scaler = GradScaler(enabled=bool(device.type == "cuda"))
        ann_file = Path(args.dataset_root).resolve() / "annotations" / "instances_val.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        params_trainable = sum(int(param.numel()) for param in trainable_params)
        (output_dir / "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
        metrics_log_path = output_dir / "metrics_log.jsonl"
        if metrics_log_path.exists():
            metrics_log_path.unlink()
        resume_last_ckpt = output_dir / "resume_last.pth"
        best_ap = float("-inf")
        best_ckpt = output_dir / "model_best.pth"
        start = time.perf_counter()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        step_count = 0
        completed_epoch = 0
        eval_interval = max(int(getattr(args, "eval_every_epochs", 1)), 0)
        resume_save_every_epochs = max(
            int(getattr(args, "resume_save_every_epochs", MODEL_DEFAULTS["resume_save_every_epochs"])),
            1,
        )
        log_every_steps = max(int(getattr(args, "log_every_steps", MODEL_DEFAULTS["log_every_steps"])), 1)
        epoch_steps_total = len(train_loader)
        planned_total_steps = int(epoch_steps_total * int(args.epochs))
        if int(args.max_train_steps) > 0:
            planned_total_steps = min(planned_total_steps, int(args.max_train_steps))
        running_step_time_total = 0.0
        non_blocking = bool(device.type == "cuda")
        resume_guard_payload = None
        if str(getattr(args, "resume_checkpoint", "")).strip():
            resume_guard_payload = _validate_resume_checkpoint_allowed(Path(str(args.resume_checkpoint)).resolve())
        _write_run_state(
            output_dir,
            status="running",
            allow_resume=bool(resume_guard_payload is not None),
            failure_reason=None,
            last_finite_step=0,
            last_finite_checkpoint="" if resume_guard_payload is None else str(Path(str(args.resume_checkpoint)).resolve()),
        )
        if str(getattr(args, "resume_checkpoint", "")).strip():
            completed_epoch, step_count, best_ap, running_step_time_total = _load_resume_payload(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                allow_unsafe_resume=bool(getattr(args, "allow_unsafe_resume", False)),
            )
            last_finite_step = int(step_count)
            last_finite_checkpoint = str(Path(str(args.resume_checkpoint)).resolve())
            _write_run_state(
                output_dir,
                status="running",
                allow_resume=True,
                failure_reason=None,
                last_finite_step=last_finite_step,
                last_finite_checkpoint=last_finite_checkpoint,
            )
            _emit_active_log(
                metrics_log_path,
                {
                    "mode": "run_resume",
                    "epoch": int(completed_epoch),
                    "global_step": int(step_count),
                    "checkpoint_path": str(Path(str(args.resume_checkpoint)).resolve()),
                    "best_metric": None if not math.isfinite(best_ap) else float(best_ap),
                },
            )
        last_epoch = int(completed_epoch)
        for epoch_index in range(int(completed_epoch), int(args.epochs)):
            model.train()
            epoch_train_start = time.perf_counter()
            for epoch_step, samples in enumerate(train_loader, start=1):
                step_start = time.perf_counter()
                images = _move_active_tensor_to_device(
                    torch.stack([sample["image"].float() for sample in samples], dim=0),
                    device,
                    non_blocking=non_blocking,
                )
                depths = None
                if include_depth:
                    depths = _move_active_tensor_to_device(
                        torch.stack([sample["depth"].float() for sample in samples], dim=0),
                        device,
                        non_blocking=non_blocking,
                    )
                pixel_values = prepare_active_input_batch(images=images, depths=depths, depth_mode=str(args.depth_mode))
                pixel_mask = _build_pixel_mask(pixel_values)
                mask_labels, class_labels = _build_label_targets(samples, device=device, non_blocking=non_blocking)
                with autocast(device_type=device.type, enabled=bool(device.type == "cuda")):
                    outputs = _run_backbone(
                        model=model,
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        mask_labels=mask_labels,
                        class_labels=class_labels,
                    )
                    backbone_loss = outputs.loss
                    if backbone_loss is None:
                        backbone_loss = pixel_values.sum() * 0.0
                local_loss, local_metrics = _train_local_modules_with_metrics(
                    model=model,
                    samples=samples,
                    pixel_values=pixel_values,
                    backbone_outputs=outputs,
                    variant_name=variant_spec.name,
                    prototype_source=prototype_source,
                    crop_size=int(args.crop_size),
                    crop_pad=int(args.crop_pad),
                )
                loss = backbone_loss + local_loss
                loss_dict = getattr(outputs, "loss_dict", None)
                non_finite_scalars = _non_finite_scalar_names(
                    {
                        "loss_total": loss,
                        "loss_backbone_total": backbone_loss,
                        "loss_local_total": local_metrics.get("loss_local_total", 0.0),
                        **({f"loss_backbone_{key}": value for key, value in loss_dict.items()} if isinstance(loss_dict, dict) else {}),
                    }
                )
                if non_finite_scalars:
                    raise NonFiniteActiveTrainingError(
                        "Non-finite training scalars detected: " + ", ".join(non_finite_scalars)
                    )
                _backward_active_loss(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    loss=loss,
                )
                step_count += 1
                last_finite_step = int(step_count)
                step_time_sec = float(time.perf_counter() - step_start)
                running_step_time_total += step_time_sec
                running_avg_step_time_sec = float(running_step_time_total / max(step_count, 1))
                elapsed_sec = float(time.perf_counter() - start)
                remaining_steps = max(int(planned_total_steps) - int(step_count), 0)
                eta_sec = float(running_avg_step_time_sec * remaining_steps)
                if (
                    step_count == 1
                    or step_count % log_every_steps == 0
                    or step_count >= planned_total_steps
                    or (int(args.max_train_steps) > 0 and step_count >= int(args.max_train_steps))
                ):
                    row = {
                        "mode": "train_step",
                        "epoch": int(epoch_index + 1),
                        "global_step": int(step_count),
                        "epoch_step": int(epoch_step),
                        "epoch_steps_total": int(epoch_steps_total),
                        "loss_total": float(loss.detach().cpu()),
                        "loss_backbone_total": float(backbone_loss.detach().cpu()),
                        "loss_local_total": float(local_metrics.get("loss_local_total", 0.0)),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "step_time_sec": step_time_sec,
                        "step_time_running_avg_sec": running_avg_step_time_sec,
                        "elapsed_sec": elapsed_sec,
                        "eta_sec": eta_sec,
                    }
                    if isinstance(loss_dict, dict):
                        for key, value in loss_dict.items():
                            row[f"loss_backbone_{key}"] = float(value.detach().cpu())
                    row.update(local_metrics)
                    _emit_active_log(metrics_log_path, row)
                if int(args.max_train_steps) > 0 and step_count >= int(args.max_train_steps):
                    break
            epoch_train_sec = float(time.perf_counter() - epoch_train_start)
            _emit_active_log(
                metrics_log_path,
                {
                    "mode": "epoch_train",
                    "epoch": int(epoch_index + 1),
                    "epoch_train_sec": epoch_train_sec,
                    "global_step": int(step_count),
                    "best_metric": None if not math.isfinite(best_ap) else float(best_ap),
                },
            )
            last_epoch = int(epoch_index + 1)
            if (epoch_index + 1) % resume_save_every_epochs == 0:
                resume_payload = _resume_payload(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    completed_epoch=int(epoch_index + 1),
                    global_step=int(step_count),
                    best_metric=float(best_ap),
                    running_step_time_total=float(running_step_time_total),
                )
                _validate_resume_payload_finite(resume_payload)
                _save_torch_payload(resume_last_ckpt, resume_payload)
                _save_json_payload(_resume_metadata_path(resume_last_ckpt), _resume_metadata_payload())
                last_finite_checkpoint = str(resume_last_ckpt.resolve())
                _write_run_state(
                    output_dir,
                    status="running",
                    allow_resume=True,
                    failure_reason=None,
                    last_finite_step=last_finite_step,
                    last_finite_checkpoint=last_finite_checkpoint,
                )
            stopping_early = int(args.max_train_steps) > 0 and step_count >= int(args.max_train_steps)
            should_eval = False
            if eval_interval > 0:
                should_eval = stopping_early or (
                    (epoch_index + 1) % eval_interval == 0
                    and (epoch_index + 1) < int(args.epochs)
                )
            if should_eval:
                eval_start = time.perf_counter()
                metrics, speed = _evaluate_active(
                    model=model,
                    loader=val_loader,
                    device=device,
                    variant_name=variant_spec.name,
                    prototype_source=prototype_source,
                    ann_file=ann_file,
                    output_dir=output_dir,
                    score_threshold=float(args.score_threshold),
                    mask_threshold=float(args.mask_threshold),
                    crop_size=int(args.crop_size),
                    crop_pad=int(args.crop_pad),
                    boundary_band_width=int(args.boundary_band_width),
                    max_images=int(args.max_val_images),
                    save_raw=False,
                    depth_mode=str(args.depth_mode),
                )
                eval_sec = float(time.perf_counter() - eval_start)
                segm_ap = float(metrics.get("segm/AP", 0.0))
                best_updated = bool(segm_ap >= best_ap)
                if best_updated:
                    best_ap = segm_ap
                    best_payload = _checkpoint_payload(model, args)
                    _validate_checkpoint_payload_finite(best_payload)
                    _save_torch_payload(best_ckpt, best_payload)
                    _emit_active_log(
                        metrics_log_path,
                        {
                            "mode": "checkpoint",
                            "epoch": int(epoch_index + 1),
                            "checkpoint_path": str(best_ckpt.resolve()),
                            "reason": "best",
                            "metric": segm_ap,
                        },
                    )
                eval_row = {
                    "mode": "epoch_eval",
                    "epoch": int(epoch_index + 1),
                    "eval_sec": eval_sec,
                    "best_updated": best_updated,
                    "metric": segm_ap,
                    "best_metric": float(best_ap),
                }
                eval_row.update(metrics)
                _emit_active_log(metrics_log_path, eval_row)
            if stopping_early:
                break
        final_ckpt = output_dir / "model_final.pth"
        final_payload = _checkpoint_payload(model, args)
        _validate_checkpoint_payload_finite(final_payload)
        _save_torch_payload(final_ckpt, final_payload)
        last_finite_checkpoint = str(final_ckpt.resolve())
        _emit_active_log(
            metrics_log_path,
            {
                "mode": "checkpoint",
                "epoch": int(last_epoch),
                "checkpoint_path": str(final_ckpt.resolve()),
                "reason": "final",
            },
        )
        final_eval_start = time.perf_counter()
        metrics, speed = _evaluate_active(
            model=model,
            loader=val_loader,
            device=device,
            variant_name=variant_spec.name,
            prototype_source=prototype_source,
            ann_file=ann_file,
            output_dir=output_dir,
            score_threshold=float(args.score_threshold),
            mask_threshold=float(args.mask_threshold),
            crop_size=int(args.crop_size),
            crop_pad=int(args.crop_pad),
            boundary_band_width=int(args.boundary_band_width),
            max_images=int(args.max_val_images),
            save_raw=False,
            depth_mode=str(args.depth_mode),
        )
        final_eval_sec = float(time.perf_counter() - final_eval_start)
        final_ap = float(metrics.get("segm/AP", 0.0))
        final_best_updated = bool(final_ap >= best_ap)
        if final_best_updated:
            best_ap = final_ap
            best_payload = _checkpoint_payload(model, args)
            _validate_checkpoint_payload_finite(best_payload)
            _save_torch_payload(best_ckpt, best_payload)
            _emit_active_log(
                metrics_log_path,
                {
                    "mode": "checkpoint",
                    "epoch": int(last_epoch),
                    "checkpoint_path": str(best_ckpt.resolve()),
                    "reason": "best",
                    "metric": final_ap,
                },
            )
        final_eval_row = {
            "mode": "epoch_eval",
            "epoch": int(last_epoch),
            "eval_sec": final_eval_sec,
            "best_updated": final_best_updated,
            "metric": final_ap,
            "best_metric": float(best_ap),
        }
        final_eval_row.update(metrics)
        _emit_active_log(metrics_log_path, final_eval_row)
        peak_memory_mb = 0.0
        if device.type == "cuda" and torch.cuda.is_available():
            peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        wall_time_sec = int(time.perf_counter() - start)
        (output_dir / "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
        (output_dir / "wall_time_sec.txt").write_text(f"{wall_time_sec}\n", encoding="utf-8")
        summary = build_run_summary_payload(
            model="mask2former",
            variant=variant_spec.name,
            modality=str(args.depth_mode),
            artifact_root=output_dir,
            metrics=metrics,
            inference_speed=speed,
            checkpoint=final_ckpt,
            dataset_root=str(Path(args.dataset_root).resolve()),
            params_trainable=params_trainable,
            training_peak_memory_mb=peak_memory_mb,
            wall_time_sec=wall_time_sec,
            benchmark=_active_benchmark_payload(variant_spec.name, str(args.depth_mode)),
            decode_config={
                "score_threshold": float(args.score_threshold),
                "mask_threshold": float(args.mask_threshold),
            },
        )
        write_json(output_dir / "run_summary.json", summary)
        full_training_steps = int(epoch_steps_total * int(args.epochs))
        completed_full_training = int(step_count) >= int(full_training_steps)
        resumable_checkpoint = str(resume_last_ckpt.resolve()) if resume_last_ckpt.exists() else ""
        _write_run_state(
            output_dir,
            status="success" if completed_full_training else "running",
            allow_resume=bool((not completed_full_training) and resumable_checkpoint),
            failure_reason=None,
            last_finite_step=last_finite_step,
            last_finite_checkpoint=last_finite_checkpoint if completed_full_training else resumable_checkpoint,
        )
        _emit_active_log(
            metrics_log_path,
            {
                "mode": "run_final",
                "wall_time_sec": float(wall_time_sec),
                "best_metric": float(best_ap),
                "final_checkpoint_path": str(final_ckpt.resolve()),
                "best_checkpoint_path": str(best_ckpt.resolve()),
            },
        )
    except Exception as exc:
        _write_run_state(
            output_dir,
            status="failed",
            allow_resume=False,
            failure_reason=str(exc),
            last_finite_step=last_finite_step,
            last_finite_checkpoint=last_finite_checkpoint,
        )
        raise
    finally:
        _release_stage_lock(lock_path)


def eval_active(args: argparse.Namespace) -> None:
    payload = _model_payload(args)
    if bool(args.dry_run):
        print(json.dumps(payload, ensure_ascii=False))
        return
    variant_spec = get_active_variant_spec(args.variant)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir_arg = getattr(args, "checkpoint_dir", "")
    checkpoint_dir = output_dir if checkpoint_dir_arg in ("", None) else Path(str(checkpoint_dir_arg)).resolve()
    checkpoint_path = _resolve_checkpoint_path(checkpoint_dir, str(args.checkpoint))
    if checkpoint_path.parent.resolve() == output_dir.resolve():
        raise ValueError("legacy eval/infer requires --checkpoint-dir to differ from --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_active_model(args).to(device)
    checkpoint_payload = torch.load(str(checkpoint_path), map_location=device)
    _validate_runtime_checkpoint_variant(
        requested_variant=variant_spec.name,
        run_variant=getattr(args, "_run_metadata_variant", None),
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=str(checkpoint_path),
        context="eval",
    )
    state_dict = _extract_state_dict(checkpoint_payload, prefix_backbone=True)
    _load_module_state_dict(
        model,
        state_dict,
        allow_partial=bool(getattr(args, "allow_partial_checkpoint_load", False)),
        context=f"eval checkpoint {checkpoint_path}",
    )
    prototype_source = None
    if variant_spec.requires_prototype_root:
        prototype_source = PrototypeBankSource(
            root=Path(str(args.prototype_root)).resolve(),
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
    ann_file = Path(args.dataset_root).resolve() / "annotations" / f"instances_{args.split}.json"
    metrics, speed = _evaluate_active(
        model=model,
        loader=loader,
        device=device,
        variant_name=variant_spec.name,
        prototype_source=prototype_source,
        ann_file=ann_file,
        output_dir=output_dir,
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        crop_size=int(args.crop_size),
        crop_pad=int(args.crop_pad),
        boundary_band_width=int(args.boundary_band_width),
        max_images=int(args.max_images),
        save_raw=False,
        depth_mode=str(args.depth_mode),
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
        benchmark=_active_benchmark_payload(variant_spec.name, str(args.depth_mode)),
        decode_config={
            "score_threshold": float(args.score_threshold),
            "mask_threshold": float(args.mask_threshold),
        },
    )
    write_json(output_dir / "run_summary.json", summary)


def infer_active(args: argparse.Namespace) -> None:
    payload = _model_payload(args)
    if bool(args.dry_run):
        print(json.dumps(payload, ensure_ascii=False))
        return
    variant_spec = get_active_variant_spec(args.variant)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir_arg = getattr(args, "checkpoint_dir", "")
    checkpoint_dir = output_dir if checkpoint_dir_arg in ("", None) else Path(str(checkpoint_dir_arg)).resolve()
    checkpoint_path = _resolve_checkpoint_path(checkpoint_dir, str(args.checkpoint))
    if checkpoint_path.parent.resolve() == output_dir.resolve():
        raise ValueError("legacy eval/infer requires --checkpoint-dir to differ from --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_active_model(args).to(device)
    checkpoint_payload = torch.load(str(checkpoint_path), map_location=device)
    _validate_runtime_checkpoint_variant(
        requested_variant=variant_spec.name,
        run_variant=getattr(args, "_run_metadata_variant", None),
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=str(checkpoint_path),
        context="infer",
    )
    state_dict = _extract_state_dict(checkpoint_payload, prefix_backbone=True)
    _load_module_state_dict(
        model,
        state_dict,
        allow_partial=bool(getattr(args, "allow_partial_checkpoint_load", False)),
        context=f"infer checkpoint {args.checkpoint}",
    )
    prototype_source = None
    if variant_spec.requires_prototype_root:
        prototype_source = PrototypeBankSource(
            root=Path(str(args.prototype_root)).resolve(),
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
    ann_file = Path(args.dataset_root).resolve() / "annotations" / f"instances_{args.split}.json"
    metrics, speed = _evaluate_active(
        model=model,
        loader=loader,
        device=device,
        variant_name=variant_spec.name,
        prototype_source=prototype_source,
        ann_file=ann_file,
        output_dir=output_dir,
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        crop_size=int(args.crop_size),
        crop_pad=int(args.crop_pad),
        boundary_band_width=int(args.boundary_band_width),
        max_images=int(args.max_images),
        save_raw=True,
        depth_mode=str(args.depth_mode),
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
        benchmark=_active_benchmark_payload(variant_spec.name, str(args.depth_mode)),
        decode_config={
            "score_threshold": float(args.score_threshold),
            "mask_threshold": float(args.mask_threshold),
        },
    )
    write_json(output_dir / "run_summary.json", summary)
