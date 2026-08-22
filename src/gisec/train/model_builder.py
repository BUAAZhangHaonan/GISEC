from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler

from gisec.backbones.mask2former.adapter import build_mask2former_model
from gisec.config.variants import get_gisec_variant_spec
from gisec.models.gisec_model import GISECModel
from gisec.train.args import model_payload


def resolve_checkpoint_path(checkpoint_dir: Path, checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute():
        return checkpoint_path.resolve()
    return (checkpoint_dir / checkpoint_path).resolve()


def extract_state_dict(
    payload: dict[str, Any], *, prefix_backbone: bool = False
) -> dict[str, Any]:
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = dict(payload["state_dict"])
    else:
        state_dict = dict(payload)
    if (
        prefix_backbone
        and state_dict
        and not any(key.startswith("backbone.") for key in state_dict)
    ):
        state_dict = {f"backbone.{key}": value for key, value in state_dict.items()}
    return state_dict


def _checkpoint_variant(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    candidates = [
        payload.get("variant"),
        payload.get("model", {}).get("variant")
        if isinstance(payload.get("model"), dict)
        else None,
    ]
    for candidate in candidates:
        if candidate not in {"", None}:
            return str(candidate)
    return None


def _partition_state_dict(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    compatible: dict[str, Any] = {}
    shape_mismatches: list[str] = []
    for key, value in source_state.items():
        if key not in target_state:
            continue
        target_value = target_state[key]
        if (
            hasattr(value, "shape")
            and hasattr(target_value, "shape")
            and tuple(value.shape) != tuple(target_value.shape)
        ):
            shape_mismatches.append(
                f"{key}: checkpoint {tuple(value.shape)} != "
                f"model {tuple(target_value.shape)}"
            )
            continue
        compatible[key] = value
    missing_keys = sorted(key for key in target_state if key not in source_state)
    unexpected_keys = sorted(key for key in source_state if key not in target_state)
    return compatible, missing_keys, unexpected_keys, shape_mismatches


def load_module_state_dict(
    module: nn.Module,
    source_state: dict[str, Any],
    *,
    allow_partial: bool,
    context: str,
) -> None:
    target_state = module.state_dict()
    compatible, missing_keys, unexpected_keys, shape_mismatches = _partition_state_dict(
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
    if allow_partial and (missing_keys or unexpected_keys or shape_mismatches):
        print(
            f"[gisec] partial checkpoint load ({context}): "
            f"missing={len(missing_keys)} {missing_keys}; "
            f"unexpected={len(unexpected_keys)} {unexpected_keys}; "
            f"shape_mismatch={len(shape_mismatches)} {shape_mismatches}",
            flush=True,
        )
    module.load_state_dict(
        compatible if allow_partial else source_state, strict=not allow_partial
    )


def _backbone_state_dict(source_state: dict[str, Any]) -> dict[str, Any]:
    backbone_state = {
        key[len("backbone.") :]: value
        for key, value in source_state.items()
        if key.startswith("backbone.")
    }
    return backbone_state or dict(source_state)


def validate_runtime_checkpoint_variant(
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
            f"{context} checkpoint {Path(checkpoint_path).resolve()} "
            f"declares variant {checkpoint_variant}, "
            f"but the requested GISEC variant is {requested_variant}."
        )
    if run_variant not in {None, ""} and str(checkpoint_variant) != str(run_variant):
        raise RuntimeError(
            f"{context} checkpoint {Path(checkpoint_path).resolve()} "
            f"declares variant {checkpoint_variant}, "
            f"but run metadata resolves to {run_variant}."
        )


def validate_checkpoint_model_args(
    *,
    payload: dict[str, Any],
    args: argparse.Namespace,
    context: str,
) -> None:
    stored = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(stored, dict):
        return
    for key in ("image_size", "crop_size", "num_queries"):
        if key not in stored:
            continue
        if int(stored[key]) != int(getattr(args, key)):
            raise RuntimeError(
                f"{context} checkpoint was trained with {key}={int(stored[key])}, "
                f"but the current run was given {key}={int(getattr(args, key))}; "
                "shape-related model arguments must match the checkpoint."
            )


def _resolve_input_channels(depth_mode: str) -> int:
    if str(depth_mode) == "rgb":
        return 3
    if str(depth_mode) == "rgbd_concat":
        return 4
    raise ValueError(f"Unsupported GISEC depth_mode: {depth_mode}")


def build_gisec_model(args: argparse.Namespace) -> GISECModel:
    variant_spec = get_gisec_variant_spec(args.variant)
    depth_mode = str(getattr(args, "depth_mode", "") or variant_spec.depth_mode)
    input_channels = _resolve_input_channels(depth_mode)
    backbone = build_mask2former_model(
        image_size=int(args.image_size),
        pretrained_model_name=(
            None
            if str(args.pretrained_model_name).strip().lower() in {"", "none"}
            else str(args.pretrained_model_name)
        ),
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
    feature_channels = int(
        getattr(backbone.config, "hidden_dim", int(args.feature_size))
    )
    return GISECModel(
        backbone=backbone,
        feature_channels=feature_channels,
        input_channels=int(input_channels),
        use_local_refine=variant_spec.use_local_refine,
        use_reference_rescue=variant_spec.use_reference_rescue,
        use_graph_rescue=variant_spec.use_graph_rescue,
        refiner_hidden_dim=int(args.refiner_hidden_dim),
        graph_hidden_dim=int(args.graph_hidden_dim),
    )


def configure_model_for_stage(model: nn.Module, args: argparse.Namespace) -> None:
    variant_spec = get_gisec_variant_spec(args.variant)
    if not variant_spec.use_local_refine:
        return
    for param in model.backbone.parameters():
        param.requires_grad = False
    init_checkpoint_value = str(getattr(args, "init_checkpoint", "") or "").strip()
    if not init_checkpoint_value:
        # Resume runs receive every weight from --resume-checkpoint in
        # train_gisec; no init checkpoint is needed on top of it.
        return
    init_checkpoint = Path(init_checkpoint_value).resolve()
    if not init_checkpoint.exists():
        raise FileNotFoundError(init_checkpoint)
    init_payload = torch.load(
        str(init_checkpoint), map_location="cpu", weights_only=True
    )
    state_dict = extract_state_dict(init_payload, prefix_backbone=True)
    load_module_state_dict(
        model.backbone,
        _backbone_state_dict(state_dict),
        allow_partial=bool(getattr(args, "allow_partial_checkpoint_load", False)),
        context=f"init checkpoint {init_checkpoint}",
    )


def checkpoint_payload(model: nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "variant": str(args.variant),
        "depth_mode": str(args.depth_mode),
        "model": model_payload(args),
    }


def resume_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    completed_epoch: int,
    global_step: int,
    best_metric: float,
    running_step_time_total: float,
    elapsed_sec: float,
    peak_memory_mb: float,
) -> dict[str, Any]:
    return {
        **checkpoint_payload(model, args),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "completed_epoch": int(completed_epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "running_step_time_total": float(running_step_time_total),
        "elapsed_sec": float(elapsed_sec),
        "peak_memory_mb": float(peak_memory_mb),
    }


def load_resume_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
) -> tuple[int, int, float, float, float, float]:
    resume_checkpoint = Path(str(args.resume_checkpoint)).resolve()
    if not resume_checkpoint.exists():
        raise FileNotFoundError(resume_checkpoint)
    payload = torch.load(str(resume_checkpoint), map_location="cpu", weights_only=True)
    optimizer_state = payload.get("optimizer_state_dict")
    if (
        not isinstance(optimizer_state, dict)
        or "param_groups" not in optimizer_state
        or "completed_epoch" not in payload
    ):
        raise RuntimeError(
            f"resume checkpoint {resume_checkpoint} carries no training state "
            "(optimizer_state_dict / completed_epoch); weight-only payloads "
            "such as model_best.pth or model_final.pth belong on "
            "--init-checkpoint, while --resume-checkpoint needs the "
            "resume_last.pth a previous run wrote."
        )
    validate_runtime_checkpoint_variant(
        requested_variant=str(args.variant),
        run_variant=None,
        checkpoint_payload=payload,
        checkpoint_path=resume_checkpoint,
        context="resume",
    )
    validate_checkpoint_model_args(payload=payload, args=args, context="resume")
    load_module_state_dict(
        model,
        extract_state_dict(payload),
        allow_partial=False,
        context=f"resume checkpoint {resume_checkpoint}",
    )
    optimizer.load_state_dict(dict(optimizer_state))
    scaler_state = payload.get("scaler_state_dict")
    if isinstance(scaler_state, dict):
        scaler.load_state_dict(scaler_state)
    return (
        int(payload["completed_epoch"]),
        int(payload.get("global_step", 0)),
        float(payload.get("best_metric", float("-inf"))),
        float(payload.get("running_step_time_total", 0.0)),
        float(payload.get("elapsed_sec", 0.0)),
        float(payload.get("peak_memory_mb", 0.0)),
    )


def save_torch_payload(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def build_pixel_mask(pixel_values: torch.Tensor) -> torch.Tensor:
    return torch.ones(
        (
            int(pixel_values.shape[0]),
            int(pixel_values.shape[-2]),
            int(pixel_values.shape[-1]),
        ),
        dtype=torch.long,
        device=pixel_values.device,
    )


def run_backbone(
    *,
    model: GISECModel,
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
