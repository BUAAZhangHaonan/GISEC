from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gisec.config.variants import get_gisec_variant_spec, gisec_variant_names


GISEC_DEPTH_MODES = ("rgb", "rgbd_concat", "rgbd_concat_valid_mask")
_DEFAULT_VARIANT = "base_rgb_1024"
_DEFAULT_PRETRAINED_MODEL = "facebook/mask2former-swin-tiny-coco-instance"


def _existing(path_str: str | None) -> Path | None:
    if path_str in (None, ""):
        return None
    path = Path(str(path_str)).resolve()
    return path if path.exists() else None


def _summary_variant(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    variant = payload.get("variant")
    return None if variant in {None, ""} else str(variant)


def _flag_values(argv: list[str], flag: str) -> list[str]:
    values: list[str] = []
    prefix = f"{flag}="
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif token.startswith(prefix):
            values.append(token[len(prefix) :])
    return values


def explicit_cli_variant(argv: list[str]) -> str | None:
    variant = None
    for value in _flag_values(argv, "--variant"):
        variant = value
    return None if variant in {None, ""} else str(variant)


def resolve_run_directory_variant(argv: list[str]) -> str | None:
    checkpoint_path = None
    output_dir = None
    checkpoint_values = _flag_values(argv, "--checkpoint")
    output_dir_values = _flag_values(argv, "--output-dir")
    if checkpoint_values:
        checkpoint_path = checkpoint_values[-1]
    if output_dir_values:
        output_dir = output_dir_values[-1]
    checkpoint = _existing(checkpoint_path)
    output_root = _existing(output_dir)
    candidate_roots = []
    if checkpoint is not None:
        candidate_roots.append(checkpoint.parent)
    if output_root is not None:
        candidate_roots.append(output_root)
    for root in candidate_roots:
        summary_variant = _summary_variant(root / "run_summary.json")
        if summary_variant in set(gisec_variant_names()):
            return summary_variant
    return None


def _resolved_gisec_variant_default(argv: list[str] | None) -> str:
    cli_variant = explicit_cli_variant(list(argv or []))
    if cli_variant in set(gisec_variant_names()):
        return str(cli_variant)
    run_variant = resolve_run_directory_variant(list(argv or []))
    if run_variant in set(gisec_variant_names()):
        return str(run_variant)
    return _DEFAULT_VARIANT


def _validate_variant_source_consistency(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None,
) -> None:
    cli_variant = explicit_cli_variant(list(argv or []))
    run_variant = resolve_run_directory_variant(list(argv or []))
    if run_variant in {None, ""}:
        return
    if cli_variant not in {None, ""} and str(cli_variant) != str(run_variant):
        parser.error(
            f"--variant {cli_variant} conflicts with run metadata variant {run_variant}"
        )
    if str(args.variant) != str(run_variant):
        parser.error(
            f"parsed GISEC variant {args.variant} does not match run metadata variant {run_variant}"
        )


def _annotate_variant_sources(args: argparse.Namespace, argv: list[str] | None) -> None:
    setattr(args, "_explicit_variant", explicit_cli_variant(list(argv or [])))
    setattr(args, "_run_metadata_variant", resolve_run_directory_variant(list(argv or [])))


def _common_parser(*, mode: str, argv: list[str] | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--reference-root", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument(
        "--variant",
        choices=list(gisec_variant_names()),
        default=_resolved_gisec_variant_default(argv),
    )
    parser.add_argument("--depth-mode", choices=list(GISEC_DEPTH_MODES), default="")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model-name", type=str, default=_DEFAULT_PRETRAINED_MODEL)
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
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
        parser.add_argument("--graph-loss-weight", type=float, default=0.1)
        parser.add_argument("--reference-match-loss-weight", type=float, default=0.05)
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--learning-rate", type=float, default=1.0e-4)
        parser.add_argument("--weight-decay", type=float, default=1.0e-4)
        parser.add_argument("--max-train-steps", type=int, default=0)
        parser.add_argument("--max-val-images", type=int, default=0)
        parser.add_argument("--eval-every-epochs", type=int, default=1)
        parser.add_argument("--log-every-steps", type=int, default=50)
        parser.add_argument("--resume-save-every-epochs", type=int, default=1)
    else:
        parser.add_argument("--checkpoint", type=str, default="")
        parser.add_argument("--split", choices=["train", "val"], default="val")
        parser.add_argument("--max-images", type=int, default=0)
    return parser


def _validate_required_args(parser: argparse.ArgumentParser, args: argparse.Namespace, required: list[str]) -> None:
    missing = [name for name in required if getattr(args, name, None) in (None, "")]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))


def _validate_variant_requirements(parser: argparse.ArgumentParser, args: argparse.Namespace, *, is_eval: bool) -> None:
    variant_spec = get_gisec_variant_spec(args.variant)
    depth_mode = str(getattr(args, "depth_mode", "") or variant_spec.depth_mode)
    if variant_spec.depth_mode == "rgb":
        if depth_mode != "rgb":
            parser.error(f"--depth-mode {depth_mode} is not allowed for GISEC variant {variant_spec.name}")
    elif depth_mode not in {"rgbd_concat", "rgbd_concat_valid_mask"}:
        parser.error(f"--depth-mode {depth_mode} is not allowed for GISEC variant {variant_spec.name}")
    args.depth_mode = depth_mode
    if variant_spec.requires_reference_root and getattr(args, "reference_root", "") in ("", None):
        parser.error(f"--reference-root is required for GISEC variant {variant_spec.name}")
    if (not is_eval) and variant_spec.use_local_refine and getattr(args, "init_checkpoint", "") in ("", None):
        parser.error(f"--init-checkpoint is required for GISEC variant {variant_spec.name}")
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
    variant_spec = get_gisec_variant_spec(args.variant)
    return {
        "variant": variant_spec.name,
        "depth_mode": str(args.depth_mode),
        "use_local_refine": variant_spec.use_local_refine,
        "use_reference_rescue": variant_spec.use_reference_rescue,
        "use_graph_rescue": variant_spec.use_graph_rescue,
        "requires_reference_root": variant_spec.requires_reference_root,
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
        "reference_root": str(getattr(args, "reference_root", "")),
        "init_checkpoint": str(getattr(args, "init_checkpoint", "")),
        "resume_checkpoint": str(getattr(args, "resume_checkpoint", "")),
        "checkpoint": str(getattr(args, "checkpoint", "")),
        "split": str(getattr(args, "split", "val")),
        "max_train_steps": int(getattr(args, "max_train_steps", 0)),
        "max_val_images": int(getattr(args, "max_val_images", 0)),
        "max_images": int(getattr(args, "max_images", 0)),
        "epochs": int(getattr(args, "epochs", 0)),
        "log_every_steps": int(getattr(args, "log_every_steps", 50)),
        "learning_rate": float(getattr(args, "learning_rate", 0.0)),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "eval_every_epochs": int(getattr(args, "eval_every_epochs", 1)),
        "allow_partial_checkpoint_load": bool(getattr(args, "allow_partial_checkpoint_load", False)),
        "resume_save_every_epochs": int(getattr(args, "resume_save_every_epochs", 1)),
        "seed": int(getattr(args, "seed", 42)),
    }
