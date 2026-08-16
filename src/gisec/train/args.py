from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gisec.config.variants import get_gisec_variant_spec, gisec_variant_names


GISEC_DEPTH_MODES = ("rgb", "rgbd_concat")
# Standard COCO candidate protocol: one shared default for `gisec eval` and
# the trainer's epoch-val, so best-model selection matches eval exactly.
EVAL_SCORE_THRESHOLD_DEFAULT = 0.05
_DEFAULT_VARIANT = "base_rgb_1024"
_DEFAULT_PRETRAINED_MODEL = "facebook/mask2former-swin-tiny-coco-instance"


def _flag_value(argv: list[str], flag: str) -> str | None:
    """Last value of a repeated CLI flag, accepting ``--flag value`` and ``--flag=value``."""
    value = None
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith(f"{flag}="):
            value = token[len(flag) + 1:]
    return None if value in (None, "") else str(value)


def _run_metadata_variant(argv: list[str]) -> str | None:
    """Variant recorded in the run_summary.json next to --checkpoint / --output-dir."""
    candidate_roots: list[Path] = []
    checkpoint = _flag_value(argv, "--checkpoint")
    if checkpoint is not None:
        candidate_roots.append(Path(checkpoint).resolve().parent)
    output_dir = _flag_value(argv, "--output-dir")
    if output_dir is not None:
        candidate_roots.append(Path(output_dir).resolve())
    for root in candidate_roots:
        summary = root / "run_summary.json"
        if not summary.exists():
            continue
        try:
            variant = json.loads(summary.read_text(
                encoding="utf-8")).get("variant")
        except json.JSONDecodeError:
            print(f"[gisec] ignoring malformed run summary: {summary}",
                  flush=True)
            continue
        if variant in gisec_variant_names():
            return str(variant)
    return None


def _resolved_variant_default(argv: list[str]) -> str:
    """Variant comes from --variant, else from the run directory, else the default."""
    cli_variant = _flag_value(argv, "--variant")
    if cli_variant in gisec_variant_names():
        return str(cli_variant)
    run_variant = _run_metadata_variant(argv)
    if run_variant is not None:
        return run_variant
    return _DEFAULT_VARIANT


def _check_variant_consistency(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str],
) -> None:
    run_variant = _run_metadata_variant(argv)
    if run_variant is None:
        return
    setattr(args, "_run_metadata_variant", run_variant)
    if str(args.variant) != str(run_variant):
        parser.error(
            f"--variant {args.variant} conflicts with run metadata variant {run_variant}"
        )


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
        default=_resolved_variant_default(list(argv or [])),
    )
    parser.add_argument(
        "--depth-mode", choices=list(GISEC_DEPTH_MODES), default="")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model-name", type=str,
                        default=_DEFAULT_PRETRAINED_MODEL)
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
    parser.add_argument("--reference-view-sampler",
                        choices=["all", "uniform", "pose_farthest"], default="pose_farthest")
    parser.add_argument("--allow-partial-checkpoint-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    if mode == "train":
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
        parser.add_argument("--graph-loss-weight", type=float, default=0.1)
        parser.add_argument("--reference-match-loss-weight",
                            type=float, default=0.05)
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--learning-rate", type=float, default=1.0e-4)
        parser.add_argument("--weight-decay", type=float, default=1.0e-4)
        parser.add_argument("--max-train-steps", type=int, default=0)
        parser.add_argument("--max-val-images", type=int, default=0)
        parser.add_argument(
            "--eval-every-epochs",
            type=int,
            default=1,
            help="evaluate on the val split every N epochs; 0 disables epoch "
                 "evals so only the final epoch is evaluated",
        )
        parser.add_argument("--log-every-steps", type=int, default=50)
        parser.add_argument("--resume-save-every-epochs", type=int, default=1)
        parser.add_argument(
            "--eval-score-threshold",
            type=float,
            default=EVAL_SCORE_THRESHOLD_DEFAULT,
            help="score threshold for the epoch-val candidate set; the same "
                 "standard COCO protocol gisec eval uses, not the 0.5 of "
                 "--score-threshold",
        )
    else:
        parser.add_argument("--checkpoint", type=str, default="")
        parser.add_argument("--split", choices=["train", "val"], default="val")
        parser.add_argument("--max-images", type=int, default=0)
        parser.add_argument(
            "--eval-score-threshold",
            type=float,
            default=EVAL_SCORE_THRESHOLD_DEFAULT,
            help="score threshold for the eval candidate set (standard COCO "
                 "protocol); infer still saves predictions above "
                 "--score-threshold",
        )
    return parser


def _validate_required_args(parser: argparse.ArgumentParser, args: argparse.Namespace, required: list[str]) -> None:
    missing = [name for name in required if getattr(
        args, name, None) in (None, "")]
    if missing:
        parser.error("the following arguments are required: " +
                     ", ".join(f"--{name.replace('_', '-')}" for name in missing))


def _validate_variant_requirements(parser: argparse.ArgumentParser, args: argparse.Namespace, *, is_eval: bool) -> None:
    variant_spec = get_gisec_variant_spec(args.variant)
    depth_mode = str(getattr(args, "depth_mode", "")
                     or variant_spec.depth_mode)
    if variant_spec.depth_mode == "rgb":
        if depth_mode != "rgb":
            parser.error(
                f"--depth-mode {depth_mode} is not allowed for GISEC variant {variant_spec.name}")
    elif depth_mode != "rgbd_concat":
        parser.error(
            f"--depth-mode {depth_mode} is not allowed for GISEC variant {variant_spec.name}")
    args.depth_mode = depth_mode
    if variant_spec.requires_reference_root and getattr(args, "reference_root", "") in ("", None):
        parser.error(
            f"--reference-root is required for GISEC variant {variant_spec.name}")
    resuming = str(getattr(args, "resume_checkpoint", "") or "").strip() != ""
    if (
        (not is_eval)
        and variant_spec.use_local_refine
        and not resuming
        and getattr(args, "init_checkpoint", "") in ("", None)
    ):
        parser.error(
            f"--init-checkpoint is required for GISEC variant {variant_spec.name} "
            "when not resuming; --resume-checkpoint already carries all weights"
        )
    if is_eval and getattr(args, "checkpoint", "") in ("", None):
        parser.error("--checkpoint is required for eval/infer")


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser(mode="train", argv=argv)
    args = parser.parse_args(argv)
    if int(getattr(args, "eval_every_epochs", 1)) < 0:
        parser.error(
            "--eval-every-epochs must be >= 0 (0 evaluates only the final epoch)")
    _check_variant_consistency(parser, args, list(argv or []))
    _validate_required_args(parser, args, ["dataset_root", "output_dir"])
    _validate_variant_requirements(parser, args, is_eval=False)
    return args


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser(mode="eval", argv=argv)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    args = parser.parse_args(argv)
    _check_variant_consistency(parser, args, list(argv or []))
    _validate_required_args(parser, args, ["dataset_root", "output_dir"])
    _validate_variant_requirements(parser, args, is_eval=True)
    return args


def model_payload(args: argparse.Namespace) -> dict[str, Any]:
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
