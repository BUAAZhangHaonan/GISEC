from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gisec.config.io import extract_argparse_defaults, load_yaml_config, merge_config_dicts
from gisec.config.v3_models import is_alpha_enabled_model_id
from gisec.train.train_v3 import run_uq_minibatch


def _config_defaults(argv: list[str] | None, mode: str) -> dict[str, object]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", action="append", default=[])
    known, _ = parser.parse_known_args(argv)
    if not known.config:
        return {}
    merged = merge_config_dicts(load_yaml_config(path) for path in known.config)
    return extract_argparse_defaults(merged, mode)


def build_parser(argv: list[str] | None = None, *, mode: str = "train") -> argparse.ArgumentParser:
    defaults = _config_defaults(argv, mode)
    parser = argparse.ArgumentParser(
        prog=f"python -m gisec.cli.{mode}",
        description=f"GISEC v3-alpha query-only object-first {mode} surface.",
    )
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-family", default="UQ")
    parser.add_argument("--model-scale", choices=("s", "m"), default="s")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--head-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--max-val-images", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--fg-loss-weight", type=float, default=1.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
    parser.add_argument("--core-loss-weight", type=float, default=4.0)
    parser.add_argument("--ownership-loss-weight", type=float, default=0.25)
    parser.add_argument("--ownership-warmup-steps", type=int, default=16)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(**defaults)
    return parser


def _print_payload(args: argparse.Namespace, *, mode: str) -> None:
    model_id = f"{args.model_family}-{args.model_scale}"
    payload = {
        "mode": mode,
        "dataset_root": args.dataset_root,
        "output_dir": args.output_dir,
        "model_id": model_id,
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "head_lr_multiplier": float(args.head_lr_multiplier),
        "max_train_steps": int(args.max_train_steps),
        "max_val_images": int(args.max_val_images),
        "min_area": int(args.min_area),
        "fg_loss_weight": float(args.fg_loss_weight),
        "boundary_loss_weight": float(args.boundary_loss_weight),
        "core_loss_weight": float(args.core_loss_weight),
        "ownership_loss_weight": float(args.ownership_loss_weight),
        "ownership_warmup_steps": int(args.ownership_warmup_steps),
        "checkpoint": args.checkpoint,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _validate_alpha_execution_surface(parser: argparse.ArgumentParser, *, model_id: str) -> None:
    if not is_alpha_enabled_model_id(model_id):
        parser.error(f"{model_id} is reserved or unavailable and is not enabled in current v3-alpha execution surface.")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(argv, mode="train")
    args = parser.parse_args(argv)
    model_id = f"{args.model_family}-{args.model_scale}"
    _validate_alpha_execution_surface(parser, model_id=model_id)
    if args.dry_run:
        _print_payload(args, mode="train")
        return
    if not args.dataset_root:
        parser.error("--dataset-root is required unless --dry-run is set")
    if not args.output_dir:
        parser.error("--output-dir is required unless --dry-run is set")
    run_uq_minibatch(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        model_id=model_id,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        device=str(args.device),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        lr=float(args.lr),
        head_lr_multiplier=float(args.head_lr_multiplier),
        max_train_steps=int(args.max_train_steps),
        max_val_images=int(args.max_val_images),
        min_area=int(args.min_area),
        fg_loss_weight=float(args.fg_loss_weight),
        boundary_loss_weight=float(args.boundary_loss_weight),
        core_loss_weight=float(args.core_loss_weight),
        ownership_loss_weight=float(args.ownership_loss_weight),
        ownership_warmup_steps=int(args.ownership_warmup_steps),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
