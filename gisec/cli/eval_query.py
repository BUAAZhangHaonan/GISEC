from __future__ import annotations

import sys
from pathlib import Path

from gisec.cli.train_query import _print_payload, _requires_prototype_source, _validate_alpha_execution_surface, build_parser
from gisec.train.train_query import run_uq_eval


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(argv, mode="eval")
    args = parser.parse_args(argv)
    model_id = str(args.variant)
    _validate_alpha_execution_surface(parser, model_id=model_id)
    if _requires_prototype_source(model_id) and not args.prototype_root:
        parser.error("--prototype-root is required for reference query variants")
    if args.dry_run:
        _print_payload(args, mode="eval")
        return
    if not args.dataset_root:
        parser.error("--dataset-root is required unless --dry-run is set")
    if not args.output_dir:
        parser.error("--output-dir is required unless --dry-run is set")
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --dry-run is set")
    if not Path(args.checkpoint).exists():
        parser.error("checkpoint file does not exist")
    try:
        run_uq_eval(
            dataset_root=Path(args.dataset_root),
            output_dir=Path(args.output_dir),
            model_id=model_id,
            checkpoint=Path(args.checkpoint),
            prototype_root=Path(args.prototype_root) if args.prototype_root else None,
            device=str(args.device),
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_val_images=int(args.max_val_images),
            min_area=int(args.min_area),
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main(sys.argv[1:])
