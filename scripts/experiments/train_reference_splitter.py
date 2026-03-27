#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.reference_splitter.train import train_reference_splitter_alpha  # noqa: E402
from gisec.engine.runtime import build_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    return parser.parse_args()


def _load_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    config = _load_yaml(None if args.config is None else Path(args.config).resolve())
    common = dict(config.get("common", {}))
    train_cfg = dict(config.get("train", {}))
    model_cfg = dict(config.get("model", {}))

    def _resolve_value(cli_value, config_value, default_value):
        return default_value if cli_value is None and config_value is None else cli_value if cli_value is not None else config_value

    train_reference_splitter_alpha(
        cache_root=str(Path(args.cache_root).resolve()),
        reference_root=str(Path(args.reference_root).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        split=str(args.split),
        device=build_device(str(_resolve_value(args.device, common.get("device"), "cpu"))),
        epochs=int(_resolve_value(args.epochs, train_cfg.get("epochs"), 5)),
        batch_size=int(_resolve_value(args.batch_size, train_cfg.get("batch_size"), 8)),
        num_workers=int(_resolve_value(args.num_workers, train_cfg.get("num_workers"), 0)),
        roi_size=int(model_cfg.get("roi_size", 128)),
        reference_image_size=int(model_cfg.get("reference_image_size", 128)),
        slot_count=int(model_cfg.get("slot_count", 6)),
        learning_rate=float(train_cfg.get("learning_rate", 1.0e-3)),
        max_train_steps=int(_resolve_value(args.max_train_steps, train_cfg.get("max_train_steps"), 0)),
    )
    summary = json.loads((Path(args.output_dir).resolve() / "train_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
