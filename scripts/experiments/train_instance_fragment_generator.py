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

from baseline.instance_fragment_generator.train import train_instance_fragment_generator  # noqa: E402
from gisec.engine.runtime import build_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=None)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve(cli_value, config_value, default_value):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default_value


def _resolve_cache_root(cache_root: str | Path) -> Path:
    root = Path(cache_root).resolve()
    nested = root / "instance_fragment_cache_pred"
    return nested if nested.exists() else root


def _resolve_num_queries(*, cache_root: Path, split_names: list[str], cli_value: int | None, config_value: int | None) -> int:
    if cli_value is not None and int(cli_value) > 0:
        return int(cli_value)
    if config_value is not None and int(config_value) > 0:
        return int(config_value)
    maxima: list[int] = []
    for split in split_names:
        manifest_path = cache_root / str(split) / "manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            maxima.append(int(payload.get("raw_fragment_count_max", 0)))
    max_value = max(maxima) if maxima else 0
    if max_value <= 0:
        raise ValueError("num_queries could not be resolved from config or cache manifests")
    return int(max_value)


def main() -> None:
    args = parse_args()
    payload = _load_yaml(Path(args.config).resolve())
    common = dict(payload.get("common", {}))
    model_cfg = dict(payload.get("stage2_model", {}))
    train_cfg = dict(payload.get("train", {}))
    cache_root = _resolve_cache_root(args.cache_root)
    num_queries = _resolve_num_queries(
        cache_root=cache_root,
        split_names=[str(args.split)] + ([] if str(args.val_split).lower() in {"", "none"} else [str(args.val_split)]),
        cli_value=args.num_queries,
        config_value=model_cfg.get("num_queries"),
    )
    summary = train_instance_fragment_generator(
        cache_root=str(cache_root),
        dataset_root=str(Path(args.dataset_root).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        split=str(args.split),
        val_split=None if str(args.val_split).lower() in {"", "none"} else str(args.val_split),
        device=build_device(str(args.device or common.get("device", "cpu"))),
        epochs=int(_resolve(args.epochs, train_cfg.get("epochs"), 5)),
        batch_size=int(_resolve(args.batch_size, train_cfg.get("batch_size"), 4)),
        num_workers=int(_resolve(args.num_workers, common.get("num_workers"), 0)),
        max_train_steps=int(_resolve(args.max_train_steps, train_cfg.get("max_train_steps"), 0)),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        num_queries=int(num_queries),
        learning_rate=float(train_cfg.get("learning_rate", 1.0e-3)),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
