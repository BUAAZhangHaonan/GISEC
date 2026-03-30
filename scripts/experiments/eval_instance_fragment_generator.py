#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.instance_fragment_generator.eval import evaluate_instance_fragment_generator  # noqa: E402
from baseline.instance_fragment_generator.model import InstanceLocalFragmentGenerator  # noqa: E402
from gisec.engine.runtime import build_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def _load_yaml(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
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


def _resolve_run_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        checkpoint = run_dir / "model_final.pth" if args.checkpoint is None else Path(args.checkpoint).resolve()
        model_config = run_dir / "model_config.json" if args.model_config is None else Path(args.model_config).resolve()
        return checkpoint, model_config
    if args.checkpoint is None or args.model_config is None:
        raise ValueError("Provide --run-dir or both --checkpoint and --model-config")
    return Path(args.checkpoint).resolve(), Path(args.model_config).resolve()


def main() -> None:
    args = parse_args()
    config_payload = _load_yaml(None if args.config is None else Path(args.config).resolve())
    common = dict(config_payload.get("common", {}))
    checkpoint_path, model_config_path = _resolve_run_paths(args)
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = InstanceLocalFragmentGenerator(
        rgb_channels=int(model_config["rgb_channels"]),
        feature_channels=int(model_config["feature_channels"]),
        neighbor_channels=int(model_config.get("neighbor_channels", 1)),
        hidden_dim=int(model_config["hidden_dim"]),
        num_queries=int(model_config["num_queries"]),
    )
    model.load_state_dict(state_dict)
    summary = evaluate_instance_fragment_generator(
        cache_root=str(_resolve_cache_root(args.cache_root)),
        dataset_root=str(Path(args.dataset_root).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        split=str(args.split),
        device=build_device(str(args.device or common.get("device", "cpu"))),
        model=model,
        batch_size=int(_resolve(args.batch_size, model_config.get("batch_size"), 4)),
        num_workers=int(_resolve(args.num_workers, common.get("num_workers"), 0)),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
