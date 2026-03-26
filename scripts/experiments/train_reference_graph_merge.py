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

from baseline.reference_graph.train import train_reference_graph_merge  # noqa: E402
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
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--positive-edge-weight", type=float, default=None)
    parser.add_argument("--negative-edge-weight", type=float, default=None)
    return parser.parse_args()


def _load_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    config = _load_yaml(None if args.config is None else Path(args.config).resolve())
    train_cfg = dict(config.get("train", {}))
    model_cfg = dict(config.get("model", {}))
    common = dict(config.get("common", {}))
    output_dir = Path(args.output_dir).resolve()

    def _resolve_value(cli_value, config_value, default_value):
        return default_value if cli_value is None and config_value is None else cli_value if cli_value is not None else config_value

    train_reference_graph_merge(
        cache_root=str(Path(args.cache_root).resolve()),
        reference_root=str(Path(args.reference_root).resolve()),
        output_dir=str(output_dir),
        split=str(args.split),
        device=build_device(str(_resolve_value(args.device, common.get("device"), "cpu"))),
        epochs=int(_resolve_value(args.epochs, train_cfg.get("epochs"), 5)),
        batch_size=int(_resolve_value(args.batch_size, train_cfg.get("batch_size"), 8)),
        num_workers=int(_resolve_value(args.num_workers, train_cfg.get("num_workers"), 0)),
        reference_image_size=int(model_cfg.get("reference_image_size", 128)),
        reference_max_views=int(model_cfg.get("reference_max_views", 16)),
        reference_view_sampler=str(model_cfg.get("reference_view_sampler", "pose_farthest")),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        reference_hidden_dim=int(model_cfg.get("reference_hidden_dim", 32)),
        learning_rate=float(train_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
        max_train_steps=int(_resolve_value(args.max_train_steps, train_cfg.get("max_train_steps"), 0)),
        val_split=_resolve_value(args.val_split, train_cfg.get("val_split"), None),
        decision_threshold=float(_resolve_value(args.decision_threshold, train_cfg.get("decision_threshold"), 0.5)),
        positive_edge_weight=float(_resolve_value(args.positive_edge_weight, train_cfg.get("positive_edge_weight"), 1.0)),
        negative_edge_weight=float(_resolve_value(args.negative_edge_weight, train_cfg.get("negative_edge_weight"), 1.0)),
    )
    summary = json.loads((output_dir / "train_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
