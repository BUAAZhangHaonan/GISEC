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

from baseline.reference_graph.eval_pipeline import evaluate_reference_graph_merge  # noqa: E402
from gisec.engine.runtime import build_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def _load_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_value(cli_value, config_value, default_value):
    return default_value if cli_value is None and config_value is None else cli_value if cli_value is not None else config_value


def _resolve_checkpoint(model_dir: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg is not None:
        return Path(checkpoint_arg).resolve()
    for candidate in ["model_best.pth", "model_final.pth"]:
        path = model_dir / candidate
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"No checkpoint found under {model_dir}")


def _resolve_threshold(model_dir: Path, threshold_arg: float | None, train_cfg: dict) -> float:
    if threshold_arg is not None:
        return float(threshold_arg)
    for summary_name in ["train_summary.json", "val_summary.json"]:
        summary_path = model_dir / summary_name
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("best_threshold") is not None:
            return float(payload["best_threshold"])
        if payload.get("decision_threshold") is not None:
            return float(payload["decision_threshold"])
    return float(train_cfg.get("decision_threshold", 0.5))


def main() -> None:
    args = parse_args()
    config = _load_yaml(None if args.config is None else Path(args.config).resolve())
    train_cfg = dict(config.get("train", {}))
    model_cfg = dict(config.get("model", {}))
    common = dict(config.get("common", {}))
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    metrics, summary = evaluate_reference_graph_merge(
        cache_root=str(Path(args.cache_root).resolve()),
        reference_root=str(Path(args.reference_root).resolve()),
        dataset_root=str(Path(args.dataset_root).resolve()),
        output_dir=str(output_dir),
        split=str(args.split),
        device=build_device(str(_resolve_value(args.device, common.get("device"), "cpu"))),
        threshold=_resolve_threshold(model_dir, args.threshold, train_cfg),
        checkpoint_path=str(_resolve_checkpoint(model_dir, args.checkpoint)),
        batch_size=int(_resolve_value(args.batch_size, train_cfg.get("batch_size"), 8)),
        num_workers=int(_resolve_value(args.num_workers, train_cfg.get("num_workers"), 0)),
        reference_image_size=int(model_cfg.get("reference_image_size", 128)),
        reference_max_views=int(model_cfg.get("reference_max_views", 16)),
        reference_view_sampler=str(model_cfg.get("reference_view_sampler", "pose_farthest")),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        reference_hidden_dim=int(model_cfg.get("reference_hidden_dim", 32)),
    )
    print(json.dumps({"summary": summary, "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
