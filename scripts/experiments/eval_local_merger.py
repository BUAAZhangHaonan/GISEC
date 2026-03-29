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

from baseline.local_merger.dataset import LocalMergerPredictionDataset  # noqa: E402
from baseline.local_merger.eval import evaluate_local_merger  # noqa: E402
from baseline.local_merger.model import LocalMergeEdgeScorer  # noqa: E402
from gisec.engine.runtime import build_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    payload = _load_yaml(Path(args.config).resolve())
    common = dict(payload.get("common", {}))
    model_cfg = dict(payload.get("model", {}))
    device = build_device(str(args.device or common.get("device", "cpu")))
    dataset = LocalMergerPredictionDataset(prediction_root=str(Path(args.prediction_root).resolve()), split=str(args.split))
    probe = dataset[0]
    model = LocalMergeEdgeScorer(
        node_dim=int(probe["node_features"].shape[1]),
        edge_dim=int(probe["edge_features"].shape[1]),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
    ).to(device)
    model_dir = Path(args.model_dir).resolve()
    checkpoint_path = model_dir / "model_best.pth"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "model_final.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    summary = evaluate_local_merger(
        prediction_root=str(Path(args.prediction_root).resolve()),
        dataset_root=str(Path(args.dataset_root).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        split=str(args.split),
        device=device,
        model=model,
        batch_size=1,
        num_workers=int(common.get("num_workers", 0)),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
