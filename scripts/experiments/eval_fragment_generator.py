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

from baseline.fragment_generator.dataset import FragmentGeneratorCacheDataset  # noqa: E402
from baseline.fragment_generator.eval import evaluate_fragment_generator  # noqa: E402
from baseline.fragment_generator.model import LocalFragmentGenerator  # noqa: E402
from gisec.engine.runtime import build_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-root", required=True)
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
    dataset = FragmentGeneratorCacheDataset(cache_root=str(Path(args.cache_root).resolve()), split=str(args.split))
    probe = dataset[0]
    model = LocalFragmentGenerator(
        rgb_channels=int(probe["rgb_crop"].shape[0]),
        feature_channels=int(probe["pixel_feature_crop"].shape[0]),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        max_fragments=int(model_cfg.get("max_fragments", 6)),
    ).to(device)
    model_dir = Path(args.model_dir).resolve()
    checkpoint_path = model_dir / "model_best.pth"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "model_final.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    summary = evaluate_fragment_generator(
        cache_root=str(Path(args.cache_root).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        split=str(args.split),
        device=device,
        model=model,
        batch_size=1,
        num_workers=int(common.get("num_workers", 0)),
        export_predictions=True,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
