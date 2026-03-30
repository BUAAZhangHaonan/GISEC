#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.fragment_generator.cache import build_fragment_generator_cache  # noqa: E402
from baseline.mask2former.adapter import (  # noqa: E402
    build_mask2former_model,
    build_mask2former_processor,
    outputs_to_instance_masks,
)
from gisec.engine.runtime import build_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _build_mask2former_from_checkpoint(
    *,
    checkpoint_payload: dict,
    image_size: int,
) -> torch.nn.Module:
    state_dict = checkpoint_payload["state_dict"] if isinstance(checkpoint_payload, dict) and "state_dict" in checkpoint_payload else checkpoint_payload
    checkpoint_config = dict(checkpoint_payload.get("config", {})) if isinstance(checkpoint_payload, dict) else {}
    backbone_config = dict(checkpoint_config.get("backbone_config", {}))
    model = build_mask2former_model(
        image_size=int(image_size),
        pretrained_model_name=str(checkpoint_config.get("_name_or_path") or checkpoint_config.get("pretrained_model_name") or "facebook/mask2former-swin-tiny-coco-instance"),
        input_channels=int(backbone_config.get("num_channels", 3)),
        hidden_dim=int(checkpoint_config.get("hidden_dim", 256)),
        feature_size=int(checkpoint_config.get("feature_size", 256)),
        mask_feature_size=int(checkpoint_config.get("mask_feature_size", 256)),
        encoder_layers=int(checkpoint_config.get("encoder_layers", 6)),
        decoder_layers=int(checkpoint_config.get("decoder_layers", 10)),
        num_attention_heads=int(checkpoint_config.get("num_attention_heads", 8)),
        num_queries=int(checkpoint_config.get("num_queries", 100)),
        train_num_points=int(checkpoint_config.get("train_num_points", 12544)),
    )
    model.load_state_dict(state_dict)
    return model


def _reduce_feature_channels(
    feature_map: torch.Tensor,
    *,
    out_channels: int,
) -> torch.Tensor:
    if int(feature_map.shape[1]) <= int(out_channels):
        return feature_map
    batch, channels, height, width = feature_map.shape
    target = int(out_channels)
    group_size = int((channels + target - 1) // target)
    padded = torch.zeros((batch, target * group_size, height, width), dtype=feature_map.dtype, device=feature_map.device)
    padded[:, :channels] = feature_map
    reduced = padded.view(batch, target, group_size, height, width).mean(dim=2)
    return reduced


def main() -> None:
    args = parse_args()
    payload = _load_yaml(Path(args.config).resolve())
    common = dict(payload.get("common", {}))
    cache_cfg = dict(payload.get("cache", {}))
    model_cfg = dict(payload.get("model", {}))
    device = build_device(str(args.device or common.get("device", "cpu")))
    image_size = int(common.get("image_size", 1024))
    feature_channels = int(cache_cfg.get("feature_channels", 32))
    processor = build_mask2former_processor()
    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location="cpu")
    model = _build_mask2former_from_checkpoint(checkpoint_payload=checkpoint, image_size=image_size).to(device)
    model.eval()

    def _infer_sample(sample: dict[str, object]) -> tuple[torch.Tensor, list, list[float]]:
        image = sample["image"].float()
        encoded = processor(images=[image], return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)
        pixel_mask = encoded["pixel_mask"].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, output_hidden_states=True)
        masks, scores = outputs_to_instance_masks(
            outputs,
            processor=processor,
            target_size=(int(image.shape[-2]), int(image.shape[-1])),
            score_threshold=float(model_cfg.get("score_threshold", 0.5)),
            mask_threshold=float(model_cfg.get("mask_threshold", 0.5)),
        )
        feature_map = outputs.pixel_decoder_hidden_states[-1]
        target_shape = (max(int(image.shape[-2]) // 4, 1), max(int(image.shape[-1]) // 4, 1))
        if tuple(int(v) for v in feature_map.shape[-2:]) != target_shape:
            feature_map = F.interpolate(feature_map, size=target_shape, mode="bilinear", align_corners=False)
        feature_map = _reduce_feature_channels(feature_map, out_channels=int(feature_channels))
        return feature_map.detach().cpu(), masks, scores

    manifest = build_fragment_generator_cache(
        dataset_root=str(Path(args.dataset_root).resolve()),
        output_root=str(Path(args.output_root).resolve()),
        split=str(args.split),
        image_size=int(image_size),
        crop_size=int(cache_cfg.get("crop_size", 256)),
        crop_pad=int(cache_cfg.get("crop_pad", 16)),
        max_fragments=int(cache_cfg.get("max_fragments", 6)),
        target_solidity=float(cache_cfg.get("target_solidity", 0.92)),
        infer_sample=_infer_sample,
        max_images=int(args.max_images),
        cache_workers=int(common.get("num_workers", 0)),
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
