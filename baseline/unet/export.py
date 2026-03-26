from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common.dataset import BaselineInstanceDataset
from baseline.common.fragment_quality import (
    build_fragment_pair_records,
    build_fragment_records,
    summarize_fragment_quality,
)
from baseline.rgbd.fusion import prepare_unet_inputs
from baseline.unet.eval import _sigmoid_tensor, decode_instance_predictions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_unet_fragment_cache(
    *,
    model: torch.nn.Module,
    dataset_root: str,
    output_dir: str,
    image_size: int,
    device: torch.device,
    split: str,
    input_mode: str = "rgb",
    threshold: float = 0.18,
    center_threshold: float = 0.03,
    min_area: int = 8,
    watershed_enabled: bool = True,
    use_depth_split_walls: bool = False,
    depth_wall_threshold: float = 0.1,
    num_workers: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    fragment_dir = artifact_root / "fragments"
    fragment_dir.mkdir(parents=True, exist_ok=True)
    dataset = BaselineInstanceDataset(
        dataset_root=dataset_root,
        split=str(split),
        image_size=int(image_size),
        include_depth=str(input_mode) != "rgb" or bool(use_depth_split_walls),
        include_annotations=False,
        include_instance_map=True,
        depth_feature_mode="depth_geometry_dense" if str(input_mode) == "depth_geometry_dense" else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=lambda batch: batch[0],
    )
    model = model.to(device)
    model.eval()

    fragment_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with torch.no_grad():
        for sample in loader:
            image_id = int(sample["image_id"])
            file_name = str(sample["file_name"])
            inputs = prepare_unet_inputs(sample, input_mode=str(input_mode)).unsqueeze(0).to(device)
            outputs = model(inputs)
            label_map, decode_stats = decode_instance_predictions(
                fg_logits=outputs["fg_logits"][0].detach().cpu(),
                center_heatmap=outputs["center_heatmap"][0].detach().cpu(),
                offsets=outputs["offsets"][0].detach().cpu(),
                boundary_logits=outputs["boundary_logits"][0].detach().cpu(),
                query_depth=None if sample.get("depth") is None else sample["depth"].detach().cpu(),
                fg_threshold=float(threshold),
                center_threshold=float(center_threshold),
                min_area=int(min_area),
                watershed_enabled=bool(watershed_enabled),
                depth_wall_threshold=float(depth_wall_threshold),
            )
            label_map_np = label_map.numpy().astype(np.int32, copy=False)
            boundary_prob = _sigmoid_tensor(outputs["boundary_logits"][0].detach().cpu()).numpy()
            if boundary_prob.ndim == 3:
                boundary_prob = boundary_prob[0]
            offsets = outputs["offsets"][0].detach().cpu().numpy().astype(np.float16, copy=False)
            np.savez_compressed(
                fragment_dir / f"{image_id:06d}_{Path(file_name).stem}.npz",
                label_map=label_map_np.astype(np.uint16, copy=False),
                boundary_prob=boundary_prob.astype(np.float16, copy=False),
                offsets=offsets,
            )

            instance_map = sample["instance_map"].detach().cpu().numpy().astype(np.int64, copy=False)
            image_fragment_rows = build_fragment_records(label_map_np, instance_map)
            image_pair_rows = build_fragment_pair_records(label_map_np, image_fragment_rows)
            for row in image_fragment_rows:
                enriched = dict(row)
                enriched["image_id"] = image_id
                enriched["file_name"] = file_name
                enriched["decode_num_instances"] = float(decode_stats.get("num_instances", 0.0))
                enriched["decode_num_centers"] = float(decode_stats.get("num_centers", 0.0))
                fragment_rows.append(enriched)
            for row in image_pair_rows:
                enriched = dict(row)
                enriched["image_id"] = image_id
                enriched["file_name"] = file_name
                pair_rows.append(enriched)

    summary = summarize_fragment_quality(fragment_rows, pair_rows)
    summary["elapsed_sec"] = float(time.perf_counter() - start)
    manifest = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_dir": str(artifact_root),
        "split": str(split),
        "image_size": int(image_size),
        "input_mode": str(input_mode),
        "threshold": float(threshold),
        "center_threshold": float(center_threshold),
        "min_area": int(min_area),
        "watershed_enabled": bool(watershed_enabled),
        "use_depth_split_walls": bool(use_depth_split_walls),
        "depth_wall_threshold": float(depth_wall_threshold),
        "num_images": int(len(dataset)),
        "elapsed_sec": float(summary["elapsed_sec"]),
    }
    (artifact_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (artifact_root / "fragment_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(artifact_root / "fragment_records.jsonl", fragment_rows)
    _write_jsonl(artifact_root / "fragment_pairs.jsonl", pair_rows)
    return manifest, summary
