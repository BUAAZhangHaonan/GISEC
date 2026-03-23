from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gisec.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch
from gisec_v3.engine.factory import build_v3_model
from gisec_v3.engine.runtime import (
    UQRunSummary,
    predict_instance_map,
    save_run_summary,
    summarize_instance_matching,
    summarize_mask_calibration,
    summarize_object_pathology,
    summarize_matches,
    write_json,
)
from gisec_v3.train.targets import build_core_heatmap_target


def _core_target_from_instance_maps(instance_maps: torch.Tensor) -> torch.Tensor:
    targets = []
    for instance_map in instance_maps:
        core = build_core_heatmap_target(instance_map.cpu().numpy())
        targets.append(torch.from_numpy(core).float().unsqueeze(0))
    return torch.stack(targets, dim=0)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_uq_minibatch(
    *,
    dataset_root: Path,
    output_dir: Path,
    model_id: str,
    device: str = "cpu",
    image_size: int = 64,
    batch_size: int = 1,
    num_workers: int = 0,
    max_train_steps: int = 1,
    max_val_images: int = 1,
    min_area: int = 8,
) -> None:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    train_dataset = ECCGraphDataset(str(dataset_root), "train", image_size, train=True)
    val_dataset = ECCGraphDataset(str(dataset_root), "val", image_size, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_graph_batch)

    model = build_v3_model(model_id).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics_log_path = output_dir / "metrics_log.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()

    train_steps = 0
    for batch in train_loader:
        images = batch["images"].to(device_obj)
        depths = batch["depths"].to(device_obj)
        fg_target = batch["fg_target"].to(device_obj)
        boundary_target = batch["boundary_target"].to(device_obj)
        ownership_target = batch["ownership_target"].to(device_obj)
        core_target = _core_target_from_instance_maps(batch["instance_maps"]).to(device_obj)

        outputs = model(images, depths)
        loss_fg = F.binary_cross_entropy_with_logits(outputs["fg_logits"], fg_target)
        loss_boundary = F.binary_cross_entropy_with_logits(outputs["boundary_logits"], boundary_target)
        loss_core = F.binary_cross_entropy_with_logits(outputs["core_heatmap"], core_target)
        fg_mask = fg_target.expand_as(ownership_target) > 0.5
        if fg_mask.any():
            loss_ownership = F.smooth_l1_loss(outputs["ownership_offsets"][fg_mask], ownership_target[fg_mask])
        else:
            loss_ownership = outputs["ownership_offsets"].sum() * 0.0
        loss = loss_fg + loss_boundary + loss_core + loss_ownership
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        object_count = 0.0
        split_count = 0.0
        avg_cores_per_object = 0.0
        for batch_idx in range(images.shape[0]):
            pred_map, stats = predict_instance_map(
                fg_logits=outputs["fg_logits"][batch_idx, 0].detach().cpu(),
                boundary_logits=outputs["boundary_logits"][batch_idx, 0].detach().cpu(),
                core_heatmap=outputs["core_heatmap"][batch_idx, 0].detach().cpu(),
                ownership_offsets=outputs["ownership_offsets"][batch_idx].detach().cpu(),
                min_area=min_area,
            )
            object_count += stats["object_count"]
            split_count += stats["split_count"]
            avg_cores_per_object += stats["avg_cores_per_object"]

        row = {
            "mode": "train",
            "step": train_steps + 1,
            "loss": float(loss.detach().cpu()),
            "loss_fg": float(loss_fg.detach().cpu()),
            "loss_boundary": float(loss_boundary.detach().cpu()),
            "loss_core": float(loss_core.detach().cpu()),
            "loss_ownership": float(loss_ownership.detach().cpu()),
            "object_count": float(object_count / max(images.shape[0], 1)),
            "split_count": float(split_count / max(images.shape[0], 1)),
            "avg_cores_per_object": float(avg_cores_per_object / max(images.shape[0], 1)),
        }
        _append_jsonl(metrics_log_path, row)
        train_steps += 1
        if train_steps >= int(max_train_steps):
            break

    mask_rows: list[dict[str, float]] = []
    pathology_rows: list[dict[str, float]] = []
    match_rows: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= int(max_val_images):
            break
        images = batch["images"].to(device_obj)
        depths = batch["depths"].to(device_obj)
        with torch.no_grad():
            outputs = model(images, depths)
        pred_map, stats = predict_instance_map(
            fg_logits=outputs["fg_logits"][0, 0].detach().cpu(),
            boundary_logits=outputs["boundary_logits"][0, 0].detach().cpu(),
            core_heatmap=outputs["core_heatmap"][0, 0].detach().cpu(),
            ownership_offsets=outputs["ownership_offsets"][0].detach().cpu(),
            min_area=min_area,
        )
        fg_prob = torch.sigmoid(outputs["fg_logits"][0, 0].detach().cpu())
        boundary_prob = torch.sigmoid(outputs["boundary_logits"][0, 0].detach().cpu())
        mask_rows.append(
            {
                "pred_fg_rate": float((fg_prob >= 0.5).float().mean().item()),
                "pred_boundary_rate": float((boundary_prob >= 0.5).float().mean().item()),
                "target_fg_rate": float(batch["fg_target"][0, 0].float().mean().item()),
                "target_boundary_rate": float(batch["boundary_target"][0, 0].float().mean().item()),
            }
        )
        pathology_rows.append(stats)
        match_rows.append(summarize_instance_matching(batch["instance_maps"][0], pred_map))
        _append_jsonl(
            metrics_log_path,
            {
                "mode": "eval",
                "image_id": int(batch["image_ids"][0]),
                **stats,
                **match_rows[-1],
            },
        )

    write_json(output_dir / "mask_calibration_summary.json", summarize_mask_calibration(mask_rows))
    write_json(output_dir / "object_pathology_summary.json", summarize_object_pathology(pathology_rows))
    write_json(output_dir / "match_diagnostics_summary.json", summarize_matches(match_rows))
    save_run_summary(
        output_dir / "run_summary.json",
        UQRunSummary(
            model_id=model_id,
            split_mode="object_first",
            use_reference=False,
            use_graph_rescue=False,
            dataset_root=str(dataset_root),
            output_dir=str(output_dir),
            image_size=int(image_size),
            batch_size=int(batch_size),
            max_train_steps=int(max_train_steps),
            max_val_images=int(max_val_images),
        ),
    )
