from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.common.coco_export import masks_to_coco_results
from gisec.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch
from gisec.engine.runtime import evaluate_json
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
from gisec_v3.train.targets import (
    build_core_heatmap_target,
    build_fg_target,
    build_instance_boundary_target,
    build_ownership_target,
)


def _build_alpha_targets_from_instance_maps(instance_maps: torch.Tensor) -> dict[str, torch.Tensor]:
    fg_targets = []
    boundary_targets = []
    core_targets = []
    ownership_targets = []
    for instance_map in instance_maps:
        instance_map_np = instance_map.cpu().numpy()
        fg_targets.append(torch.from_numpy(build_fg_target(instance_map_np)).float().unsqueeze(0))
        boundary_targets.append(torch.from_numpy(build_instance_boundary_target(instance_map_np)).float().unsqueeze(0))
        core_targets.append(torch.from_numpy(build_core_heatmap_target(instance_map_np)).float().unsqueeze(0))
        ownership_targets.append(torch.from_numpy(build_ownership_target(instance_map_np)).float())
    return {
        "fg": torch.stack(fg_targets, dim=0),
        "boundary": torch.stack(boundary_targets, dim=0),
        "core": torch.stack(core_targets, dim=0),
        "ownership": torch.stack(ownership_targets, dim=0),
    }


def _dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _balanced_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, *, max_pos_weight: float = 64.0) -> torch.Tensor:
    pos = float(targets.sum().item())
    total = float(targets.numel())
    neg = max(total - pos, 0.0)
    if pos <= 0.0:
        pos_weight = torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
    else:
        pos_weight = torch.tensor(min(max(neg / pos, 1.0), max_pos_weight), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def _focal_heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating_factor = (1.0 - pt).pow(gamma)
    return (alpha_factor * modulating_factor * ce).mean()


def _compute_alpha_losses(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fg_bce = F.binary_cross_entropy_with_logits(outputs["fg_logits"], targets["fg"])
    fg_dice = _dice_loss_from_logits(outputs["fg_logits"], targets["fg"])
    loss_fg = fg_bce + fg_dice
    loss_boundary = _balanced_bce_with_logits(outputs["boundary_logits"], targets["boundary"])
    loss_core = _focal_heatmap_loss(outputs["core_heatmap"], targets["core"])
    fg_mask = targets["fg"].expand_as(outputs["ownership_offsets"]) > 0.5
    if fg_mask.any():
        loss_ownership = F.smooth_l1_loss(outputs["ownership_offsets"][fg_mask], targets["ownership"][fg_mask])
    else:
        loss_ownership = outputs["ownership_offsets"].sum() * 0.0
    return {
        "fg": loss_fg,
        "boundary": loss_boundary,
        "core": loss_core,
        "ownership": loss_ownership,
    }


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _predicted_masks_from_instance_map(instance_map: torch.Tensor) -> list[torch.Tensor]:
    labels = [int(x) for x in torch.unique(instance_map).tolist() if int(x) > 0]
    return [(instance_map == int(label)).to(torch.uint8) for label in labels]


def _classify_failure(gt_map: torch.Tensor, pred_map: torch.Tensor) -> str:
    gt_count = len([int(x) for x in torch.unique(gt_map).tolist() if int(x) > 0])
    pred_count = len([int(x) for x in torch.unique(pred_map).tolist() if int(x) > 0])
    pred_fg_ratio = float((pred_map > 0).float().mean().item())
    if pred_count == 0:
        return "empty"
    if pred_count == 1 and gt_count > 1 and pred_fg_ratio >= 0.15:
        return "oversized_blob"
    if pred_count < gt_count:
        return "severe_under_count"
    if pred_count > gt_count:
        return "severe_over_split"
    return "normal"


def _mask_scores_from_fg_prob(instance_map: torch.Tensor, fg_prob: torch.Tensor) -> list[float]:
    scores: list[float] = []
    for mask in _predicted_masks_from_instance_map(instance_map):
        mask_bool = mask.to(dtype=torch.bool)
        if mask_bool.any():
            scores.append(float(fg_prob[mask_bool].mean().item()))
        else:
            scores.append(0.0)
    return scores


def _write_subset_annotations(
    *,
    source_ann: Path,
    output_path: Path,
    selected_image_ids: list[int],
) -> Path:
    payload = json.loads(source_ann.read_text(encoding="utf-8"))
    selected = {int(image_id) for image_id in selected_image_ids}
    payload["images"] = [item for item in payload.get("images", []) if int(item.get("id", -1)) in selected]
    payload["annotations"] = [item for item in payload.get("annotations", []) if int(item.get("image_id", -1)) in selected]
    output_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def run_uq_minibatch(
    *,
    dataset_root: Path,
    output_dir: Path,
    model_id: str,
    checkpoint: Path | None = None,
    device: str = "cpu",
    image_size: int = 64,
    batch_size: int = 1,
    num_workers: int = 0,
    max_train_steps: int = 1,
    max_val_images: int = 1,
    min_area: int = 8,
) -> None:
    start_time = time.perf_counter()
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    train_dataset = ECCGraphDataset(str(dataset_root), "train", image_size, train=True)
    val_dataset = ECCGraphDataset(str(dataset_root), "val", image_size, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_graph_batch)

    model = build_v3_model(model_id).to(device_obj)
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(Path(checkpoint), map_location=device_obj)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics_log_path = output_dir / "metrics_log.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()

    train_steps = 0
    model.train()
    for batch in train_loader:
        images = batch["images"].to(device_obj)
        depths = batch["depths"].to(device_obj)
        alpha_targets = {
            key: value.to(device_obj)
            for key, value in _build_alpha_targets_from_instance_maps(batch["instance_maps"]).items()
        }

        outputs = model(images, depths)
        losses = _compute_alpha_losses(outputs, alpha_targets)
        loss_fg = losses["fg"]
        loss_boundary = losses["boundary"]
        loss_core = losses["core"]
        loss_ownership = losses["ownership"]
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

    checkpoint_path = output_dir / "model_best.pth"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    mask_rows: list[dict[str, float]] = []
    pathology_rows: list[dict[str, float]] = []
    match_rows: list[dict[str, float]] = []
    evaluated_image_ids: list[int] = []
    failure_counts = {
        "normal": 0,
        "empty": 0,
        "oversized_blob": 0,
        "severe_under_count": 0,
        "severe_over_split": 0,
    }
    coco_results: list[dict] = []
    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= int(max_val_images):
            break
        images = batch["images"].to(device_obj)
        depths = batch["depths"].to(device_obj)
        with torch.no_grad():
            outputs = model(images, depths)
        alpha_targets = _build_alpha_targets_from_instance_maps(batch["instance_maps"])
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
                "target_fg_rate": float(alpha_targets["fg"][0, 0].float().mean().item()),
                "target_boundary_rate": float(alpha_targets["boundary"][0, 0].float().mean().item()),
            }
        )
        pathology_rows.append(stats)
        match_rows.append(summarize_instance_matching(batch["instance_maps"][0], pred_map))
        failure_counts[_classify_failure(batch["instance_maps"][0], pred_map)] += 1
        pred_masks = [mask.cpu().numpy().astype("uint8") for mask in _predicted_masks_from_instance_map(pred_map)]
        mask_scores = _mask_scores_from_fg_prob(pred_map, fg_prob)
        coco_results.extend(
            masks_to_coco_results(
                image_id=int(batch["image_ids"][0]),
                masks=pred_masks,
                scores=mask_scores,
            )
        )
        evaluated_image_ids.append(int(batch["image_ids"][0]))
        _append_jsonl(
            metrics_log_path,
            {
                "mode": "eval",
                "image_id": int(batch["image_ids"][0]),
                **stats,
                **match_rows[-1],
            },
        )

    results_json = output_dir / "coco_instances_results.json"
    results_json.write_text(json.dumps(coco_results, ensure_ascii=False) + "\n", encoding="utf-8")
    ann_file = dataset_root / "annotations" / "instances_val.json"
    subset_ann_file = _write_subset_annotations(
        source_ann=ann_file,
        output_path=output_dir / "instances_val.subset.json",
        selected_image_ids=evaluated_image_ids,
    )
    metrics = evaluate_json(subset_ann_file, results_json)
    write_json(output_dir / "metrics.cocoeval.json", metrics)
    write_json(output_dir / "mask_calibration_summary.json", summarize_mask_calibration(mask_rows))
    write_json(output_dir / "object_pathology_summary.json", summarize_object_pathology(pathology_rows))
    write_json(output_dir / "match_diagnostics_summary.json", summarize_matches(match_rows))
    write_json(
        output_dir / "failure_summary.json",
        {
            "total_images": int(sum(failure_counts.values())),
            "counts": failure_counts,
        },
    )
    save_run_summary(
        output_dir / "run_summary.json",
        UQRunSummary(
            variant=model_id,
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
            metrics=metrics,
            inference_speed={},
            params_trainable=int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
            wall_time_sec=float(time.perf_counter() - start_time),
            results_json=str(results_json),
        ),
    )
