from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


DEFAULT_REFERENCE_GRAPH_THRESHOLDS: tuple[float, ...] = (
    0.50,
    0.505,
    0.51,
    0.515,
    0.52,
    0.525,
    0.53,
    0.535,
    0.54,
    0.545,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)


def _safe_div(num: float, den: float) -> float:
    return 0.0 if float(den) <= 0.0 else float(num) / float(den)


def compute_edge_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    if logits.numel() == 0:
        return {
            "threshold": float(threshold),
            "edge_accuracy": 0.0,
            "edge_positive_rate": 0.0,
            "pred_positive_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_valid_edges": 0.0,
        }
    logits = logits.float().reshape(-1)
    targets = targets.float().reshape(-1)
    pred = (torch.sigmoid(logits) >= float(threshold)).float()
    accuracy = float((pred == targets).float().mean().item())
    tp = float(((pred == 1.0) & (targets == 1.0)).float().sum().item())
    fp = float(((pred == 1.0) & (targets == 0.0)).float().sum().item())
    fn = float(((pred == 0.0) & (targets == 1.0)).float().sum().item())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "edge_accuracy": accuracy,
        "edge_positive_rate": float(targets.mean().item()),
        "pred_positive_rate": float(pred.mean().item()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_valid_edges": float(targets.numel()),
    }


def summarize_threshold_sweep(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    thresholds: Iterable[float] = DEFAULT_REFERENCE_GRAPH_THRESHOLDS,
    conservative_f1_margin: float = 0.005,
) -> dict[str, object]:
    rows = [compute_edge_metrics(logits, targets, threshold=float(threshold)) for threshold in thresholds]
    if not rows:
        empty = compute_edge_metrics(torch.zeros((0,)), torch.zeros((0,)), threshold=0.5)
        return {
            "rows": [],
            "best": dict(empty),
            "best_conservative": dict(empty),
        }
    best = max(rows, key=lambda row: (float(row["f1"]), float(row["precision"]), float(row["threshold"])))
    f1_floor = float(best["f1"]) - float(conservative_f1_margin)
    conservative_candidates = [row for row in rows if float(row["f1"]) >= f1_floor]
    best_conservative = min(
        conservative_candidates,
        key=lambda row: (-float(row["precision"]), float(row["pred_positive_rate"]), -float(row["threshold"])),
    )
    return {
        "rows": rows,
        "best": dict(best),
        "best_conservative": dict(best_conservative),
    }


@torch.no_grad()
def evaluate_reference_graph_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: Iterable[float] = DEFAULT_REFERENCE_GRAPH_THRESHOLDS,
    conservative_f1_margin: float = 0.005,
) -> dict[str, object]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    loss_total = 0.0
    batch_count = 0
    for batch in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        logits = model(batch)
        valid_mask = ~batch["edge_ignore_mask"].to(torch.bool)
        if logits.numel() == 0 or not bool(valid_mask.any()):
            continue
        valid_logits = logits[valid_mask].detach().float().cpu()
        valid_targets = batch["edge_targets"][valid_mask].detach().float().cpu()
        logits_all.append(valid_logits)
        targets_all.append(valid_targets)
        loss_total += float(F.binary_cross_entropy_with_logits(valid_logits, valid_targets).item())
        batch_count += 1
    if batch_count <= 0:
        sweep = summarize_threshold_sweep(torch.zeros((0,)), torch.zeros((0,)), thresholds=thresholds, conservative_f1_margin=conservative_f1_margin)
        return {
            "loss_total": 0.0,
            **sweep,
        }
    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    sweep = summarize_threshold_sweep(
        logits_cat,
        targets_cat,
        thresholds=thresholds,
        conservative_f1_margin=conservative_f1_margin,
    )
    return {
        "loss_total": float(loss_total) / float(batch_count),
        **sweep,
    }
