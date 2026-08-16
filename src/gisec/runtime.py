from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _binary_morphology(mask: torch.Tensor, *, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {tuple(mask.shape)}")
    kernel = 2 * int(radius) + 1
    mask4 = mask.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(mask4, kernel_size=kernel,
                           stride=1, padding=radius)[0, 0]
    eroded = 1.0 - \
        F.max_pool2d(1.0 - mask4, kernel_size=kernel,
                     stride=1, padding=radius)[0, 0]
    return dilated, eroded


def _boundary_band(mask: torch.Tensor, *, width: int) -> torch.Tensor:
    dilated, eroded = _binary_morphology(mask, radius=max(int(width), 1))
    return (dilated - eroded) > 0.0


def _bernoulli_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1.0e-6, 1.0 - 1.0e-6)
    return -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())


def select_refinement_instances(
    *,
    mask_probs: torch.Tensor,
    binary_masks: torch.Tensor,
    instance_scores: torch.Tensor,
    boundary_band_width: int = 4,
) -> list[int]:
    if mask_probs.ndim != 3 or binary_masks.ndim != 3:
        raise ValueError(
            f"Expected mask_probs and binary_masks with shape (N, H, W), got {tuple(mask_probs.shape)} and {tuple(binary_masks.shape)}"
        )
    if mask_probs.shape != binary_masks.shape:
        raise ValueError(
            f"mask_probs and binary_masks must match, got {tuple(mask_probs.shape)} and {tuple(binary_masks.shape)}"
        )
    if instance_scores.ndim != 1 or int(instance_scores.shape[0]) != int(mask_probs.shape[0]):
        raise ValueError(
            f"Expected instance_scores with shape ({int(mask_probs.shape[0])},), got {tuple(instance_scores.shape)}"
        )
    instance_count = int(mask_probs.shape[0])
    if instance_count == 0:
        return []
    budget = min(8, int(math.ceil(0.25 * float(instance_count))))
    if budget <= 0:
        return []

    rows: list[tuple[float, float, int]] = []
    entropy_map = _bernoulli_entropy(mask_probs.float())
    for index in range(instance_count):
        band = _boundary_band(
            binary_masks[index].float(), width=int(boundary_band_width))
        if bool(band.any()):
            uncertainty = float(entropy_map[index][band].mean().item())
        else:
            uncertainty = float(entropy_map[index].mean().item())
        rows.append(
            (uncertainty, -float(instance_scores[index].item()), index))

    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [index for _uncertainty, _score_tiebreak, index in rows[:budget]]
