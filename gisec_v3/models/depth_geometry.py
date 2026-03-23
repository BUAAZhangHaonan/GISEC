from __future__ import annotations

import torch


def depth_to_geometry(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expected depth tensor of shape (N, 1, H, W), got {tuple(depth.shape)}")

    depth = depth.float()
    finite = torch.isfinite(depth)
    depth = torch.where(finite, depth, torch.zeros_like(depth))

    depth_min = depth.amin(dim=(-1, -2), keepdim=True)
    depth_max = depth.amax(dim=(-1, -2), keepdim=True)
    normalized = (depth - depth_min) / (depth_max - depth_min).clamp_min(1e-6)

    grad_x = torch.zeros_like(normalized)
    grad_y = torch.zeros_like(normalized)
    grad_x[:, :, :, :-1] = normalized[:, :, :, 1:] - normalized[:, :, :, :-1]
    grad_y[:, :, :-1, :] = normalized[:, :, 1:, :] - normalized[:, :, :-1, :]
    gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    discontinuity = (gradient > 0.05).float()
    return torch.cat([normalized, gradient, discontinuity], dim=1)
