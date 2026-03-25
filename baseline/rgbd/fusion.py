from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    depth = depth.float()
    min_value = float(depth.min())
    max_value = float(depth.max())
    if max_value - min_value <= 1.0e-6:
        return torch.zeros_like(depth)
    return (depth - min_value) / (max_value - min_value)


def _gradient_magnitude(depth: torch.Tensor) -> torch.Tensor:
    grad_x = torch.zeros_like(depth)
    grad_y = torch.zeros_like(depth)
    grad_x[..., :, 1:] = depth[..., :, 1:] - depth[..., :, :-1]
    grad_y[..., 1:, :] = depth[..., 1:, :] - depth[..., :-1, :]
    return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-8)


def _sobel_xy(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    squeeze = False
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)
        squeeze = True
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=depth.dtype,
        device=depth.device,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=depth.dtype,
        device=depth.device,
    ).unsqueeze(0)
    dx = F.conv2d(depth, kernel_x, padding=1)
    dy = F.conv2d(depth, kernel_y, padding=1)
    if squeeze:
        dx = dx.squeeze(0)
        dy = dy.squeeze(0)
    return dx, dy


def build_depth_geometry_channels(depth: torch.Tensor) -> torch.Tensor:
    normalized = _normalize_depth(depth)
    gradient = _gradient_magnitude(normalized)
    discontinuity = (gradient >= 0.1).float()
    return torch.cat([normalized, gradient, discontinuity], dim=0)


def build_depth_geometry_dense_channels(depth: torch.Tensor) -> torch.Tensor:
    normalized = _normalize_depth(depth)
    sobel_dx, sobel_dy = _sobel_xy(normalized)
    gradient = torch.sqrt(sobel_dx.square() + sobel_dy.square() + 1.0e-8)
    discontinuity = (gradient >= 0.1).float()
    return torch.cat([normalized, sobel_dx, sobel_dy, gradient, discontinuity], dim=0)


def prepare_unet_inputs(sample: dict, *, input_mode: str) -> torch.Tensor:
    mode = str(input_mode)
    image = sample["image"].float()
    if mode == "rgb":
        return image
    depth = sample.get("depth")
    if depth is None:
        raise ValueError(f"Depth input is required for input_mode={mode}")
    depth = depth.float()
    if mode == "rgbd":
        return torch.cat([image, depth], dim=0)
    if mode == "depth_geometry":
        return torch.cat([image, build_depth_geometry_channels(depth)], dim=0)
    if mode == "depth_geometry_dense":
        features = sample.get("depth_features")
        if features is None:
            features = build_depth_geometry_dense_channels(depth)
        return torch.cat([image, features.float()], dim=0)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def prepare_unet_batch_inputs(batch: dict, *, input_mode: str) -> torch.Tensor:
    mode = str(input_mode)
    images = batch["images"].float()
    if mode == "rgb":
        return images
    depths = batch.get("depths")
    if depths is None:
        raise ValueError(f"Depth input is required for input_mode={mode}")
    depths = depths.float()
    if mode == "rgbd":
        return torch.cat([images, depths], dim=1)
    if mode == "depth_geometry":
        geometry = torch.stack([build_depth_geometry_channels(depth) for depth in depths], dim=0)
        return torch.cat([images, geometry], dim=1)
    if mode == "depth_geometry_dense":
        features = batch.get("depth_features")
        if features is None:
            features = torch.stack([build_depth_geometry_dense_channels(depth) for depth in depths], dim=0)
        return torch.cat([images, features.float()], dim=1)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def unet_input_channels(*, input_mode: str) -> int:
    mode = str(input_mode)
    if mode == "rgb":
        return 3
    if mode == "rgbd":
        return 4
    if mode == "depth_geometry":
        return 6
    if mode == "depth_geometry_dense":
        return 8
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def unet_variant_name(*, input_mode: str, task_mode: str = "semantic_smoke") -> str:
    mode = str(input_mode)
    task = str(task_mode)
    suffix = "smoke" if task == "semantic_smoke" else "instance"
    if mode == "rgb":
        return f"rgb_{suffix}"
    if mode == "rgbd":
        return f"rgbd_{suffix}"
    if mode == "depth_geometry":
        return f"depth_geometry_{suffix}"
    if mode == "depth_geometry_dense":
        return f"depth_geometry_dense_{suffix}"
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def unet_modality(*, input_mode: str) -> str:
    return "rgb" if str(input_mode) == "rgb" else "rgbd"
