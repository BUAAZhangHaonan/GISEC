from __future__ import annotations

import numpy as np
import torch


def boxes_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    converted = boxes.clone()
    converted[:, 2] = converted[:, 0] + converted[:, 2]
    converted[:, 3] = converted[:, 1] + converted[:, 3]
    return converted


def sample_to_mask_rcnn_target(sample: dict) -> dict:
    boxes = boxes_xywh_to_xyxy(sample["boxes"].float())
    masks = sample["masks"].to(torch.uint8)
    labels = sample["labels"].to(torch.int64)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return {
        "boxes": boxes,
        "labels": labels,
        "masks": masks,
        "image_id": torch.tensor([int(sample["image_id"])], dtype=torch.int64),
        "area": area,
        "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
    }


def outputs_to_instance_masks(output: dict, *, score_threshold: float) -> tuple[list[np.ndarray], list[float]]:
    masks: list[np.ndarray] = []
    scores: list[float] = []
    for score, mask in zip(output["scores"].tolist(), output["masks"]):
        if float(score) < float(score_threshold):
            continue
        binary = (mask[0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
        if int(binary.sum()) <= 0:
            continue
        masks.append(binary)
        scores.append(float(score))
    return masks, scores
