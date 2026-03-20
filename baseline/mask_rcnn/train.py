from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn

from baseline.common.dataset import BaselineInstanceDataset
from baseline.mask_rcnn.adapter import sample_to_mask_rcnn_target
from baseline.mask_rcnn.eval import evaluate_mask_rcnn_baseline


def train_mask_rcnn_baseline(
    *,
    dataset_root: str,
    output_dir: str,
    image_size: int,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    max_train_steps: int = 0,
    max_val_images: int = 0,
    score_threshold: float = 0.05,
) -> None:
    if int(batch_size) != 1:
        raise ValueError("Minimal Mask R-CNN smoke baseline currently expects batch_size=1")
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=image_size,
        max_size=image_size,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3, momentum=0.9)
    dataset = BaselineInstanceDataset(dataset_root=dataset_root, split="train", image_size=image_size, include_depth=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: batch[0])
    start = time.time()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    step_count = 0
    for _epoch in range(int(epochs)):
        for sample in loader:
            image = sample["image"].to(device)
            target = sample_to_mask_rcnn_target(sample)
            target = {key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()}
            losses = model([image], [target])
            loss = sum(value for value in losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_count += 1
            if max_train_steps > 0 and step_count >= int(max_train_steps):
                break
        if max_train_steps > 0 and step_count >= int(max_train_steps):
            break
    torch.save(model.state_dict(), artifact_root / "model_final.pth")
    params_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    (artifact_root / "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
    peak_memory_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    (artifact_root / "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
    wall_time_sec = int(time.time() - start)
    (artifact_root / "wall_time_sec.txt").write_text(f"{wall_time_sec}\n", encoding="utf-8")
    evaluate_mask_rcnn_baseline(
        model=model,
        dataset_root=dataset_root,
        output_dir=output_dir,
        image_size=image_size,
        device=device,
        num_workers=num_workers,
        score_threshold=score_threshold,
        max_images=max_val_images,
    )
