from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.common.dataset import BaselineInstanceDataset
from baseline.unet.eval import evaluate_unet_baseline
from baseline.unet.model import build_unet_family_model


def train_unet_baseline(
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
    threshold: float = 0.5,
    model_name: str = "unet",
) -> None:
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    model = build_unet_family_model(str(model_name)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    dataset = BaselineInstanceDataset(
        dataset_root=dataset_root,
        split="train",
        image_size=image_size,
        include_depth=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if int(batch_size) != 1:
        raise ValueError("Minimal U-Net smoke baseline currently expects batch_size=1")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )
    start = time.time()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    step_count = 0
    for _epoch in range(int(epochs)):
        for sample in loader:
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["masks"].unsqueeze(0).to(device).float().amax(dim=1, keepdim=True)
            logits = model(image)
            loss = F.binary_cross_entropy_with_logits(logits, target)
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
    evaluate_unet_baseline(
        model=model,
        model_name=str(model_name),
        dataset_root=dataset_root,
        output_dir=output_dir,
        image_size=image_size,
        device=device,
        num_workers=num_workers,
        threshold=threshold,
        max_images=max_val_images,
    )
