from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline.common.dataset import BaselineInstanceDataset
from baseline.mask2former.adapter import (
    build_mask2former_model,
    build_mask2former_processor,
    move_mask2former_inputs_to_device,
    sample_to_mask2former_inputs,
)
from baseline.mask2former.eval import evaluate_mask2former_baseline


def train_mask2former_baseline(
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
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    pretrained_model_name: str | None = None,
    hidden_dim: int = 64,
    feature_size: int = 64,
    mask_feature_size: int = 64,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
    num_attention_heads: int = 4,
    num_queries: int = 16,
    train_num_points: int = 512,
) -> None:
    if int(batch_size) != 1:
        raise ValueError("Minimal Mask2Former smoke baseline currently expects batch_size=1")
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    processor = build_mask2former_processor()
    model = build_mask2former_model(
        image_size=image_size,
        pretrained_model_name=pretrained_model_name,
        hidden_dim=hidden_dim,
        feature_size=feature_size,
        mask_feature_size=mask_feature_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_attention_heads=num_attention_heads,
        num_queries=num_queries,
        train_num_points=train_num_points,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    dataset = BaselineInstanceDataset(dataset_root=dataset_root, split="train", image_size=image_size, include_depth=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: batch[0])
    start = time.time()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    step_count = 0
    for _epoch in range(int(epochs)):
        for sample in loader:
            encoded = sample_to_mask2former_inputs(sample, processor=processor)
            encoded = move_mask2former_inputs_to_device(encoded, device)
            outputs = model(
                pixel_values=encoded["pixel_values"],
                pixel_mask=encoded["pixel_mask"],
                mask_labels=encoded["mask_labels"],
                class_labels=encoded["class_labels"],
                output_hidden_states=True,
            )
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Mask2Former training did not produce a loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_count += 1
            if max_train_steps > 0 and step_count >= int(max_train_steps):
                break
        if max_train_steps > 0 and step_count >= int(max_train_steps):
            break
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()}, artifact_root / "model_final.pth")
    params_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    (artifact_root / "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
    peak_memory_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    (artifact_root / "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
    wall_time_sec = int(time.time() - start)
    (artifact_root / "wall_time_sec.txt").write_text(f"{wall_time_sec}\n", encoding="utf-8")
    evaluate_mask2former_baseline(
        model=model,
        processor=processor,
        dataset_root=dataset_root,
        output_dir=output_dir,
        image_size=image_size,
        device=device,
        num_workers=num_workers,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        max_images=max_val_images,
    )
