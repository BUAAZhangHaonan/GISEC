from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.reference_splitter.dataset import ReferenceSplitCacheDataset, collate_reference_splitter_batch
from baseline.reference_splitter.model import ReferenceLocalSplitter


def train_reference_splitter_alpha(
    *,
    cache_root: str,
    reference_root: str,
    output_dir: str,
    split: str,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    roi_size: int = 128,
    reference_image_size: int = 128,
    slot_count: int = 6,
    learning_rate: float = 1.0e-3,
    max_train_steps: int = 0,
) -> None:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset = ReferenceSplitCacheDataset(
        cache_root=cache_root,
        reference_root=reference_root,
        split=split,
        roi_size=int(roi_size),
        reference_image_size=int(reference_image_size),
        slot_count=int(slot_count),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_reference_splitter_batch,
    )
    model = ReferenceLocalSplitter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    start = time.time()
    step_count = 0
    loss_total = 0.0
    loss_single_total = 0.0
    loss_count_total = 0.0
    loss_center_total = 0.0

    for _epoch in range(int(epochs)):
        model.train()
        for batch in loader:
            outputs = model(
                query_rgb=batch["query_rgb"].to(device),
                query_depth=batch["query_depth"].to(device),
                blob_mask=batch["blob_mask"].to(device),
                reference_rgb=batch["reference_rgb"].to(device),
                reference_depth=batch["reference_depth"].to(device),
                reference_mask=batch["reference_mask"].to(device),
                reference_view_ids=batch["reference_view_ids"],
            )
            single_target = batch["single_target"].to(device)
            count_target = (batch["instance_count"].to(device) - 1).clamp_min(0).clamp_max(outputs["count_logits"].shape[1] - 1)
            center_target = batch["center_heatmap"].to(device)
            loss_single = F.binary_cross_entropy_with_logits(outputs["single_object_logit"], single_target)
            loss_count = F.cross_entropy(outputs["count_logits"], count_target)
            loss_center = F.binary_cross_entropy_with_logits(outputs["center_heatmap"], center_target)
            loss = loss_single + loss_count + loss_center
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step_count += 1
            loss_total += float(loss.item())
            loss_single_total += float(loss_single.item())
            loss_count_total += float(loss_count.item())
            loss_center_total += float(loss_center.item())
            if max_train_steps > 0 and step_count >= int(max_train_steps):
                break
        if max_train_steps > 0 and step_count >= int(max_train_steps):
            break

    torch.save(model.state_dict(), artifact_root / "model_final.pth")
    summary = {
        "cache_root": str(Path(cache_root).resolve()),
        "reference_root": str(Path(reference_root).resolve()),
        "split": str(split),
        "epochs": int(epochs),
        "steps": int(step_count),
        "loss_total": 0.0 if step_count == 0 else loss_total / float(step_count),
        "loss_single": 0.0 if step_count == 0 else loss_single_total / float(step_count),
        "loss_count": 0.0 if step_count == 0 else loss_count_total / float(step_count),
        "loss_center": 0.0 if step_count == 0 else loss_center_total / float(step_count),
        "wall_time_sec": int(time.time() - start),
    }
    (artifact_root / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
