"""Training entry for the 14-18M baselines on 32254.

Equal budget to GISEC: 20 epochs, batch 8 @ 1024, 3206 steps/epoch,
1024 direct read, no multi-scale augmentation. Community-standard
optimizers: Mask R-CNN SGD 0.02, Mask2Former AdamW 5e-5 (backbone 0.1x).
Both use 500-step linear warmup + cosine decay. bf16 AMP.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from build_models import build_model
from common import (
    CALIB_EPOCHS,
    EPOCH_STEPS,
    EPOCHS,
    FAMILIES,
    PARAM_BUDGET,
    Baseline16mDataset,
    JsonLogger,
    collate_m2f,
    collate_mrcnn,
    family_data_flags,
    num_params,
    unpack_masks,
)
from torch.utils.data import DataLoader

WARMUP_STEPS = 500
LOG_EVERY = 100


def make_optimizer(family: str, model: torch.nn.Module):
    if family.startswith("mrcnn16"):
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.02,
            momentum=0.9,
            weight_decay=1e-4,
        )
        groups = [group for group in optimizer.param_groups]
    else:
        backbone_ids = {
            id(p) for p in model.model.pixel_level_module.encoder._backbone.parameters()
        }
        backbone_params = [
            p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids
        ]
        other_params = [
            p
            for p in model.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 5e-6},
                {"params": other_params, "lr": 5e-5},
            ],
            weight_decay=0.05,
        )
        groups = optimizer.param_groups
    return optimizer, groups


def lr_lambda(step: int) -> float:
    total = EPOCH_STEPS * EPOCHS
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonLogger(out_dir / "history.jsonl")

    is_mrcnn = args.family.startswith("mrcnn16")
    include_depth, imagenet_norm = family_data_flags(args.family)
    model = build_model(args.family).cuda()
    # torch>=2.10 optimizer.load_state_dict REPLACES the param-group
    # dicts, so nothing captured here survives a resume: read lr
    # through the live optimizer when logging.
    optimizer, _ = make_optimizer(args.family, model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    total_params = num_params(model)
    assert total_params <= PARAM_BUDGET, (
        f"{args.family}: {total_params:,} trainable params exceed the "
        f"strict equal-budget ceiling {PARAM_BUDGET:,}"
    )
    logger.write({"event": "start", "family": args.family, "params": total_params})

    collate = collate_m2f if not is_mrcnn else collate_mrcnn
    dataset = Baseline16mDataset(
        "train",
        include_depth=include_depth,
        include_annotations=True,
        imagenet_norm=imagenet_norm,
    )
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        prefetch_factor=4,
        generator=generator,
        persistent_workers=True,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        last = out_dir / "resume_last.pth"
        if last.exists():
            payload = torch.load(last, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            start_epoch = int(payload["epoch"])
            global_step = int(payload["global_step"])
            logger.write({"event": "resume", "epoch": start_epoch})

    smoke_limit = args.smoke_steps
    max_epochs = 1 if smoke_limit else args.epochs
    amp_dtype = torch.bfloat16
    torch.backends.cudnn.benchmark = True
    model.train()
    done = False
    for epoch in range(start_epoch, max_epochs):
        epoch_t0 = time.time()
        for batch in loader:
            step_t0 = time.time()
            if is_mrcnn:
                images, targets = batch
                images = [img.cuda(non_blocking=True) for img in images]
                targets = [
                    {
                        "boxes": t["boxes"].cuda(non_blocking=True),
                        "labels": t["labels"].cuda(non_blocking=True),
                        "masks": unpack_masks(
                            t["packed_masks"].cuda(non_blocking=True)
                        ),
                    }
                    for t in targets
                ]
                with torch.autocast("cuda", dtype=amp_dtype):
                    losses = model(images, targets)
                    loss = sum(losses.values())
            else:
                pixel_values, pixel_mask, packed_masks, class_labels = batch
                pixel_values = pixel_values.cuda(non_blocking=True)
                pixel_mask = pixel_mask.cuda(non_blocking=True)
                mask_labels = [
                    unpack_masks(p.cuda(non_blocking=True)).float()
                    for p in packed_masks
                ]
                class_labels = [c.cuda(non_blocking=True) for c in class_labels]
                with torch.autocast("cuda", dtype=amp_dtype):
                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        mask_labels=mask_labels,
                        class_labels=class_labels,
                        output_hidden_states=True,
                    )
                    loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % LOG_EVERY == 0 or global_step <= 5 or smoke_limit:
                logger.write(
                    {
                        "event": "step",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": round(float(loss.detach()), 4),
                        "lr": optimizer.param_groups[0]["lr"],
                        "sec_per_step": round(time.time() - step_t0, 3),
                        "peak_mem_gb": round(
                            torch.cuda.max_memory_allocated() / 2**30, 1
                        ),
                    }
                )
            if smoke_limit and global_step >= smoke_limit:
                done = True
                break
        logger.write(
            {
                "event": "epoch",
                "epoch": epoch,
                "epoch_sec": round(time.time() - epoch_t0, 1),
            }
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "family": args.family,
                "params": total_params,
            },
            out_dir / "resume_last.pth",
        )
        # Calibration checkpoints: one state_dict per epoch in CALIB_EPOCHS
        # (0-based index; epoch_19.pth duplicates model_final.pth - kept
        # so calibrate_and_report.py sees a uniform epoch_10..19 grid).
        # ~68 MB each, ~700 MB per arm.
        if not smoke_limit and epoch in CALIB_EPOCHS:
            torch.save(model.state_dict(), out_dir / f"epoch_{epoch}.pth")
        if done:
            break

    if not smoke_limit:
        torch.save(model.state_dict(), out_dir / "model_final.pth")
    logger.write({"event": "end", "family": args.family, "steps": global_step})


if __name__ == "__main__":
    main()
