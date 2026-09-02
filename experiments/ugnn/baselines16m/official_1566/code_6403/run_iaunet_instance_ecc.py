#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

BASELINES_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINES_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.baseline_adapter_utils import (
    annotations_to_instance_targets,
    binary_masks_to_coco_rows,
    coco_rows_to_jsonable,
    write_baseline_run_artifacts,
)
from baselines.coco_eval_results import evaluate_coco_results
from baselines.ecc_data_utils import load_ecc_coco_rgb_image, load_ecc_coco_rgb_records
from baselines.iaunet_instance_models import (
    IAUNetCriterion,
    IAUNetHungarianMatcher,
    IAUNetInstanceModel,
    count_trainable_parameters,
    iaunet_inference,
)
from baselines.runtime_telemetry import RuntimeTelemetry


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _configure_process_threads(thread_count: int) -> None:
    thread_count = max(1, int(thread_count))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)
    try:
        cv2.setNumThreads(thread_count)
    except Exception:
        pass
    try:
        torch.set_num_threads(thread_count)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _worker_init_fn(_worker_id: int) -> None:
    _configure_process_threads(1)


def build_loader_kwargs(num_workers: int, use_cuda: bool) -> Dict[str, Any]:
    num_workers = int(num_workers)
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(use_cuda),
        "persistent_workers": bool(num_workers > 0),
        "collate_fn": _collate,
    }
    if num_workers > 0:
        kwargs["worker_init_fn"] = _worker_init_fn
        kwargs["prefetch_factor"] = 1
    return kwargs


def _resize_masks(masks: Sequence[np.ndarray], image_size: int) -> torch.Tensor:
    if len(masks) == 0:
        return torch.zeros((0, int(image_size), int(image_size)), dtype=torch.float32)
    resized = [
        cv2.resize(mask.astype(np.uint8), (int(image_size), int(image_size)), interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ]
    return torch.from_numpy(np.stack(resized, axis=0).astype(np.float32))


class ECCIAUNetDataset(Dataset):
    def __init__(self, dataset_root: str, split: str, image_size: int, *, train: bool) -> None:
        self.records = load_ecc_coco_rgb_records(dataset_root, split, include_targets=False)
        self.image_size = int(image_size)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = load_ecc_coco_rgb_image(record["image_path"], image_size=self.image_size)
        annotation_targets = annotations_to_instance_targets(
            record["annotations"],
            height=int(record["height"]),
            width=int(record["width"]),
        )
        masks = _resize_masks(annotation_targets["masks"], self.image_size)
        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            if masks.numel() > 0:
                masks = torch.flip(masks, dims=[2])

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        labels = torch.ones((masks.shape[0],), dtype=torch.int64)
        return {
            "image": image_tensor,
            "target": {"labels": labels, "masks": masks},
            "image_id": int(record["image_id"]),
            "orig_size": (int(record["height"]), int(record["width"])),
        }


def _collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([sample["image"] for sample in batch], dim=0),
        "targets": [sample["target"] for sample in batch],
        "image_ids": [int(sample["image_id"]) for sample in batch],
        "orig_sizes": [tuple(sample["orig_size"]) for sample in batch],
    }


def _load_existing_metrics(metrics_log_path: Path) -> List[Dict[str, Any]]:
    if not metrics_log_path.exists():
        return []
    metrics: List[Dict[str, Any]] = []
    for line in metrics_log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and "epoch" in payload:
            metrics.append(payload)
    return metrics


_EPOCH_RESULTS_RE = re.compile(r"^epoch_(\d{4})_results\.json$")


def _load_existing_epoch_metrics(output_dir: Path) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for results_path in sorted(output_dir.glob("epoch_*_results.json")):
        match = _EPOCH_RESULTS_RE.match(results_path.name)
        if match is None:
            continue
        try:
            epoch = int(match.group(1))
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        metrics.append({"epoch": epoch, **payload})
    metrics.sort(key=lambda row: int(row["epoch"]))
    return metrics


def _resolve_resume_state(output_dir: Path) -> Dict[str, Any]:
    metrics_log_path = output_dir / "metrics.jsonl"
    existing_metrics = _load_existing_metrics(metrics_log_path)
    if not existing_metrics:
        existing_metrics = _load_existing_epoch_metrics(output_dir)

    resume_epoch = 1
    best_ap = -1.0
    best_epoch = 0
    trusted_metrics = list(existing_metrics)
    stale_epochs: List[int] = []
    if existing_metrics:
        latest_epoch = max(int(row["epoch"]) for row in existing_metrics)
        resume_epoch = latest_epoch + 1
        best_ap = max(float(row.get("segm/AP", -1.0)) for row in existing_metrics)
        best_epoch_candidates = [
            int(row["epoch"])
            for row in existing_metrics
            if float(row.get("segm/AP", -1.0)) == best_ap
        ]
        best_epoch = best_epoch_candidates[-1] if best_epoch_candidates else 0
    elif metrics_log_path.exists():
        metrics_log_path.unlink()

    final_checkpoint = output_dir / "model_final.pth"
    best_checkpoint = output_dir / "model_best.pth"
    resume_checkpoint = final_checkpoint if final_checkpoint.exists() else best_checkpoint
    latest_trainer_state = output_dir / "trainer_state_latest.pth"
    final_trainer_state = output_dir / "trainer_state_final.pth"
    best_trainer_state = output_dir / "trainer_state_best.pth"
    if latest_trainer_state.exists():
        resume_trainer_state = latest_trainer_state
    else:
        resume_trainer_state = final_trainer_state if final_trainer_state.exists() else best_trainer_state
    if existing_metrics and not final_checkpoint.exists() and best_checkpoint.exists():
        # A killed run may have metrics for epochs after model_best.pth was saved.
        # Those weights are not resumable, so keep only metrics that match the checkpoint.
        trusted_metrics = [row for row in existing_metrics if int(row["epoch"]) <= int(best_epoch)]
        stale_epochs = [int(row["epoch"]) for row in existing_metrics if int(row["epoch"]) > int(best_epoch)]
        resume_epoch = int(best_epoch) + 1

    return {
        "resume_epoch": resume_epoch,
        "resume_checkpoint": resume_checkpoint,
        "resume_trainer_state": resume_trainer_state,
        "existing_metrics": trusted_metrics,
        "best_ap": best_ap,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
    }


def _should_eval_epoch(*, epoch: int, epochs: int, eval_every: int, stop_after_epoch: bool = False) -> bool:
    if bool(stop_after_epoch):
        return True
    if int(epoch) == int(epochs):
        return True
    if int(eval_every) <= 0:
        return False
    return int(epoch) % int(eval_every) == 0


@torch.no_grad()
def run_eval(
    *,
    model: IAUNetInstanceModel,
    loader: DataLoader,
    device: torch.device,
    ann_file: Path,
    results_json: Path,
    iteration: int,
    score_threshold: float,
    mask_threshold: float,
    min_area: int,
    max_images: int | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    seen = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=device.type == "cuda")
        outputs = model(images)
        predictions = iaunet_inference(
            outputs,
            original_sizes=batch["orig_sizes"],
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            min_area=min_area,
        )
        for image_id, prediction in zip(batch["image_ids"], predictions):
            if max_images is not None and seen >= int(max_images):
                break
            rows.extend(
                binary_masks_to_coco_rows(
                    image_id=int(image_id),
                    masks=prediction["masks"].numpy(),
                    scores=prediction["scores"].numpy(),
                    category_ids=prediction["category_ids"].numpy(),
                    score_threshold=score_threshold,
                    mask_threshold=mask_threshold,
                )
            )
            seen += 1
        if max_images is not None and seen >= int(max_images):
            break

    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(coco_rows_to_jsonable(rows), ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = evaluate_coco_results(ann_file=ann_file, results_json=results_json, iteration=iteration)
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[512, 1024], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--val-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--transformer-blocks-per-stage", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    _configure_process_threads(max(1, min(4, (os.cpu_count() or 1) // max(1, int(args.num_workers) or 1))))
    train_dataset = ECCIAUNetDataset(args.dataset_root, args.train_split, args.image_size, train=True)
    val_dataset = ECCIAUNetDataset(args.dataset_root, args.val_split, args.image_size, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch),
        shuffle=True,
        **build_loader_kwargs(args.num_workers, use_cuda=use_cuda),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(args.val_batch)),
        shuffle=False,
        **build_loader_kwargs(args.num_workers, use_cuda=use_cuda),
    )

    model = IAUNetInstanceModel(
        in_channels=3,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        transformer_blocks_per_stage=args.transformer_blocks_per_stage,
        num_heads=args.num_heads,
    ).to(device)
    criterion = IAUNetCriterion(matcher=IAUNetHungarianMatcher())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp) and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    trainable_params = count_trainable_parameters(model)
    start_time = time.time()
    telemetry = RuntimeTelemetry(output_dir, run_name=f"iaunet_{int(args.image_size)}")
    metrics_log_path = output_dir / "metrics.jsonl"
    resume_state = _resolve_resume_state(output_dir)
    existing_metrics = resume_state["existing_metrics"]
    resume_epoch = int(resume_state["resume_epoch"])
    resume_checkpoint = Path(resume_state["resume_checkpoint"])
    resume_trainer_state = Path(resume_state["resume_trainer_state"])
    best_ap = float(resume_state["best_ap"])
    best_epoch = int(resume_state["best_epoch"])
    if resume_trainer_state.exists() and existing_metrics:
        trainer_state = torch.load(resume_trainer_state, map_location=device, weights_only=False)
        model.load_state_dict(trainer_state["model"])
        if "optimizer" in trainer_state:
            optimizer.load_state_dict(trainer_state["optimizer"])
        best_ap = float(trainer_state.get("best_ap", best_ap))
        best_epoch = int(trainer_state.get("best_epoch", best_epoch))
    elif resume_checkpoint.exists() and existing_metrics:
        state = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)

    ann_file = Path(args.dataset_root) / "annotations" / f"instances_{args.val_split}.json"
    best_path = output_dir / "model_best.pth"
    total_steps = 0
    stop_after_epoch = False
    grad_accum_steps = max(1, int(args.grad_accum_steps))

    for epoch in range(resume_epoch, int(args.epochs) + 1):
        model.train()
        last_step_end = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        pending_accum_steps = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            data_ready = time.perf_counter()
            data_time = max(0.0, data_ready - last_step_end)
            compute_start = time.perf_counter()
            images = batch["images"].to(device, non_blocking=device.type == "cuda")
            targets = [
                {
                    "labels": target["labels"].to(device, non_blocking=device.type == "cuda"),
                    "masks": target["masks"].to(device, non_blocking=device.type == "cuda"),
                }
                for target in batch["targets"]
            ]
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                losses = criterion(outputs, targets)
                loss = sum(losses.values())
                scaled_loss = loss / float(grad_accum_steps)

            scaler.scale(scaled_loss).backward()
            pending_accum_steps += 1
            will_stop_after_batch = int(args.max_train_steps) > 0 and (total_steps + 1) >= int(args.max_train_steps)
            is_accum_boundary = pending_accum_steps >= grad_accum_steps
            is_last_batch = int(batch_idx) == len(train_loader)
            if is_accum_boundary or is_last_batch or will_stop_after_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_accum_steps = 0
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time = max(0.0, time.perf_counter() - compute_start)
            last_step_end = time.perf_counter()

            total_steps += 1
            if total_steps == 1 or (int(args.log_every) > 0 and total_steps % int(args.log_every) == 0):
                telemetry.log_event(
                    "train_step",
                    {
                        "epoch": int(epoch),
                        "step": int(total_steps),
                        "batch": int(images.shape[0]),
                        "data_time_sec": data_time,
                        "compute_time_sec": compute_time,
                        "imgs_per_sec": float(images.shape[0]) / max(compute_time, 1e-9),
                        "loss": float(loss.detach().cpu()),
                    },
                )
            if will_stop_after_batch:
                stop_after_epoch = True
                break

        should_eval = _should_eval_epoch(
            epoch=int(epoch),
            epochs=int(args.epochs),
            eval_every=int(args.eval_every),
            stop_after_epoch=bool(stop_after_epoch),
        )
        if should_eval:
            epoch_results_path = output_dir / f"epoch_{epoch:04d}_results.json"
            with telemetry.stage("epoch_eval", epoch=int(epoch)):
                metrics, _rows = run_eval(
                    model=model,
                    loader=val_loader,
                    device=device,
                    ann_file=ann_file,
                    results_json=epoch_results_path,
                    iteration=epoch,
                    score_threshold=args.score_threshold,
                    mask_threshold=args.mask_threshold,
                    min_area=args.min_area,
                    max_images=int(args.max_val_images) if int(args.max_val_images) > 0 else None,
                )
            with metrics_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False) + "\n")

            segm_ap = float(metrics.get("segm/AP", 0.0))
            if segm_ap >= best_ap:
                best_ap = segm_ap
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
        torch.save(
            {
                "epoch": int(epoch),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_ap": float(best_ap),
                "best_epoch": int(best_epoch),
            },
            output_dir / "trainer_state_latest.pth",
        )
        if stop_after_epoch:
            break

    final_ckpt = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)

    final_results_path = output_dir / "coco_instances_results.json"
    with telemetry.stage("final_eval"):
        final_metrics, final_rows = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            ann_file=ann_file,
            results_json=final_results_path,
            iteration=int(args.epochs),
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            min_area=args.min_area,
            max_images=int(args.max_val_images) if int(args.max_val_images) > 0 else None,
        )

    metadata = {
        "model_id": "iaunet",
        "model_name": "iaunet",
        "image_size": int(args.image_size),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "num_queries": int(args.num_queries),
        "hidden_dim": int(args.hidden_dim),
        "base_channels": int(args.base_channels),
        "num_decoder_layers": int(args.num_decoder_layers),
        "transformer_blocks_per_stage": int(args.transformer_blocks_per_stage),
        "batch": int(args.batch),
        "val_batch": int(args.val_batch),
        "num_workers": int(args.num_workers),
        "eval_every": int(args.eval_every),
        "grad_accum_steps": int(grad_accum_steps),
        "amp": bool(args.amp),
        "best_epoch": int(best_epoch),
        "best_segm_ap": float(best_ap),
        "implementation_kind": "paper-faithful-reimplementation",
        "official_code_used": False,
        "paper_faithful": True,
        "score_source": "class_probability_times_maskness",
    }
    write_baseline_run_artifacts(
        output_dir,
        coco_rows=final_rows,
        metrics=final_metrics,
        metadata=metadata,
        last_checkpoint=final_ckpt.name,
        wall_time_sec=int(time.time() - start_time),
        trainable_params=trainable_params,
    )
    telemetry.write_summary({"model_id": "iaunet", "image_size": int(args.image_size)})
    for stale_state in [
        output_dir / "trainer_state_latest.pth",
        output_dir / "trainer_state_final.pth",
        output_dir / "trainer_state_best.pth",
    ]:
        stale_state.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
