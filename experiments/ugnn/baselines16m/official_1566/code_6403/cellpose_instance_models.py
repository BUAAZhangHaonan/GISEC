from __future__ import annotations

import json
import math
import os
import inspect
import time
import shutil
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

import sys

BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from baseline_adapter_utils import binary_masks_to_coco_rows, coco_rows_to_jsonable, decode_coco_segmentation
from runtime_telemetry import RuntimeTelemetry

try:
    import cellpose
    from cellpose import dynamics as CELLPOSE_DYNAMICS
    from cellpose.models import CellposeModel as CELLPOSE_MODEL_CLS
    from cellpose.train import train_seg as CELLPOSE_TRAIN_SEG_FN
except ImportError as exc:  # pragma: no cover - this should fail loudly in the project env.
    raise RuntimeError(
        "Official Cellpose v3 is required for the CellPose baseline. "
        "Install cellpose==3.1.1.1 in the magformer environment."
    ) from exc


CELLPOSE_VERSION = str(getattr(cellpose, "version", getattr(cellpose, "__version__", "unknown")))
CELLPOSE_REQUIRED_VERSION = "3.1.1.1"
if CELLPOSE_VERSION != CELLPOSE_REQUIRED_VERSION:  # pragma: no cover - version drift is an environment error.
    raise RuntimeError(f"Expected cellpose=={CELLPOSE_REQUIRED_VERSION}, found {CELLPOSE_VERSION}")

CELLPOSE_FLOW_LOGIT_SCALE = 5.0
CELLPOSE_TARGET_CACHE_VERSION = f"official-cellpose-{CELLPOSE_REQUIRED_VERSION}-diffusion"
CELLPOSE_SCORE_SOURCE = "official_cellprob_mean_sigmoid"


def _load_lightweight_ecc_records(dataset_root: str | Path, split: str) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    ann_path = root / "annotations" / f"instances_{split}.json"
    img_dir = root / "images" / split
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    annotations_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        annotations_by_image_id.setdefault(int(annotation.get("image_id", -1)), []).append(dict(annotation))

    records: List[Dict[str, Any]] = []
    for image_info in payload.get("images", []):
        image_id = int(image_info["id"])
        records.append(
            {
                "image_id": image_id,
                "file_name": str(image_info["file_name"]),
                "image_path": str(img_dir / image_info["file_name"]),
                "height": int(image_info["height"]),
                "width": int(image_info["width"]),
                "annotations": annotations_by_image_id.get(image_id, []),
            }
        )
    return records


def _annotations_to_instance_map(annotations: Sequence[Mapping[str, Any]], height: int, width: int) -> np.ndarray:
    instance_map = np.zeros((int(height), int(width)), dtype=np.int32)
    for instance_id, annotation in enumerate(annotations, start=1):
        mask = decode_coco_segmentation(annotation.get("segmentation"), int(height), int(width))
        instance_map[mask > 0] = int(instance_id)
    return instance_map


def _resize_instance_map(instance_map: np.ndarray, image_size: int) -> np.ndarray:
    if instance_map.shape[:2] == (int(image_size), int(image_size)):
        return instance_map.astype(np.int32, copy=False)
    return cv2.resize(
        instance_map.astype(np.int32, copy=False),
        (int(image_size), int(image_size)),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32, copy=False)


def _cellpose_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _official_flow_array(flow_result: Any) -> np.ndarray:
    if isinstance(flow_result, tuple):
        flow_result = flow_result[0]
    flow = np.asarray(flow_result, dtype=np.float32)
    if flow.ndim == 4 and flow.shape[0] == 1:
        flow = flow[0]
    if flow.shape[0] > 2:
        flow = flow[-2:]
    return flow.astype(np.float32, copy=False)


def instance_map_to_cellpose_targets(
    instance_map: np.ndarray,
    *,
    device: str | torch.device | None = None,
    niter: int | None = None,
) -> Dict[str, np.ndarray]:
    """Build official Cellpose diffusion-flow targets for a label map."""

    instance_map = np.asarray(instance_map, dtype=np.int32)
    flow_result = CELLPOSE_DYNAMICS.masks_to_flows_gpu(
        instance_map.astype(int, copy=False),
        device=_cellpose_device(device),
        niter=niter,
    )
    flow = _official_flow_array(flow_result)
    return {
        "instance_map": instance_map,
        "cellprob": (instance_map > 0).astype(np.float32),
        "flow": flow,
    }


def _sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def _extract_cellprob(flows_for_image: Any) -> np.ndarray | None:
    if flows_for_image is None:
        return None
    if isinstance(flows_for_image, (list, tuple)) and len(flows_for_image) >= 3:
        cellprob = flows_for_image[2]
    else:
        arr = np.asarray(flows_for_image)
        if arr.ndim >= 3 and arr.shape[0] >= 3:
            cellprob = arr[2]
        else:
            return None
    return np.asarray(cellprob, dtype=np.float32)


def label_map_to_instance_predictions(
    label_map: np.ndarray,
    *,
    cellprob: np.ndarray | None = None,
    min_area: int = 15,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    label_map = np.asarray(label_map, dtype=np.int32)
    cellprob_arr = None if cellprob is None else np.asarray(cellprob, dtype=np.float32)
    masks: List[np.ndarray] = []
    scores: List[float] = []

    for instance_id in [int(v) for v in np.unique(label_map).tolist() if int(v) > 0]:
        mask = (label_map == instance_id).astype(np.uint8)
        if int(mask.sum()) < int(min_area):
            continue
        masks.append(mask)
        if cellprob_arr is None:
            scores.append(1.0)
        else:
            scores.append(float(_sigmoid_np(cellprob_arr[mask > 0]).mean()))

    return masks, np.asarray(scores, dtype=np.float32), np.zeros((len(masks),), dtype=np.int64)


def predictions_from_logits(
    *,
    flow_logits: Any,
    cellprob_logits: Any,
    min_area: int = 20,
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
    flow_niter: int = 200,
    flow_logit_scale: float = CELLPOSE_FLOW_LOGIT_SCALE,
    flow_step_size: float = 0.1,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Official Cellpose mask reconstruction from flow/cellprob logits.

    This compatibility helper no longer performs local endpoint clustering. It
    delegates mask reconstruction to `cellpose.dynamics.compute_masks`.
    """

    del flow_step_size
    flow = np.asarray(flow_logits.detach().cpu() if hasattr(flow_logits, "detach") else flow_logits, dtype=np.float32)
    if flow.ndim == 4:
        flow = flow[0]
    flow = flow / max(float(flow_logit_scale), 1e-6)
    cellprob = np.asarray(
        cellprob_logits.detach().cpu() if hasattr(cellprob_logits, "detach") else cellprob_logits,
        dtype=np.float32,
    )
    if cellprob.ndim == 3:
        cellprob = cellprob[0]

    label_map = CELLPOSE_DYNAMICS.compute_masks(
        flow,
        cellprob,
        niter=int(flow_niter),
        cellprob_threshold=0.0 if float(mask_threshold) == 0.5 else float(mask_threshold),
        flow_threshold=0.4,
        interp=True,
        do_3D=False,
        min_size=int(min_area),
        device=_cellpose_device(),
    )
    masks, scores, category_ids = label_map_to_instance_predictions(label_map, cellprob=cellprob, min_area=min_area)
    keep = scores >= float(score_threshold)
    return [mask for mask, ok in zip(masks, keep.tolist()) if ok], scores[keep], category_ids[keep]


def _cellpose_cache_key(record: Mapping[str, Any]) -> str:
    return f"{int(record['image_id']):012d}.npz"


def _read_cached_cellpose_targets(cache_path: Path) -> Dict[str, np.ndarray] | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as payload:
            cache_version = str(payload["cache_version"].item()) if "cache_version" in payload.files else ""
            if cache_version != CELLPOSE_TARGET_CACHE_VERSION:
                return None
            return {
                "instance_map": payload["instance_map"].astype(np.int32, copy=False),
                "cellprob": payload["cellprob"].astype(np.float32, copy=False),
                "flow": payload["flow"].astype(np.float32, copy=False),
            }
    except Exception:
        cache_path.unlink(missing_ok=True)
        return None


def _write_cached_cellpose_targets(cache_path: Path, targets: Mapping[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
    instance_map = np.asarray(targets["instance_map"], dtype=np.int32)
    max_instance_id = int(instance_map.max()) if instance_map.size else 0
    if max_instance_id <= np.iinfo(np.uint16).max:
        instance_map = instance_map.astype(np.uint16, copy=False)
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            instance_map=instance_map,
            cellprob=np.asarray(targets["cellprob"] > 0, dtype=np.uint8),
            flow=np.asarray(targets["flow"], dtype=np.float16),
            cache_version=np.asarray(CELLPOSE_TARGET_CACHE_VERSION),
        )
    os.replace(tmp_path, cache_path)


class ECCCellPoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_size: int,
        train: bool,
        target_cache_dir: str | Path | None = None,
    ):
        from ecc_data_utils import load_ecc_coco_rgb_image

        self.dataset_root = Path(dataset_root)
        self.split = str(split)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.records = _load_lightweight_ecc_records(self.dataset_root, self.split)
        self._load_image = load_ecc_coco_rgb_image
        self.target_cache_dir = Path(target_cache_dir) if target_cache_dir else None

    def __len__(self) -> int:
        return len(self.records)

    def _targets_for_record(self, record: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        cache_path = None
        if self.target_cache_dir is not None:
            cache_path = self.target_cache_dir / _cellpose_cache_key(record)
            cached = _read_cached_cellpose_targets(cache_path)
            if cached is not None:
                return cached

        instance_map = _annotations_to_instance_map(
            record["annotations"],
            height=int(record["height"]),
            width=int(record["width"]),
        )
        instance_map = _resize_instance_map(instance_map, self.image_size)
        targets = instance_map_to_cellpose_targets(instance_map)
        if cache_path is not None:
            _write_cached_cellpose_targets(cache_path, targets)
        return targets

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image = self._load_image(record["image_path"], image_size=self.image_size)
        targets = self._targets_for_record(record)
        return {
            "image": image,
            "instance_map": targets["instance_map"],
            "cellprob": targets["cellprob"],
            "flow": targets["flow"],
            "image_id": int(record["image_id"]),
            "file_name": str(record["file_name"]),
            "height": int(record["height"]),
            "width": int(record["width"]),
        }


def _precompute_cellpose_record(
    *,
    record: Mapping[str, Any],
    image_size: int,
    cache_dir: Path,
) -> str:
    cache_path = cache_dir / _cellpose_cache_key(record)
    if cache_path.exists() and _read_cached_cellpose_targets(cache_path) is not None:
        return "existing"
    instance_map = _annotations_to_instance_map(
        record["annotations"],
        height=int(record["height"]),
        width=int(record["width"]),
    )
    instance_map = _resize_instance_map(instance_map, int(image_size))
    _write_cached_cellpose_targets(cache_path, instance_map_to_cellpose_targets(instance_map))
    return "created"


def precompute_cellpose_target_cache(
    *,
    dataset_root: str | Path,
    split: str,
    image_size: int,
    target_cache_dir: str | Path,
    num_workers: int = 0,
    max_images: int = 0,
) -> Dict[str, Any]:
    records = _load_lightweight_ecc_records(dataset_root, split)
    if int(max_images) > 0:
        records = records[: int(max_images)]
    cache_dir = Path(target_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def _run(record: Mapping[str, Any]) -> str:
        return _precompute_cellpose_record(record=record, image_size=int(image_size), cache_dir=cache_dir)

    if int(num_workers) > 0 and len(records) > 1:
        with ThreadPoolExecutor(max_workers=int(num_workers)) as pool:
            results = list(pool.map(_run, records))
    else:
        results = [_run(record) for record in records]
    return {
        "records": len(records),
        "created": int(sum(result == "created" for result in results)),
        "existing": int(sum(result == "existing" for result in results)),
        "cache_dir": str(cache_dir),
        "image_size": int(image_size),
        "split": str(split),
        "cache_version": CELLPOSE_TARGET_CACHE_VERSION,
        "elapsed_sec": max(0.0, time.perf_counter() - start),
    }


def _collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        out[key] = [item[key] for item in batch]
    return out


def _count_params(model: Any) -> int:
    net = getattr(model, "net", model)
    if not hasattr(net, "parameters"):
        return 0
    return int(sum(param.numel() for param in net.parameters() if getattr(param, "requires_grad", False)))


def _load_arrays_for_split(
    dataset_root: str | Path,
    split: str,
    image_size: int,
    *,
    max_images: int = 0,
    num_workers: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    from ecc_data_utils import load_ecc_coco_rgb_image

    records = _load_lightweight_ecc_records(dataset_root, split)
    if int(max_images) > 0:
        records = records[: int(max_images)]

    def _load_one(record: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        image = load_ecc_coco_rgb_image(record["image_path"], image_size=int(image_size))
        image = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        instance_map = _annotations_to_instance_map(
            record["annotations"],
            height=int(record["height"]),
            width=int(record["width"]),
        )
        instance_map = _resize_instance_map(instance_map, int(image_size))
        return image, np.asarray(instance_map, dtype=np.int32)

    if int(num_workers) > 0 and len(records) > 1:
        with ThreadPoolExecutor(max_workers=int(num_workers)) as pool:
            loaded = list(pool.map(_load_one, records))
    else:
        loaded = [_load_one(record) for record in records]
    if not loaded:
        return [], []
    images, labels = zip(*loaded)
    return list(images), list(labels)


def _effective_epochs(epochs: int, max_train_steps: int, num_images: int, batch: int) -> int:
    if int(max_train_steps) <= 0:
        return int(epochs)
    steps_per_epoch = max(1, int(math.ceil(max(1, int(num_images)) / max(1, int(batch)))))
    return max(1, min(int(epochs), int(math.ceil(int(max_train_steps) / steps_per_epoch))))


def _scheduled_eval_epochs(epochs: int, eval_every: int) -> List[int]:
    epochs = int(epochs)
    eval_every = int(eval_every)
    if epochs <= 0 or eval_every <= 0:
        return []
    scheduled = list(range(eval_every, epochs + 1, eval_every))
    if not scheduled or scheduled[-1] != epochs:
        scheduled.append(epochs)
    return scheduled


def train_cellpose_model(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    image_size: int,
    epochs: int,
    batch: int,
    lr: float,
    num_workers: int,
    device: str,
    train_split: str = "train",
    val_split: str = "val",
    max_train_steps: int = 0,
    target_cache_dir: str | Path | None = None,
    log_every: int = 50,
    telemetry: RuntimeTelemetry | None = None,
    eval_every: int = 0,
    min_area: int = 20,
    max_val_images: int = 0,
    inference_batch_size: int = 4,
) -> Dict[str, Any]:
    del target_cache_dir, log_every
    if int(image_size) not in {16, 32, 64, 128, 256, 512, 1024}:
        raise ValueError("--image-size must be a positive supported square size")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    if telemetry is not None:
        telemetry.log_event(
            "official_cellpose_data_load_start",
            {
                "train_split": str(train_split),
                "val_split": str(val_split),
                "image_size": int(image_size),
                "num_workers": int(num_workers),
            },
        )
    train_images, train_labels = _load_arrays_for_split(
        dataset_root,
        train_split,
        image_size,
        num_workers=int(num_workers),
    )
    val_images, val_labels = _load_arrays_for_split(
        dataset_root,
        val_split,
        image_size,
        max_images=32,
        num_workers=int(num_workers),
    )
    if telemetry is not None:
        telemetry.log_event(
            "official_cellpose_data_load_end",
            {
                "train_images": len(train_images),
                "val_images": len(val_images),
                "elapsed_sec": max(0.0, time.perf_counter() - load_start),
            },
        )
    effective_epochs = _effective_epochs(epochs, max_train_steps, len(train_images), batch)
    cp_device = _cellpose_device(device)
    model = CELLPOSE_MODEL_CLS(
        gpu=cp_device.type == "cuda",
        pretrained_model=False,
        nchan=3,
        device=cp_device,
        backbone="default",
    )
    if telemetry is not None:
        telemetry.log_event(
            "official_cellpose_train_start",
            {
                "epochs": int(effective_epochs),
                "requested_epochs": int(epochs),
                "batch": int(batch),
                "train_images": len(train_images),
                "val_images": len(val_images),
            },
        )
    checkpoint_name = "model_final.pth"
    train_losses_all: List[float] = []
    test_losses_all: List[float] = []
    eval_epochs = _scheduled_eval_epochs(effective_epochs, eval_every)
    train_until_epochs = eval_epochs if eval_epochs else [int(effective_epochs)]
    current_epoch = 0
    final_ckpt = out_dir / checkpoint_name
    metrics_log_path = out_dir / "metrics.jsonl"
    if eval_epochs and metrics_log_path.exists():
        metrics_log_path.unlink()

    for target_epoch in train_until_epochs:
        chunk_epochs = int(target_epoch) - int(current_epoch)
        if chunk_epochs <= 0:
            continue
        filename, train_losses, test_losses = CELLPOSE_TRAIN_SEG_FN(
            model.net,
            train_data=train_images,
            train_labels=train_labels,
            test_data=val_images,
            test_labels=val_labels,
            load_files=False,
            batch_size=int(batch),
            learning_rate=float(lr),
            n_epochs=int(chunk_epochs),
            weight_decay=1e-5,
            channels=None,
            channel_axis=None,
            rgb=True,
            normalize=True,
            compute_flows=True,
            save_path=str(out_dir),
            save_every=max(int(chunk_epochs) + 1, 2),
            save_each=False,
            min_train_masks=1,
            model_name=checkpoint_name,
        )
        current_epoch = int(target_epoch)
        train_losses_all.extend(float(v) for v in np.asarray(train_losses).reshape(-1).tolist())
        test_losses_all.extend(float(v) for v in np.asarray(test_losses).reshape(-1).tolist())
        final_ckpt = Path(filename)
        if not final_ckpt.exists():
            final_ckpt = out_dir / checkpoint_name
            model.net.save_model(str(final_ckpt))
        root_final_ckpt = out_dir / checkpoint_name
        if final_ckpt.resolve() != root_final_ckpt.resolve():
            shutil.copy2(final_ckpt, root_final_ckpt)
            final_ckpt = root_final_ckpt

        if current_epoch in eval_epochs:
            eval_bundle = {
                "model": model,
                "checkpoint": final_ckpt,
                "trainable_params": _count_params(model),
                "epochs": int(current_epoch),
            }
            stage_ctx = telemetry.stage("epoch_eval", epoch=int(current_epoch)) if telemetry is not None else nullcontext()
            with stage_ctx:
                rows = predict_records(
                    model_bundle=eval_bundle,
                    dataset_root=dataset_root,
                    eval_split=val_split,
                    image_size=image_size,
                    min_area=min_area,
                    device=device,
                    max_val_images=max_val_images,
                    inference_batch_size=int(inference_batch_size),
                    telemetry=telemetry,
                )
                metrics = evaluate_results(
                    dataset_root=dataset_root,
                    eval_split=val_split,
                    rows=rows,
                    image_size=image_size,
                    iteration=int(current_epoch),
                )
            (out_dir / f"epoch_{current_epoch:04d}_results.json").write_text(
                json.dumps(metrics, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with metrics_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": int(current_epoch), **metrics}, ensure_ascii=False) + "\n")

    return {
        "model": model,
        "checkpoint": final_ckpt,
        "trainable_params": _count_params(model),
        "epochs": int(effective_epochs),
        "train_losses": train_losses_all,
        "test_losses": test_losses_all,
    }


def _normalize_eval_output(eval_output: Any) -> Tuple[List[np.ndarray], List[Any]]:
    if not isinstance(eval_output, tuple) or len(eval_output) < 2:
        raise RuntimeError("CellposeModel.eval returned an unexpected payload")
    masks_payload = eval_output[0]
    flows_payload = eval_output[1]
    if isinstance(masks_payload, np.ndarray) and masks_payload.ndim == 2:
        masks = [masks_payload]
    else:
        masks = [np.asarray(mask) for mask in list(masks_payload)]
    if isinstance(flows_payload, (list, tuple)) and len(flows_payload) == len(masks):
        flows = list(flows_payload)
    else:
        flows = [flows_payload for _ in masks]
    return masks, flows


def _predict_rows_for_records_batch(
    *,
    model: Any,
    records: Sequence[Mapping[str, Any]],
    image_size: int,
    min_area: int,
    device: str,
    score_threshold: float,
    mask_threshold: float,
    inference_batch_size: int = 4,
) -> List[Dict[str, Any]]:
    from ecc_data_utils import load_ecc_coco_rgb_image

    del device, mask_threshold
    if not records:
        return []
    images = [
        np.asarray(load_ecc_coco_rgb_image(record["image_path"], image_size=image_size), dtype=np.float32).transpose(2, 0, 1)
        for record in records
    ]
    eval_output = model.eval(
        images,
        batch_size=max(1, int(inference_batch_size)),
        channel_axis=0,
        normalize=True,
        compute_masks=True,
        resample=True,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=int(min_area),
    )
    label_maps, flows = _normalize_eval_output(eval_output)
    rows: List[Dict[str, Any]] = []
    for record, label_map, flow_payload in zip(records, label_maps, flows):
        cellprob = _extract_cellprob(flow_payload)
        masks, scores, category_ids = label_map_to_instance_predictions(label_map, cellprob=cellprob, min_area=min_area)
        original_size = (int(record["height"]), int(record["width"]))
        if masks and original_size != (int(image_size), int(image_size)):
            masks = [
                cv2.resize(mask.astype(np.uint8, copy=False), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
                for mask in masks
            ]
        rows.extend(
            binary_masks_to_coco_rows(
                image_id=int(record["image_id"]),
                masks=masks,
                scores=scores,
                category_ids=category_ids,
                score_threshold=float(score_threshold),
                mask_threshold=0.5,
            )
        )
    return rows


def _predict_rows_for_record(
    *,
    model: Any,
    record: Mapping[str, Any],
    image_size: int,
    min_area: int,
    device: str,
    score_threshold: float,
    mask_threshold: float,
) -> List[Dict[str, Any]]:
    return _predict_rows_for_records_batch(
        model=model,
        records=[record],
        image_size=image_size,
        min_area=min_area,
        device=device,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        inference_batch_size=1,
    )


def predict_records(
    *,
    model_bundle: Mapping[str, Any],
    dataset_root: str | Path,
    eval_split: str,
    image_size: int,
    min_area: int,
    device: str,
    max_val_images: int = 0,
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
    telemetry: RuntimeTelemetry | None = None,
    inference_batch_size: int = 4,
) -> List[Dict[str, Any]]:
    model = model_bundle["model"]
    records = _load_lightweight_ecc_records(dataset_root, eval_split)
    if int(max_val_images) > 0:
        records = records[: int(max_val_images)]
    rows: List[Dict[str, Any]] = []
    batch_size = max(1, int(inference_batch_size))
    for start_idx in range(0, len(records), batch_size):
        batch_records = records[start_idx : start_idx + batch_size]
        batch_start = time.perf_counter()
        rows.extend(
            _predict_rows_for_records_batch(
                model=model,
                records=batch_records,
                image_size=image_size,
                min_area=min_area,
                device=device,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                inference_batch_size=batch_size,
            )
        )
        if telemetry is not None:
            telemetry.log_event(
                "predict_batch",
                {
                    "batch_size": len(batch_records),
                    "predict_time_sec": max(0.0, time.perf_counter() - batch_start),
                },
            )
    return rows


def evaluate_results(
    *,
    dataset_root: str | Path,
    eval_split: str,
    rows: Sequence[Mapping[str, Any]],
    image_size: int,
    iteration: int,
) -> Dict[str, Any]:
    from coco_eval_results import evaluate_coco_results

    ann_file = Path(dataset_root) / "annotations" / f"instances_{eval_split}.json"
    tmp_results = Path(dataset_root) / f".cellpose_eval_{int(image_size)}_{int(iteration)}.json"
    tmp_results.write_text(
        json.dumps(coco_rows_to_jsonable(rows), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        return evaluate_coco_results(ann_file, tmp_results, iteration=int(iteration))
    finally:
        if tmp_results.exists():
            tmp_results.unlink()


def _load_model_from_checkpoint(checkpoint: str | Path, *, device: str | torch.device = "cpu") -> Any:
    cp_device = _cellpose_device(device)
    return CELLPOSE_MODEL_CLS(
        gpu=cp_device.type == "cuda",
        pretrained_model=str(checkpoint),
        nchan=3,
        device=cp_device,
        backbone="default",
    )


def run_experiment(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    image_size: int,
    epochs: int,
    batch: int,
    lr: float,
    num_workers: int,
    min_area: int,
    device: str,
    max_train_steps: int = 0,
    max_val_images: int = 0,
    train_split: str = "train",
    val_split: str = "val",
    target_cache_dir: str | Path | None = None,
    log_every: int = 50,
    inference_batch_size: int = 4,
    eval_every: int = 0,
    train_model_fn=train_cellpose_model,
    predict_records_fn=predict_records,
    evaluate_results_fn=evaluate_results,
) -> Dict[str, Any]:
    from baseline_adapter_utils import write_baseline_run_artifacts

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    telemetry = RuntimeTelemetry(output_dir, run_name=f"cellpose_{int(image_size)}")

    train_kwargs = {
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "image_size": image_size,
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "num_workers": num_workers,
        "device": device,
        "train_split": train_split,
        "val_split": val_split,
        "max_train_steps": max_train_steps,
        "target_cache_dir": target_cache_dir,
        "log_every": log_every,
        "telemetry": telemetry,
        "eval_every": eval_every,
        "min_area": min_area,
        "max_val_images": max_val_images,
        "inference_batch_size": inference_batch_size,
    }
    accepted_train_params = set(inspect.signature(train_model_fn).parameters)
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in inspect.signature(train_model_fn).parameters.values()):
        train_kwargs = {key: value for key, value in train_kwargs.items() if key in accepted_train_params}
    with telemetry.stage("train", epochs=int(epochs), batch=int(batch), num_workers=int(num_workers)):
        bundle = train_model_fn(**train_kwargs)
    if not isinstance(bundle, Mapping):
        bundle = {
            "model": bundle,
            "checkpoint": output_dir / "model_final.pth",
            "trainable_params": _count_params(bundle),
            "epochs": int(epochs),
        }

    checkpoint = Path(bundle.get("checkpoint", output_dir / "model_final.pth"))
    trainable_params = int(bundle.get("trainable_params", _count_params(bundle.get("model"))))
    predict_kwargs = {
        "model_bundle": bundle,
        "dataset_root": dataset_root,
        "eval_split": val_split,
        "image_size": image_size,
        "min_area": min_area,
        "device": device,
        "max_val_images": max_val_images,
        "inference_batch_size": int(inference_batch_size),
        "telemetry": telemetry,
    }
    accepted_predict_params = set(inspect.signature(predict_records_fn).parameters)
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in inspect.signature(predict_records_fn).parameters.values()):
        predict_kwargs = {key: value for key, value in predict_kwargs.items() if key in accepted_predict_params}
    with telemetry.stage("predict", max_val_images=int(max_val_images)):
        rows = predict_records_fn(**predict_kwargs)
    with telemetry.stage("coco_eval", rows=len(rows)):
        metrics = evaluate_results_fn(
            dataset_root=dataset_root,
            eval_split=val_split,
            rows=rows,
            image_size=image_size,
            iteration=int(bundle.get("epochs", epochs)),
        )
    telemetry.write_summary({"model_id": "cellpose", "image_size": int(image_size)})

    artifacts = write_baseline_run_artifacts(
        output_dir,
        coco_rows=rows,
        metrics=metrics,
        metadata={
            "model_id": "cellpose",
            "model_name": "cellpose",
            "image_size": int(image_size),
            "epochs": int(bundle.get("epochs", epochs)),
            "requested_epochs": int(epochs),
            "batch": int(batch),
            "lr": float(lr),
            "num_workers": int(num_workers),
            "min_area": int(min_area),
            "train_split": str(train_split),
            "val_split": str(val_split),
            "max_train_steps": int(max_train_steps),
            "max_val_images": int(max_val_images),
            "target_cache_dir": str(target_cache_dir) if target_cache_dir else "",
            "log_every": int(log_every),
            "eval_every": int(eval_every),
            "cellpose_version": CELLPOSE_VERSION,
            "cellpose_backbone": "default",
            "score_source": CELLPOSE_SCORE_SOURCE,
            "official_code_used": True,
        },
        last_checkpoint=checkpoint.name,
        wall_time_sec=max(0.0, time.time() - start_time),
        trainable_params=trainable_params,
    )
    return {"bundle": bundle, "metrics": metrics, "rows": rows, "artifacts": artifacts}
