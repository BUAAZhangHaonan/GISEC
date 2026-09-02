from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import torch


BASELINES_DIR = Path(__file__).resolve().parent
ULTRA_REPO_ROOT = BASELINES_DIR / "ultralytics"
if str(ULTRA_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRA_REPO_ROOT))

from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import DEFAULT_CFG


def _stats_tensors(
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if mean is None or std is None:
        return None, None
    mean_tensor = torch.tensor([float(v) for v in mean], device=device, dtype=dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor([max(float(v), 1.0) for v in std], device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean_tensor, std_tensor


def normalize_images_uint8(
    images: torch.Tensor,
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
) -> torch.Tensor:
    mean_tensor, std_tensor = _stats_tensors(mean, std, device=images.device, dtype=images.dtype)
    if mean_tensor is None or std_tensor is None:
        return images / 255.0
    return (images - mean_tensor) / std_tensor


class _YOLOStatsMixin:
    def _init_rgb_stats(self, mean: Sequence[float] | None, std: Sequence[float] | None) -> None:
        self.rgb_mean = None if mean is None else [float(v) for v in mean]
        self.rgb_std = None if std is None else [max(float(v), 1.0) for v in std]

    def _normalize_batch_images(self, images: torch.Tensor) -> torch.Tensor:
        return normalize_images_uint8(images, self.rgb_mean, self.rgb_std)


class StatsNormalizedSegmentationValidator(_YOLOStatsMixin, SegmentationValidator):
    def __init__(self, *args, rgb_mean=None, rgb_std=None, **kwargs) -> None:
        self._init_rgb_stats(rgb_mean, rgb_std)
        super().__init__(*args, **kwargs)

    def preprocess(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["img"] = self._normalize_batch_images(batch["img"])
        batch["masks"] = batch["masks"].float()
        return batch


class StatsNormalizedSegmentationTrainer(_YOLOStatsMixin, SegmentationTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        rgb_mean = overrides.pop("rgb_mean", None)
        rgb_std = overrides.pop("rgb_std", None)
        self._init_rgb_stats(rgb_mean, rgb_std)
        super().__init__(cfg, overrides, _callbacks)

    def preprocess_batch(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = self._normalize_batch_images(batch["img"].float())
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            import math
            import random
            import torch.nn as nn

            size = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            scale_factor = size / max(imgs.shape[2:])
            if scale_factor != 1:
                new_shape = [math.ceil(x * scale_factor / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=new_shape, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return StatsNormalizedSegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks,
            rgb_mean=self.rgb_mean,
            rgb_std=self.rgb_std,
        )


class StatsNormalizedSegmentationPredictor(_YOLOStatsMixin, SegmentationPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        rgb_mean = overrides.pop("rgb_mean", None)
        rgb_std = overrides.pop("rgb_std", None)
        self._init_rgb_stats(rgb_mean, rgb_std)
        super().__init__(cfg, overrides, _callbacks)

    def preprocess(self, im):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            import numpy as np

            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]
            im = im.transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        if not_tensor:
            im = self._normalize_batch_images(im)
        return im
