from __future__ import annotations

import torch
import torch.nn as nn

from gisec.config.v3_models import V3ModelSpec
from gisec.models.v3_uq_backbone import UQBackbone


class UQModel(nn.Module):
    def __init__(self, spec: V3ModelSpec):
        super().__init__()
        self.spec = spec
        self.backbone = UQBackbone(spec)

    def forward(self, images: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(images, depth)
