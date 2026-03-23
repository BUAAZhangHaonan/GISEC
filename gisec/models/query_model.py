from __future__ import annotations

import torch
import torch.nn as nn

from gisec.config.query_models import QueryModelSpec
from gisec.models.query_uq_backbone import UQBackbone


class UQModel(nn.Module):
    def __init__(self, spec: QueryModelSpec):
        super().__init__()
        self.spec = spec
        self.backbone = UQBackbone(spec)

    def forward(self, images: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(images, depth)
