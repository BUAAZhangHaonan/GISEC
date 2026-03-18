from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FragmentProposalBundle:
    feature_map: torch.Tensor
    fg_logits: torch.Tensor
    boundary_logits: torch.Tensor
    affinity_logits: torch.Tensor
    depth_map: torch.Tensor
