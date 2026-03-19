from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FragmentProposalBundle:
    feature_map: torch.Tensor
    fg_logits: torch.Tensor
    boundary_logits: torch.Tensor
    ownership_offsets: torch.Tensor
    depth_map: torch.Tensor

    @property
    def affinity_logits(self) -> torch.Tensor:
        # Temporary alias while graph construction still uses the historical field name.
        return self.ownership_offsets
