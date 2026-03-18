from __future__ import annotations

import torch
import torch.nn as nn

from gisec.models.fragment_bundle import FragmentProposalBundle


class ExternalProposalAdapter(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        if input_channels == output_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def to_fragment_bundle(
        self,
        *,
        feature_map: torch.Tensor,
        fg_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        affinity_logits: torch.Tensor,
        depth_map: torch.Tensor,
    ) -> FragmentProposalBundle:
        return FragmentProposalBundle(
            feature_map=self.proj(feature_map),
            fg_logits=fg_logits,
            boundary_logits=boundary_logits,
            affinity_logits=affinity_logits,
            depth_map=depth_map,
        )
