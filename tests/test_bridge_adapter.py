from __future__ import annotations

import torch

from gisec.bridge.adapter import ExternalProposalAdapter


def test_external_proposal_adapter_projects_feature_channels() -> None:
    adapter = ExternalProposalAdapter(input_channels=12, output_channels=8)
    feature_map = torch.randn(1, 12, 32, 32)
    fg_logits = torch.randn(1, 1, 32, 32)
    boundary_logits = torch.randn(1, 1, 32, 32)
    affinity_logits = torch.randn(1, 2, 32, 32)
    depth_map = torch.randn(1, 1, 32, 32)

    bundle = adapter.to_fragment_bundle(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=affinity_logits,
        depth_map=depth_map,
    )

    assert bundle.feature_map.shape == (1, 8, 32, 32)
    assert bundle.fg_logits.shape == (1, 1, 32, 32)
    assert bundle.affinity_logits.shape == (1, 2, 32, 32)
