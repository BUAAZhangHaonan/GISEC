from __future__ import annotations

from gisec_v3.config.model_registry import get_v3_model_spec
from gisec_v3.models.uq_backbone import UQBackbone

import torch


def _count_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def test_uq_backbone_emits_shared_alpha_outputs_for_s_and_m() -> None:
    images = torch.randn(2, 3, 64, 64)
    depth = torch.randn(2, 1, 64, 64)

    model_s = UQBackbone(get_v3_model_spec("UQ-s"))
    model_m = UQBackbone(get_v3_model_spec("UQ-m"))

    outputs_s = model_s(images, depth)
    outputs_m = model_m(images, depth)

    for outputs in (outputs_s, outputs_m):
        assert set(outputs) == {
            "fg_logits",
            "boundary_logits",
            "core_heatmap",
            "ownership_offsets",
            "feature_map",
        }
        assert outputs["fg_logits"].shape == (2, 1, 64, 64)
        assert outputs["boundary_logits"].shape == (2, 1, 64, 64)
        assert outputs["core_heatmap"].shape == (2, 1, 64, 64)
        assert outputs["ownership_offsets"].shape == (2, 2, 64, 64)
        assert outputs["feature_map"].shape[0] == 2
        assert outputs["feature_map"].shape[-2:] == (64, 64)

    assert _count_params(model_m) > _count_params(model_s)
