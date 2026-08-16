from __future__ import annotations

from torch import nn

from gisec.models.gisec_model import GISECModel


def _refine_model() -> GISECModel:
    return GISECModel(
        backbone=nn.Conv2d(3, 4, kernel_size=1),
        feature_channels=4,
        input_channels=3,
        use_local_refine=True,
    )


def _freeze_backbone(model: GISECModel) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False


def test_train_mode_keeps_frozen_backbone_in_eval() -> None:
    model = _refine_model()
    _freeze_backbone(model)

    model.train()

    assert model.backbone.training is False
    assert model.refiner.training is True


def test_train_mode_keeps_trainable_backbone_in_train() -> None:
    model = _refine_model()

    model.train()

    assert model.backbone.training is True
    assert model.refiner.training is True


def test_eval_mode_disables_everything() -> None:
    model = _refine_model()
    _freeze_backbone(model)

    model.eval()

    assert model.backbone.training is False
    assert model.refiner.training is False
