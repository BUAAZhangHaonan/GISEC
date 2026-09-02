"""SeedNet: the 16.851M three-head U-Net (E10 architecture, frozen).

Two variants live here:
  ``SeedNet``    widened decoder + deep semantic head (E10, and verbatim
                 E20/E24/E25 -- the canonical model)
  ``SeedNetE9``  the pre-E10 narrow-decoder head, kept only so E9
                 lineage checkpoints stay loadable by the evaluator
                 (``gisec.eval.fullval --arch e9``)
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn

DECODER_CHANNELS = (384, 192, 96, 48, 24)
PARAM_BUDGET = 17_000_000


class SeedNet(nn.Module):
    """E10 SeedNet: smp.Unet(resnet18) + widened decoder + deep seg head."""

    def __init__(self) -> None:
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=4,
            classes=1,
            decoder_channels=DECODER_CHANNELS,
        )
        cin = DECODER_CHANNELS[-1]
        self.unet.segmentation_head = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cin // 2, 3, padding=1),
            nn.BatchNorm2d(cin // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin // 2, 1, 3, padding=1),
        )
        self.seed_head = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(cin, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x):
        feats = self.unet.encoder(x)
        dec = self.unet.decoder(feats)
        sem = self.unet.segmentation_head(dec)
        seed = self.seed_head(dec)
        return sem, seed


class SeedNetE9(nn.Module):
    """E9 pre-E10 arch: smp default decoder (16-ch) + shallow seg head."""

    def __init__(self) -> None:
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=4,
            classes=1,
        )
        # avg-pool to stride 4 first, then convs at 256 (the v1
        # head ran the first conv at 1024 and cost ~0.12 s/step)
        self.seed_head = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x):
        feats = self.unet.encoder(x)
        dec = self.unet.decoder(feats)
        sem = self.unet.segmentation_head(dec)
        seed = self.seed_head(dec)
        return sem, seed
