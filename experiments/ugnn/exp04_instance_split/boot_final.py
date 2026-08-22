"""E4: 200x scene bootstrap for the FINAL config (standalone)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))

import segmentation_models_pytorch as smp  # noqa: E402
from eval_pipeline import (  # noqa: E402
    load_split,
    predict_semantic,
    scene_bootstrap,
    score_area,
)
from eval_watershed import run_config  # noqa: E402

from gisec.eval.coco_export import masks_to_coco_results  # noqa: E402

RUNS = HERE / "runs"

model = smp.Unet(
    encoder_name="resnet18", encoder_weights=None, in_channels=4, classes=1
)
model.load_state_dict(
    torch.load(
        HERE.parent / "exp03_unet_dense" / "runs" / "best.pth", map_location="cpu"
    )
)
model.cuda()
items = load_split("val")
preds = predict_semantic(model, items)
per_img, _ = run_config(items, preds, "depth_grad", 15, "merge")

results = []
for it, insts in zip(items, per_img, strict=True):
    scores = score_area(insts, *it["img"].shape[:2])
    results.extend(
        masks_to_coco_results(
            image_id=it["image_id"],
            masks=[m for m, _ in insts],
            scores=scores,
            category_id=1,
        )
    )

ci = scene_bootstrap(items, results, n_boot=200)
print("bootstrap", ci, flush=True)
(RUNS / "bootstrap_final.json").write_text(json.dumps(ci, indent=2))
