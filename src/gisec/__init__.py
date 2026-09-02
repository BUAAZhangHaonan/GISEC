"""GISEC: instance segmentation for electronic components in dense
clutter.

Canonical recipe (E25, switched 2026-09-02): a 16.851M three-head
U-Net (semantic + CenterNet heatmap + offset, :mod:`gisec.model`)
trained with band-weighted BCE x8 + projected-anchor seeds
(:mod:`gisec.train` ``--anchor projected``), decoded into instances
by the depth-guided watershed (:mod:`gisec.postproc_fast`).
Full-3276-image val segm AP **0.87350** (ckpt ``ema_ep77.pth`` +
SEM_THR 0.95 + legacy decode); lineage E24 0.86113, E20 0.84880.

Module map:
  paths      dataset/record/cache locations (env-overridable)
  model      SeedNet (E10 arch) + the E9 legacy variant
  targets    CenterNet GT stamping (numba)
  anchors    in-mask projected anchor p* (E24/E25 seed source)
  losses     dice / penalty-reduced focal / offset L1 (frozen arithmetic)
  datasets   records loading + split metadata + all record builders
  train      trainer (E20/E24/E25 = --anchor centroid/projected)
  deploy_eval  in-training deployment monitor (500-img AP + overlays)
  decode     heatmap NMS + cell->pixel decode (legacy/fixed/grid)
  inference  GPU forward + RGB pre-decode cache
  postproc_fast  watershed decode + merge + COCO RLE (numba)
  eval       COCO scoring/export, diagnostics, scene bootstrap,
             full-val evaluator CLI
"""

from gisec.config.variants import (
    GisecVariantSpec,
    get_gisec_variant_spec,
    gisec_variant_names,
)

__version__ = "0.2.0"

__all__ = [
    "GisecVariantSpec",
    "__version__",
    "get_gisec_variant_spec",
    "gisec_variant_names",
]
