"""Filesystem layout for the GISEC repository.

All dataset / record / cache locations resolve from the repository
root that hosts this package (the project runs as ``pip install -e .``
from the repo checkout) and can be overridden through environment
variables, so no module hardcodes absolute paths.

Environment overrides (read once at import):
  GISEC_DATA_ROOT           dataset root (default
                            ``<repo>/datasets/20260318_1K_32254``)
  GISEC_GT_RECORDS          E9 GT-record dir (stats/sem memmaps)
  GISEC_BAND_RECORDS        E17 band-record dir
  GISEC_PROJANCHOR_RECORDS  E24 projected-anchor record dir
  GISEC_RGB_CACHE           inference RGB pre-decode cache root
  GISEC_POSTPROC_CACHE      watershed rank cache root
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UGNN = REPO_ROOT / "experiments" / "ugnn"

DATA_ROOT = Path(
    os.environ.get("GISEC_DATA_ROOT", str(REPO_ROOT / "datasets" / "20260318_1K_32254"))
)
GT_RECORDS = Path(
    os.environ.get(
        "GISEC_GT_RECORDS", str(UGNN / "exp09_centernet_seeds" / "gt_records")
    )
)
BAND_RECORDS = Path(
    os.environ.get("GISEC_BAND_RECORDS", str(UGNN / "exp17_band_ema" / "gt_records"))
)
PROJANCHOR_RECORDS = Path(
    os.environ.get(
        "GISEC_PROJANCHOR_RECORDS", str(UGNN / "exp24_proj_anchor" / "gt_records")
    )
)
RGB_CACHE = Path(os.environ.get("GISEC_RGB_CACHE", str(REPO_ROOT / "cache_rgb")))
POSTPROC_CACHE = Path(
    os.environ.get("GISEC_POSTPROC_CACHE", str(REPO_ROOT / "cache_postproc"))
)
