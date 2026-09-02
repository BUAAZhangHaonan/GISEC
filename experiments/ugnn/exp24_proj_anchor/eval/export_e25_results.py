"""Export the E25 canonical per-image predictions (ep77 @ SEM_THR 0.95,
legacy decode, full 3276) for future paired bootstraps.

Reuses the full-set forward cache _cache_fwd128k/ep77 (written by
eval_full_e128k.py). Verification gate: reproduced segm AP must match
the canonical 0.87350 +- 5e-4; on mismatch the output is NOT written.

Output: exp24_proj_anchor/eval/e25_fullval_results.json
"""
from __future__ import annotations
import contextlib, io, json, multiprocessing as mp, sys
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parents[1]
EVAL = HERE / "eval"
UGNN = HERE.parent
sys.path.insert(0, str(UGNN / "exp09_centernet_seeds"))
sys.path.insert(0, str(UGNN / "lib"))
import eval_centernet as ec  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import load_split  # noqa: E402

ANN = UGNN.parents[1] / "datasets" / "20260318_1K_32254" / "annotations" / "instances_val.json"
FWD = HERE / "_cache_fwd128k" / "ep77"
THR = 0.95
CANONICAL_AP = 0.87350
GATE_TOL = 5e-4
OUT = EVAL / "e25_fullval_results.json"


def _one(meta):
    image_id = meta["image_id"]
    z = np.load(FWD / f"{image_id}.npz")
    coords, cells = ec._cn_markers_with_cells(z["hm"], z["off"], decode="legacy")
    peaks = ec._marker_peaks(z["hm"], coords, cells)
    sem = (1.0 / (1.0 + np.exp(-z["sem_logit"])) > THR).astype(np.uint8)
    _, results = pf.process(image_id, coords, sem, z["depth"], z["sem_logit"], peaks)
    return results


def main() -> None:
    metas, _ = load_split("val")
    with mp.get_context("fork").Pool(16) as pool:
        all_results = []
        for rs in pool.imap_unordered(_one, metas, chunksize=8):
            all_results.extend(rs)
    coco_gt = COCO(str(ANN))
    img_ids = sorted(m["image_id"] for m in metas)
    ev = COCOeval(coco_gt, coco_gt.loadRes(all_results), "segm")
    ev.params.imgIds = img_ids
    ev.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    ap = float(ev.stats[0])
    print(f"reproduced segm AP {ap:.7f} vs canonical {CANONICAL_AP}")
    if abs(ap - CANONICAL_AP) > GATE_TOL:
        raise SystemExit("GATE FAIL: e25 export deviates from canonical")
    OUT.write_text(json.dumps(all_results))
    print(f"wrote {OUT} ({len(all_results)} instances)")


if __name__ == "__main__":
    main()
