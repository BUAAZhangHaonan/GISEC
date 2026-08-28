"""Export the E20 canonical per-image predictions for the baselines16m
paired bootstrap (calibrate_and_report.py report --e20-results).

Regenerates the E20 full-val prediction set with the exact canonical
configuration - exp20_band8/runs/best.pt (SeedNetE10), fast profile,
SEM_THR 0.9, legacy decode - by driving eval_centernet internals
(read-only; eval_centernet.py itself is untouched).  eval_centernet
only persists the bootstrap summary, which is why the per-image RLE
results need this export.

Run ON k100 (GPU + rgb cache + postproc rank cache), never on 6401:

    cd experiments/ugnn/baselines16m
    python export_e20_results.py            # full 3276 (~15 min)
    python export_e20_results.py --max-images 8   # smoke

Verification gate (full run only): the reproduced segm AP must match
the pre-registered canonical 0.84880 +- 0.0005 (decode_fix gate iii);
on mismatch the output file is NOT written.  With --max-images the
subset AP is printed without the gate.

Output: baselines16m/e20_fullval_results.json (list of COCO segm
dicts).  Copy it to 6401 (or run `report` on k100) before the queue's
paired-bootstrap step.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UGNN = HERE.parent
REPO = UGNN.parents[1]
E9 = UGNN / "exp09_centernet_seeds"
for _p in (str(E9), str(UGNN / "lib"), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_centernet as ec  # noqa: E402
import torch  # noqa: E402
from eval_scale import load_split  # noqa: E402

from gisec.eval.coco_eval import evaluate_json  # noqa: E402

CKPT = UGNN / "exp20_band8" / "runs" / "best.pth"
ANN = REPO / "datasets" / "20260318_1K_32254" / "annotations" / "instances_val.json"
CANONICAL_AP = 0.84880
GATE_TOL = 5e-4
OUT = HERE / "e20_fullval_results.json"
N_WORKERS = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    assert ec.SEM_THR == 0.9 and ec.DECODE == "legacy", (
        "the canonical configuration must be SEM_THR 0.9 / legacy decode"
    )
    ec.load_rgb_index()
    # fork pool BEFORE any CUDA context in this process (same order as
    # eval_centernet.main); workers run _worker_one in fast mode
    pool = mp.get_context("fork").Pool(
        N_WORKERS, initializer=ec._worker_init, initargs=("fast",)
    )
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = ec.SeedNetE10()
    model.load_state_dict(ckpt["model"])  # strict: arch-parity check
    model.cuda().eval()
    ec._gpu_divisors()

    metas, _ = load_split("val")
    if args.max_images:
        metas = metas[: args.max_images]
    print(f"exporting E20 predictions for {len(metas)} images", flush=True)
    t0 = time.perf_counter()

    def payloads():
        for meta in metas:
            img = ec.load_rgb_cached(meta)
            depth = ec.ep.load_depth_array(Path(meta["dpath"]))
            sem_logit, hm, off = ec._forward(model, img, depth)
            del img
            yield meta, sem_logit, hm, off, depth

    results = []
    with pool:
        for done, out in enumerate(
            pool.imap_unordered(ec._worker_one, payloads(), chunksize=1), 1
        ):
            results.extend(out["results"]["centernet"])
            if done % 250 == 0 or done == len(metas):
                print(
                    f"  {done}/{len(metas)} "
                    f"({(time.perf_counter() - t0) / done:.2f} s/img)",
                    flush=True,
                )

    img_ids = [m["image_id"] for m in metas]
    ev = evaluate_json(Path(ANN), results, img_ids=img_ids)
    row = {
        "n_images": len(img_ids),
        "segm_AP": ev["segm/AP"],
        "segm_AP50": ev["segm/AP50"],
        "segm_AP75": ev["segm/AP75"],
        "n_pred": len(results),
    }
    print(json.dumps(row), flush=True)
    if not args.max_images:
        if abs(row["segm_AP"] - CANONICAL_AP) > GATE_TOL:
            raise SystemExit(
                f"verification gate FAILED: segm AP {row['segm_AP']:.5f} "
                f"vs canonical {CANONICAL_AP} +- {GATE_TOL}; output NOT written"
            )
        print("verification gate PASS (0.84880 +- 0.0005)", flush=True)
    Path(args.out).write_text(json.dumps(results))
    print(f"wrote {args.out} ({len(results)} instances)", flush=True)


if __name__ == "__main__":
    main()
