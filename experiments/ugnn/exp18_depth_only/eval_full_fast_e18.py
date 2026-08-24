"""E18 depth-only full 3276 fast FINAL (pre-registered line 2).

Same scheduling shape as eval_centernet.py --profile fast (FINAL-only,
workers do no GT work) but: model = E18 1ch SeedNet (from
train_depth_only), forward feeds the depth channel only (train-parity
calibration), SEM_THR comes from --thr (the E18 sweep winner) via a
module-global override in eval_centernet — eval_centernet.py itself is
NOT modified. Scoring via gisec.eval.coco_eval.evaluate_json (same as
the canonical full-fast reports).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
UGNN = HERE.parent
E9 = UGNN / "exp09_centernet_seeds"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(UGNN / "exp08_scale_32254"))
sys.path.insert(0, str(UGNN / "exp03_unet_dense"))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
from eval_scale import DATA, load_split, rss_gb  # noqa: E402
from train_depth_only import SeedNet as SeedNetE18  # noqa: E402

ep.DATA = DATA
N_WORKERS = 16

_FLO = _FRANGE = None


def _gpu_divisors() -> None:
    global _FLO, _FRANGE
    _FLO = torch.tensor(ep.DEPTH_LO, dtype=torch.float32, device="cuda")
    _FRANGE = torch.tensor(ep.DEPTH_HI - ep.DEPTH_LO, dtype=torch.float32, device="cuda")


@torch.no_grad()
def _forward(model, depth):
    d_t = torch.from_numpy(depth).cuda()
    dn = d_t.sub(_FLO).div(_FRANGE).clamp(-1.0, 2.0)
    x = dn[..., None].permute(2, 0, 1)[None].contiguous()
    sem, seed = model(x)
    return (
        sem[0, 0].cpu().numpy(),
        torch.sigmoid(seed[0, 0]).cpu().numpy(),
        seed[0, 1:3].cpu().numpy(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "best.pth"))
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--out", default=str(HERE / "eval_full_fast_e18.json"))
    args = ap.parse_args()

    ec.SEM_THR = args.thr  # script-managed thr; eval_centernet.py untouched
    ec.load_rgb_index()
    ec._gpu_divisors()
    _gpu_divisors()
    pool = mp.get_context("fork").Pool(
        N_WORKERS, initializer=ec._worker_init, initargs=("fast",)
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = SeedNetE18()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    print(f"loaded {args.ckpt} step={ckpt.get('step')} thr={args.thr}", flush=True)

    metas, _ = load_split("val")
    ann_file = DATA / "annotations" / "instances_val.json"
    results = []
    n_pred = 0
    max_markers = 0
    t_fwd = t_depth = t_worker = 0.0
    t0 = time.perf_counter()
    with pool:

        def payloads():
            nonlocal t_depth, t_fwd
            for meta in metas:
                tp = time.perf_counter()
                depth = ep.load_depth_array(Path(meta["dpath"]))
                t_depth += time.perf_counter() - tp
                tp = time.perf_counter()
                sem_logit, hm, off = _forward(model, depth)
                t_fwd += time.perf_counter() - tp
                yield (meta, sem_logit, hm, off, depth)

        it_ = iter(payloads())
        pending = []

        def submit_more(n=8):
            for _ in range(n):
                try:
                    pending.append(pool.apply_async(ec._worker_one, (next(it_),)))
                except StopIteration:
                    break

        submit_more()
        done = 0
        while pending:
            out = pending.pop(0).get()
            t_worker += out.pop("t_worker")
            max_markers = max(max_markers, out.pop("n_markers"))
            results += out["results"]["centernet"]
            n_pred += out["counts"]["centernet"]["n_pred"]
            done += 1
            del out
            if done % 25 == 0 or done == len(metas):
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                dt = time.perf_counter() - t0
                print(
                    f"  {done}/{len(metas)} "
                    f"({dt / done:.2f} s/img, fwd {t_fwd / done:.3f} s)"
                    f" rss={rss_gb():.2f} GB",
                    flush=True,
                )
            submit_more()

    from gisec.eval.coco_eval import evaluate_json

    ev = evaluate_json(Path(ann_file), results)
    report = {
        "profile": "fast",
        "ckpt": str(args.ckpt),
        "sem_thr": args.thr,
        "grid": [
            {
                "tag": "centernet",
                "segm_AP": ev["segm/AP"],
                "segm_AP50": ev["segm/AP50"],
                "segm_AP75": ev["segm/AP75"],
                "bbox_AP": ev["bbox/AP"],
                "bbox_AP50": ev["bbox/AP50"],
                "bbox_AP75": ev["bbox/AP75"],
                "n_pred": n_pred,
                "n_pred_per_img": n_pred / len(metas),
            }
        ],
        "max_markers_per_img": max_markers,
        "latency_s_per_img": {
            "forward": t_fwd / len(metas),
            "depth_load": t_depth / len(metas),
            "worker_compute": t_worker / N_WORKERS / len(metas),
        },
        "n_images": len(metas),
    }
    Path(args.out).resolve().write_text(json.dumps(report, indent=2))
    print(json.dumps(report["grid"][0], indent=2), flush=True)
    print(f"rss_final={rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
