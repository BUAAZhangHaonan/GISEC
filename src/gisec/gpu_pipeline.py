"""GPU fast path: the whole inference chain on the GPU except the
numba watershed.

Deployment-throughput twin of the canonical chain
(``gisec.eval.fullval`` / ``postproc_fast.process``), which stays
untouched as the reference. Despite the name "fast path", this
caliber is constructed to be BITWISE-equal to the canonical
predictions (see Numerics below); the full-val paired scene
bootstrap gate vs the canonical E26b still runs before numbers from
this path are quoted anywhere.

Data flow per image (single default CUDA stream; the CPU split
releases the GIL so it overlaps the next image's IO+GPU stage):

  img u8 + depth f32  --pinned async H2D-->          ~2 ms / 7 MB
  GPU preproc (norm/concat, verbatim inference math)
  SeedNet forward                                     ~12 ms
  GPU sigmoid + 3x3 max-pool NMS + peak gather        (markers,
    scores bitwise-equal to the canonical decode: max is exact,
    nonzero/argsort raster order, round half-to-even in float64)
  GPU binarize (sem_logit > logit(thr) -> u8)
  GPU sobel magnitudes (sem_logit + raw depth) + sort rank
    (torch.sort stable radix); mix rank int64-exact
  --D2H--> sem u8 1MB + mix rank i32 4MB + small arrays
  CPU ``postproc_fast.split_from_rank`` (watershed + RLE)   ~60 ms

Numerics vs the canonical CPU chain: by construction this caliber is
expected to be BITWISE-equal to ``postproc_fast.process`` — the
slicing-arithmetic sobel mirrors the numba f64-promotion rounding,
the magnitude uses f64 hypot with a single final rounding (bitwise ==
np.hypot on f32 inputs), the sort-based rank reproduces the tie-shared
dense rank exactly, and the marker/sem stages are exact integer or
f64 comparisons. The full-val gate (paired scene bootstrap vs the
canonical E26b) still runs as belt-and-suspenders before any number
from this path is quoted. ``rank_d`` is computed fresh on the GPU —
the canonical disk rank cache is deliberately NOT consulted (the
cache md5 check would cost more than the GPU recompute).

torch is imported lazily; on a CPU-only host ``GpuPipeline`` raises
RuntimeError instead of silently falling back (a silent fallback
would falsify the caliber — the canonical chain is the CPU path).
"""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from gisec import decode, inference, postproc_fast
from gisec.datasets.coco_utils import load_depth_array
from gisec.decode import HM_THR, MAX_MARKERS
from gisec.eval.coco_eval import evaluate_json
from gisec.model import SeedNet
from gisec.paths import DATA_ROOT
from gisec.targets import STRIDE

__all__ = ["GpuPayload", "GpuPipeline", "cpu_stage", "infer_one", "load_model"]


# ------------------------------------------------------------ backend
_BACKEND = None  # None = not tried, False = no CUDA, dict = ready
_LEAK = ()  # fork-inherited handles kept alive in children


def _reset_after_fork() -> None:
    """Fork hygiene (colosseum team_c pattern): a child forked BEFORE
    parent CUDA init lazily builds its own context; a child forked
    AFTER cannot touch the inherited runtime, so it flips to
    no-backend and deliberately leaks the handles (no deallocator
    runs, no abort)."""
    global _BACKEND, _LEAK
    if isinstance(_BACKEND, dict):
        _LEAK = (_BACKEND,)
        _BACKEND = False


with contextlib.suppress(AttributeError, OSError):  # POSIX only
    os.register_at_fork(after_in_child=_reset_after_fork)


class _PinnedBuf:
    """Growable page-locked staging buffer visible to numpy and torch."""

    __slots__ = ("_torch", "a", "cap", "dtype", "t")

    def __init__(self, torch, dtype):
        self._torch = torch
        self.dtype = dtype
        self.t = None
        self.a = None
        self.cap = 0

    def ensure(self, n: int) -> None:
        if n <= self.cap:
            return
        torch = self._torch
        if self.t is not None:
            torch.cuda.synchronize()  # drain in-flight reads before realloc
        self.t = torch.empty(n, dtype=self.dtype, pin_memory=True)
        self.a = self.t.numpy()
        self.cap = n


def _init_backend():
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return False
        torch.cuda.init()
        dev = torch.device("cuda", torch.cuda.current_device())
        with torch.inference_mode():
            _, i = torch.sort(torch.zeros(4096, device=dev), stable=True)
            i.cpu()  # force full context + allocator warmup
        return {
            "torch": torch,
            "dev": dev,
            "u8": _PinnedBuf(torch, torch.uint8),
            "f32": _PinnedBuf(torch, torch.float32),
        }
    except Exception:
        return False


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _init_backend()
    return _BACKEND


# ------------------------------------------------------------ GPU stage
@dataclass
class GpuPayload:
    """CPU-side outputs of one GPU stage (everything the watershed needs)."""

    coords: list  # [(y, x)] rounded legacy-decode markers
    peaks: np.ndarray  # f64 heatmap values at the source cells
    sem: np.ndarray  # u8 (H, W) binary semantic mask
    rank: np.ndarray  # i32 (H, W) mix elevation rank
    nrank: int


def _gpu_rank_dev(st, keys, n):
    """Device-resident dense rank: sort -> boundary != -> cumsum ->
    scatter. Returns (rank i32 CUDA tensor, nrank python int); the
    nrank .item() is the only forced sync."""
    torch = st["torch"]
    keys = keys.reshape(-1)  # device-resident callers may pass (H, W)
    with torch.inference_mode():
        vals, order = torch.sort(keys, stable=True)  # cub radix
        grp = torch.empty(n, dtype=torch.int32, device=keys.device)
        grp[0] = 0
        torch.cumsum(vals[1:] != vals[:-1], dim=0, dtype=torch.int32, out=grp[1:])
        out = torch.empty(n, dtype=torch.int32, device=keys.device)
        out.scatter_(0, order, grp)
        return out, int(grp[-1].item()) + 1


def _gpu_sobel_mag_parts(st, x):
    """Sobel gx/gy by slicing arithmetic on replicate pads — the same
    IEEE add/mul-by-2 operations in the same order as the numba
    ``_sobel_xy`` (replicate borders == the CPU index clamp), so
    gx/gy are bitwise-equal to the CPU reference. x: (H, W) f32 CUDA.

    CPU reference (postproc_fast._sobel_xy):
      tmp[i,j]  = -d[i, jm1] + d[i, jp1]          (W-clamp only, f32)
      gx[i,j]   = tmp[im1, j] + 2 tmp[i, j] + tmp[ip1, j]   (H-clamp)
      tmp2[i,j] = -d[im1, j] + d[ip1, j]          (H-clamp only, f32)
      gy[i,j]   = tmp2[i, jm1] + 2 tmp2[i, j] + tmp2[i, jp1] (W-clamp)

    Numba's ``2.0 *`` on an f32 array promotes to float64 (the literal
    is f64), so each smoothing sum is computed in f64 and rounded to
    f32 once on store — the GPU version mirrors that (f64 adds, one
    final .to(float32)) to stay bitwise-equal."""
    torch = st["torch"]
    F = torch.nn.functional

    def _rep(t, pad):
        return F.pad(t[None, None], pad, mode="replicate")[0, 0]

    xw = _rep(x, (1, 1, 0, 0))  # (H, W+2)
    tmp = -xw[:, :-2] + xw[:, 2:]  # (H, W)
    tp = _rep(tmp, (0, 0, 1, 1))  # (H+2, W)
    gx = (
        tp[:-2].to(torch.float64)
        + 2.0 * tmp.to(torch.float64)
        + tp[2:].to(torch.float64)
    ).to(torch.float32)

    xh = _rep(x, (0, 0, 1, 1))  # (H+2, W)
    tmp2 = -xh[:-2, :] + xh[2:, :]  # (H, W)
    tp2 = _rep(tmp2, (1, 1, 0, 0))  # (H, W+2)
    gy = (
        tp2[:, :-2].to(torch.float64)
        + 2.0 * tmp2.to(torch.float64)
        + tp2[:, 2:].to(torch.float64)
    ).to(torch.float32)
    return gx, gy


class GpuPipeline:
    """GPU stage of the gpu_fast caliber over one loaded SeedNet."""

    def __init__(self, model, sem_thr: float | None = None):
        st = _backend()
        if st is False:
            raise RuntimeError(
                "CUDA is required for the gpu_fast pipeline; use the "
                "canonical eval chain on CPU-only hosts"
            )
        self.st = st
        self.model = model
        self.sem_thr = sem_thr  # None -> decode.SEM_THR at call time
        inference._gpu_divisors()  # _F255/_FLO/_FRANGE divisor tensors

    def gpu_stage(
        self, img_u8: np.ndarray, depth_f32: np.ndarray, sem_thr: float | None = None
    ) -> GpuPayload:
        """One GPU stage. sem_thr overrides self.sem_thr per call (the
        deploy monitor evaluates several thresholds per forward)."""
        torch = self.st["torch"]
        if sem_thr is None:
            sem_thr = decode.SEM_THR if self.sem_thr is None else self.sem_thr
        logit_thr = (
            float(np.log(sem_thr / (1.0 - sem_thr))) if 0.0 < sem_thr < 1.0 else None
        )
        with torch.inference_mode():
            # ---- uploads (pinned, async) -------------------------------
            pin8 = self.st["u8"]
            n8 = int(img_u8.size)
            pin8.ensure(n8)
            np.copyto(pin8.a[:n8].reshape(img_u8.shape), img_u8)
            img_t = pin8.t[:n8].view(img_u8.shape).to(self.st["dev"], non_blocking=True)
            pinf = self.st["f32"]
            nf = int(depth_f32.size)
            pinf.ensure(nf)
            np.copyto(pinf.a[:nf].reshape(depth_f32.shape), depth_f32)
            d_t = (
                pinf.t[:nf].view(depth_f32.shape).to(self.st["dev"], non_blocking=True)
            )

            # ---- preproc + forward (verbatim inference._forward math) --
            rgbf = img_t.to(torch.float32).div(inference._F255)
            dn = d_t.sub(inference._FLO).div(inference._FRANGE).clamp(-1.0, 2.0)
            x = (
                torch.cat([rgbf, dn[..., None]], dim=-1)
                .permute(2, 0, 1)[None]
                .contiguous()
            )
            sem, seed = self.model(x)
            sem_logit = sem[0, 0]  # (H, W) f32, stays on device
            hm = torch.sigmoid(seed[0, 0])  # (H4, W4)
            off = seed[0, 1:3]  # (2, H4, W4)

            # ---- NMS: 3x3 max pool == maximum_filter(mode='nearest') --
            mx = torch.nn.functional.max_pool2d(
                hm[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]
            peak_mask = (hm >= mx) & (hm > HM_THR)
            ys, xs = torch.nonzero(peak_mask, as_tuple=True)  # raster order
            vals = hm[ys, xs]
            if ys.numel() > MAX_MARKERS:
                order = torch.argsort(-vals, stable=True)[:MAX_MARKERS]
                ys, xs, vals = ys[order], xs[order], vals[order]
            # legacy decode in float64 (the numpy path promotes to f64):
            # y = 4*cell + off, round half-to-even, clamp, int
            hm_h4, hm_w4 = hm.shape
            y_px = ys.to(torch.float64) * STRIDE + off[0, ys, xs].to(torch.float64)
            x_px = xs.to(torch.float64) * STRIDE + off[1, ys, xs].to(torch.float64)
            y_px = torch.clamp(torch.round(y_px), 0, hm_h4 * STRIDE - 1).to(torch.int32)
            x_px = torch.clamp(torch.round(x_px), 0, hm_w4 * STRIDE - 1).to(torch.int32)

            # ---- binarize ------------------------------------------------
            if logit_thr is not None:
                sem_bin = (sem_logit > logit_thr).to(torch.uint8)
            else:  # thr outside (0,1): mirror the literal sigmoid form
                sem_bin = (torch.sigmoid(sem_logit) > sem_thr).to(torch.uint8)

            # ---- elevation ranks, device-resident ----------------------
            # hypot in f64 then one rounding == np.hypot(f32,f32)
            # bitwise (numpy computes its f32 hypot in double; verified
            # 100% equal on random grids — see tests/test_gpu_pipeline)
            def _mag(x_, y_):
                return torch.hypot(x_.to(torch.float64), y_.to(torch.float64)).to(
                    torch.float32
                )

            gx, gy = _gpu_sobel_mag_parts(self.st, sem_logit)
            rank_s = _gpu_rank_dev(self.st, _mag(gx, gy) + 0.0, gx.numel())[0]
            dgx, dgy = _gpu_sobel_mag_parts(self.st, d_t)  # raw depth
            rank_d = _gpu_rank_dev(self.st, _mag(dgx, dgy) + 0.0, dgx.numel())[0]
            mixed = rank_d.to(torch.int64) + 2 * rank_s.to(torch.int64)
            flat_rank, nrank = _gpu_rank_dev(self.st, mixed, mixed.numel())
            rank = flat_rank.view(sem_logit.shape)

            # ---- downloads ----------------------------------------------
            coords = list(
                zip(
                    y_px.cpu().numpy().tolist(),
                    x_px.cpu().numpy().tolist(),
                    strict=True,
                )
            )
            peaks = vals.cpu().numpy().astype(np.float64)
            sem_np = sem_bin.cpu().numpy()
            rank_np = rank.cpu().numpy()
        return GpuPayload(coords, peaks, sem_np, rank_np, nrank)


# ------------------------------------------------------------ CPU stage
def cpu_stage(payload: GpuPayload, image_id: int):
    """Frozen numba watershed + RLE over one GPU payload."""
    coords, peaks = postproc_fast.dedup_markers(payload.coords, payload.peaks)
    return postproc_fast.split_from_rank(
        image_id, coords, peaks, payload.sem, payload.rank, payload.nrank
    )


# ------------------------------------------------------------ helpers
def load_model(ckpt: str | Path):
    """Eval-mode SeedNet on CUDA from an EMA-format checkpoint."""
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = SeedNet()
    model.load_state_dict(ck["model"])
    return model.cuda().eval()


def infer_one(
    model, img_u8, depth_f32, image_id: int = 0, sem_thr: float | None = None
):
    """Single-image deployment API (pure compute; no disk IO).

    img_u8: (H, W, 3) uint8 RGB; depth_f32: (H, W) raw depth (same
    units as the training calibration). Returns (insts, coco_results),
    the postproc_fast.process contract."""
    gp = getattr(model, "_gisec_gpu_pipeline", None)
    if gp is None or (sem_thr is not None and gp.sem_thr != sem_thr):
        gp = GpuPipeline(model, sem_thr)
        with contextlib.suppress(Exception):
            model._gisec_gpu_pipeline = gp  # cache on the model instance
    payload = gp.gpu_stage(img_u8, depth_f32)
    return cpu_stage(payload, image_id)


# ------------------------------------------------------------ batch + CLI
def _cpu_job(payload, image_id):
    t = perf_counter()
    out = cpu_stage(payload, image_id)
    return out, perf_counter() - t


def run_batch(
    model,
    metas,
    sem_thr: float | None = None,
    pipeline: str = "threaded",
    on_result=None,
):
    """Batch inference over split metas. Returns (results, latency).

    pipeline='threaded' overlaps the CPU watershed of image i with the
    IO+GPU stage of image i+1 on one worker thread (numba kernels and
    torch ops release the GIL); 'serial' alternates for profiling."""
    gp = GpuPipeline(model, sem_thr)
    t_io = t_gpu = t_cpu = 0.0
    results = []
    ex = ThreadPoolExecutor(max_workers=1) if pipeline == "threaded" else None
    pending = None  # (future, meta) when threaded
    t0 = perf_counter()
    try:
        for meta in metas:
            t = perf_counter()
            img = inference.load_rgb_cached(meta)
            depth = load_depth_array(Path(meta["dpath"]))
            t_io += perf_counter() - t

            t = perf_counter()
            payload = gp.gpu_stage(img, depth)
            t_gpu += perf_counter() - t
            del img

            if ex is not None:
                if pending is not None:
                    out, cpu_dt = pending[0].result()
                    t_cpu += cpu_dt
                    results += out[1]
                    if on_result is not None:
                        on_result(pending[1], out)
                pending = (ex.submit(_cpu_job, payload, meta["image_id"]), meta)
            else:
                t = perf_counter()
                out = cpu_stage(payload, meta["image_id"])
                t_cpu += perf_counter() - t
                results += out[1]
                if on_result is not None:
                    on_result(meta, out)
            del payload
        if pending is not None:
            out, cpu_dt = pending[0].result()
            t_cpu += cpu_dt
            results += out[1]
            if on_result is not None:
                on_result(pending[1], out)
    finally:
        if ex is not None:
            ex.shutdown(wait=True)
    wall = perf_counter() - t0
    lat = {
        "io": t_io / len(metas),
        "gpu_stage": t_gpu / len(metas),
        "cpu_stage": t_cpu / len(metas),
        "wall_total": wall / len(metas),
    }
    return results, lat


def drift_report(model, metas, sem_thr):
    """gpu_fast vs canonical chain on the given images: per-pixel mix
    rank equality, n_pred totals, and subset AP for both engines."""
    import torch

    from gisec.datasets.split import DATA

    gp = GpuPipeline(model, sem_thr)
    rank_eq = rank_ne = 0
    nrank_d = []
    n_pred_g = n_pred_c = 0
    res_g, res_c = [], []
    ann = DATA / "annotations" / f"instances_{metas[0].get('split', 'val')}.json"

    with torch.inference_mode():
        for meta in metas:
            img = inference.load_rgb_cached(meta)
            depth = load_depth_array(Path(meta["dpath"]))
            sem_logit, hm, off = inference._forward(model, img, depth)
            coords, cells = decode._cn_markers_with_cells(hm, off)
            peaks = decode._marker_peaks(hm, coords, cells)
            sem = decode.sem_binary(sem_logit, sem_thr)
            rank_d, _ = postproc_fast.load_or_compute_rank(
                meta["image_id"], depth, meta.get("split", "val")
            )
            rank_s, _ = postproc_fast.sem_logit_rank(sem_logit)
            rank_c, nrank_c = postproc_fast.mix_elevation_rank(rank_d, rank_s)
            _, coco_c = postproc_fast.process(
                meta["image_id"],
                coords,
                sem,
                depth,
                sem_logit,
                peaks,
                split=meta.get("split", "val"),
            )
            payload = gp.gpu_stage(img, depth)
            _, coco_g = cpu_stage(payload, meta["image_id"])
            eq = int((payload.rank == rank_c).sum())
            rank_eq += eq
            rank_ne += payload.rank.size - eq
            nrank_d.append(payload.nrank - nrank_c)
            n_pred_c += len(coco_c)
            n_pred_g += len(coco_g)
            res_c += coco_c
            res_g += coco_g
    img_ids = [m["image_id"] for m in metas]
    ev_c = evaluate_json(ann, res_c, img_ids=img_ids)
    ev_g = evaluate_json(ann, res_g, img_ids=img_ids)
    return {
        "n_images": len(metas),
        "mix_rank_pixel_equality": rank_eq / max(rank_eq + rank_ne, 1),
        "nrank_delta_abs_mean": float(np.mean(np.abs(nrank_d))),
        "n_pred_canonical": n_pred_c,
        "n_pred_gpu_fast": n_pred_g,
        "segm_AP_canonical": ev_c["segm/AP"],
        "segm_AP_gpu_fast": ev_g["segm/AP"],
    }


def main() -> None:
    import argparse
    import json

    from gisec.datasets.split import load_split
    from gisec.eval.diagnostics import rss_gb

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="eval_report_gpu.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--sem-thr", type=float, default=None)
    ap.add_argument("--pipeline", choices=("threaded", "serial"), default="threaded")
    ap.add_argument(
        "--drift",
        type=int,
        default=0,
        help=">0: canonical-vs-gpu_fast drift stats on the first N images",
    )
    args = ap.parse_args()
    if args.sem_thr is not None:
        decode.SEM_THR = args.sem_thr

    model = load_model(args.ckpt)
    metas, _ = load_split(args.split)
    if args.max_images:
        metas = metas[: args.max_images]
    inference.load_rgb_index(args.split)

    t0 = perf_counter()
    results, lat = run_batch(model, metas, args.sem_thr, args.pipeline)
    wall = perf_counter() - t0
    print(
        f"{len(metas)} imgs, wall {wall / len(metas) * 1000:.1f} ms/img "
        f"(io {lat['io'] * 1000:.1f} gpu {lat['gpu_stage'] * 1000:.1f} "
        f"cpu {lat['cpu_stage'] * 1000:.1f}) rss={rss_gb():.2f} GB",
        flush=True,
    )

    ann = DATA_ROOT / "annotations" / f"instances_{args.split}.json"
    img_ids = [m["image_id"] for m in metas]
    ev = evaluate_json(ann, results, img_ids=img_ids)
    row = {
        "tag": "centernet",
        "segm_AP": ev["segm/AP"],
        "segm_AP50": ev["segm/AP50"],
        "segm_AP75": ev["segm/AP75"],
        "bbox_AP": ev["bbox/AP"],
        "bbox_AP50": ev["bbox/AP50"],
        "bbox_AP75": ev["bbox/AP75"],
        "n_pred": len(results),
        "n_pred_per_img": len(results) / len(metas),
    }
    print(row, flush=True)
    report = {
        "caliber": "gpu_fast",
        "profile": "fast",
        "split": args.split,
        "pipeline": args.pipeline,
        "grid": [row],
        "latency_s_per_img": lat,
    }
    if args.drift:
        report["drift"] = drift_report(model, metas[: args.drift], args.sem_thr)
        print("drift", report["drift"], flush=True)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"rss_final={rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
