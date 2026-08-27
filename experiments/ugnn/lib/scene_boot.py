"""Multiplicity-aware scene bootstrap + scene-disjoint threshold
cross-fitting for COCO AP (C2/M3 statistical repair, 2026-08-27).

Why
---
pycocotools COCOeval.evaluate() starts with
``p.imgIds = list(np.unique(p.imgIds))`` (cocoeval.py), silently
dropping duplicated image ids.  Scene bootstrap resamples *scenes*
with replacement, so a scene drawn twice must count its images
twice; expanding the draw into repeated imgIds and handing it to
COCOeval loses that multiplicity (210 scene slots collapse to ~133
effective scenes) and mis-sizes every scene CI computed that way
(the pre-2026-08-27 estimator in eval_scale / eval_pipeline /
eval_centernet / sweep_e20).

Fix (pycocotools untouched): run evaluate() ONCE over the unique
imgIds, cache the per-image evaluateImg entries (dtScores /
dtMatches / dtIgnore / gtIgnore), and redo the accumulate() math
with per-image integer weights.  Weighting image i detections by
m_i in the cumsum and its non-ignored GT count by m_i is
arithmetically identical to evaluating a literal m_i-fold copy of
the dataset (unit-tested: unit mult reproduces COCOeval to 1e-12;
doubled-image toys match hand math and literal duplication).
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
from pycocotools.cocoeval import COCOeval

DEFAULT_MAXDETS = (1, 10, 100)


def _eval_entries(coco_gt, coco_dt, img_ids, iou_type, max_dets, iou_thrs):
    """One quiet COCOeval.evaluate() pass over unique imgIds."""
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.params.imgIds = list(img_ids)
    ev.params.maxDets = sorted(max_dets)
    if iou_thrs is not None:
        ev.params.iouThrs = list(iou_thrs)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
    return ev


class ApWeighted:
    """AP(iouType, area=all, maxDets=max) under per-image integer
    multiplicities, from one cached COCOeval.evaluate() pass.

    ap(mult) equals COCOeval stats[0] on the dataset where image i is
    replaced by mult[i] identical copies (single-category datasets
    only; this repo has exactly one category)."""

    def __init__(
        self,
        coco_gt,
        coco_dt,
        img_ids,
        iou_type="segm",
        max_dets=DEFAULT_MAXDETS,
        iou_thrs=None,
    ):
        img_ids = sorted({int(i) for i in img_ids})
        ev = _eval_entries(coco_gt, coco_dt, img_ids, iou_type, max_dets, iou_thrs)
        p = ev.params
        if p.useCats and len(p.catIds) != 1:
            raise ValueError(
                f"ApWeighted: single-category datasets only (got {len(p.catIds)})"
            )
        i0 = len(p.imgIds)
        a0 = list(p.areaRngLbl).index("all")
        max_det = p.maxDets[-1]
        npig = np.zeros(i0, dtype=np.int64)
        n_det = np.zeros(i0, dtype=np.int64)
        score_parts, tp_parts, fp_parts = [], [], []
        # evalImgs flat layout [cat][areaRng][img]; k=0 (single cat), a='all'
        for pos, e in enumerate(ev.evalImgs[a0 * i0 : (a0 + 1) * i0]):
            if e is None:
                continue
            scores = np.asarray(e["dtScores"][:max_det], dtype=np.float64)
            dt_m = np.asarray(e["dtMatches"][:, :max_det]) != 0
            dt_ig = np.asarray(e["dtIgnore"][:, :max_det]) != 0
            npig[pos] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
            n_det[pos] = len(scores)
            score_parts.append(scores)
            tp_parts.append(dt_m & ~dt_ig)
            fp_parts.append(~dt_m & ~dt_ig)
        if score_parts:
            scores = np.concatenate(score_parts)
            tp = np.concatenate(tp_parts, axis=1)
            fp = np.concatenate(fp_parts, axis=1)
        else:
            scores = np.zeros(0)
            tp = np.zeros((len(p.iouThrs), 0), dtype=bool)
            fp = np.zeros((len(p.iouThrs), 0), dtype=bool)
        # one global stable score sort; per-image blocks are already
        # score-sorted by pycocotools, so this ordering is exactly what a
        # per-draw concatenate + mergesort would produce
        order = np.argsort(-scores, kind="mergesort")
        self.scores = scores[order]
        self.tp = tp[:, order]
        self.fp = fp[:, order]
        self.det_img = np.repeat(np.arange(i0, dtype=np.int64), n_det)[order]
        self.npig = npig
        self.iou_thrs = np.asarray(p.iouThrs, dtype=np.float64)
        self.rec_thrs = np.asarray(p.recThrs, dtype=np.float64)
        self.img_ids = img_ids

    def ap(self, mult) -> float:
        """stats[0]-equivalent AP for per-image multiplicities ``mult``
        (int array aligned to self.img_ids; 0 = image excluded)."""
        mult = np.asarray(mult, dtype=np.int64)
        if mult.shape != (len(self.img_ids),):
            raise ValueError(f"mult must have shape ({len(self.img_ids)},)")
        npig = float(mult @ self.npig)
        if npig == 0.0:
            return -1.0
        w = mult[self.det_img]
        keep = w > 0
        tp_cum = np.cumsum(self.tp[:, keep] * w[keep], axis=1).astype(np.float64)
        fp_cum = np.cumsum(self.fp[:, keep] * w[keep], axis=1).astype(np.float64)
        rc = tp_cum / npig
        pr = tp_cum / (fp_cum + tp_cum + np.spacing(1))
        pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
        n = rc.shape[1]
        aps = np.zeros(len(self.iou_thrs))
        for t in range(len(self.iou_thrs)):
            q = np.zeros(len(self.rec_thrs))
            inds = np.searchsorted(rc[t], self.rec_thrs, side="left")
            ok = inds < n
            q[ok] = pr[t, inds[ok]]
            aps[t] = q.mean()
        return float(aps.mean())


class SceneResampler:
    """Scene-cluster bootstrap multiplicities.

    draw() samples n_scenes scene slots with replacement and expands
    them to a per-image multiplicity vector (image i gets its scene
    slot count) -- the multiplicity the old imgIds-expansion
    estimators lost to COCOeval np.unique."""

    def __init__(self, img_ids, scene_keys):
        if len(img_ids) != len(scene_keys):
            raise ValueError("img_ids and scene_keys must be aligned")
        self.img_ids = [int(i) for i in img_ids]
        uniq = sorted(set(scene_keys))
        pos = {s: j for j, s in enumerate(uniq)}
        self.scenes = uniq
        self.scene_of_img = np.array([pos[s] for s in scene_keys], dtype=np.int64)
        self.n_scenes = len(uniq)

    def unit(self) -> np.ndarray:
        return np.ones(len(self.img_ids), dtype=np.int64)

    def draw(self, rng, scene_subset=None) -> np.ndarray:
        idx = (
            np.arange(self.n_scenes)
            if scene_subset is None
            else np.asarray(sorted(scene_subset), dtype=np.int64)
        )
        slots = rng.integers(0, len(idx), size=len(idx))
        counts = np.bincount(slots, minlength=len(idx))
        full = np.zeros(self.n_scenes, dtype=np.int64)
        full[idx] = counts
        return full[self.scene_of_img]


def _dist(vals) -> dict:
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "std": float(vals.std(ddof=1)),
    }


def scene_bootstrap_ci(acc: ApWeighted, resampler: SceneResampler, n_boot=2000, seed=0):
    """Multiplicity-aware scene bootstrap CI for one accumulator."""
    if acc.img_ids != resampler.img_ids:
        raise ValueError("ApWeighted / SceneResampler img_ids mismatch")
    rng = np.random.default_rng(seed)
    vals = np.array([acc.ap(resampler.draw(rng)) for _ in range(n_boot)])
    return {
        "mean": float(vals.mean()),
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
    }


def paired_scene_bootstrap(
    acc_a: ApWeighted,
    acc_b: ApWeighted,
    resampler: SceneResampler,
    n_boot=2000,
    seed=0,
    independent: bool = False,
):
    """Delta CI for two models over one shared image set.

    independent=False shares the multiplicity vector within each draw
    (paired design: the delta keeps only model-difference variance);
    independent=True draws separately per model (control that shows
    the pairing gain)."""
    if acc_a.img_ids != resampler.img_ids or acc_b.img_ids != resampler.img_ids:
        raise ValueError("img_ids mismatch across accumulator/resampler")
    rng = np.random.default_rng(seed)
    rng_b = rng if not independent else np.random.default_rng(seed + 1)
    a = np.empty(n_boot)
    b = np.empty(n_boot)
    for d in range(n_boot):
        mult = resampler.draw(rng)
        a[d] = acc_a.ap(mult)
        b[d] = acc_b.ap(mult if not independent else resampler.draw(rng_b))
    return {
        "a": _dist(a),
        "b": _dist(b),
        "delta": _dist(a - b),
        "independent": independent,
        "n_boot": n_boot,
        "seed": seed,
    }


def cross_fit_threshold(accs_by_thr, resampler: SceneResampler, n_boot=2000, seed=0):
    """M3 winner's-curse repair: scene-disjoint threshold cross-fitting.

    Scenes are split once into disjoint calibration / gating halves.
    Each draw resamples scenes within each half independently, the
    threshold is re-picked on the calibration replicate, and only that
    choice is scored on the gating replicate -- selection and CI never
    see the same scenes.

    accs_by_thr: {variant: {thr: ApWeighted}} over one shared image set."""
    for variant, thr_accs in accs_by_thr.items():
        for thr, acc in thr_accs.items():
            if acc.img_ids != resampler.img_ids:
                raise ValueError(f"img_ids mismatch: {variant}@{thr}")
    rng0 = np.random.default_rng(seed)
    perm = rng0.permutation(resampler.n_scenes)
    half = resampler.n_scenes // 2
    calib, gate = perm[:half], perm[half:]
    rng = np.random.default_rng(seed + 1)
    variants = list(accs_by_thr)
    gate_aps = {v: np.empty(n_boot) for v in variants}
    thr_pick = {v: [] for v in variants}
    for d in range(n_boot):
        cm = resampler.draw(rng, calib)
        gm = resampler.draw(rng, gate)
        for v in variants:
            thrs = list(accs_by_thr[v])
            star = max(thrs, key=lambda t: accs_by_thr[v][t].ap(cm))
            gate_aps[v][d] = accs_by_thr[v][star].ap(gm)
            thr_pick[v].append(star)
    out = {
        "n_boot": n_boot,
        "seed": seed,
        "n_scenes": resampler.n_scenes,
        "n_scenes_calib": len(calib),
        "n_scenes_gate": int(resampler.n_scenes - len(calib)),
        "variants": {},
        "in_sample_best_ap": {
            v: float(
                max(accs_by_thr[v][t].ap(resampler.unit()) for t in accs_by_thr[v])
            )
            for v in variants
        },
    }
    for i, v in enumerate(variants):
        hist: dict[str, int] = {}
        for t in thr_pick[v]:
            key = str(t)
            hist[key] = hist.get(key, 0) + 1
        entry = {"gate_ap": _dist(gate_aps[v]), "thr_hist": hist}
        if i > 0:
            entry["delta_vs_base"] = _dist(gate_aps[v] - gate_aps[variants[0]])
        out["variants"][v] = entry
    return out
