"""E23 seam ranking loss: supervise the deployed elevation's derivative.

Deployment elevates seeds by rank(sobel(sem logit)); E15 forensics
showed union supervision welds touching instances into one confident
blob (self coverage median 0.9982, contact local precision 0.3535).
L_seam asks the FULL-RESOLUTION semantic logit z for a rankable
margin across foreground-foreground contact seams without giving up
foreground coverage:

  E+ = adjacent pixel pairs across a seam (different instances)
  E- = adjacent pairs of ONE instance inside the E17 band rim
  L   = w-mean softplus(margin + g- - g+),  g = |z_u - z_v|
      + floor_w * mean softplus(tau_fg - min(z_u, z_v))  on E+

Depth-flat weighting: w_e = 1 / (1 + |grad d|_e / s), s = batch
median of |grad d| over the sampled E+ edges, w normalised to mean 1
-- flat contacts are exactly where the depth elevation cannot cut, so
their seam logit gradient is upweighted. The floor blocks the
degenerate solution "push one side to background to fabricate a
seam".

Pure torch (CPU-testable), no repo imports. seam_edges_from_idmap is
the single source of truth for the record geometry, shared by
build_seam_records.py and the tests.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_EMPTY_STATS = {
    "n_pos": 0,
    "n_neg": 0,
    "rank": 0.0,
    "floor": 0.0,
    "g_plus": 0.0,
    "g_minus": 0.0,
    "s_depth": 0.0,
    "w_min": 0.0,
    "w_max": 0.0,
}


def seam_edges_from_idmap(
    id_map: np.ndarray, band: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(seam_h, seam_v, neg_h, neg_v) bool (H, W) from an instance id map.

    id_map: int (H, W), 0 = background, >0 = instance label (annotation
    index, NOT connected component -- 29.2% of GT masks are naturally
    multi-component). band: bool (H, W), the E17 union rim.

    seam_h[u, v] = 1 iff (u, v) and (u, v+1) are both foreground and
    their ids differ (last column always 0); seam_v analog with the
    last row 0. neg_h / neg_v mark same-id adjacent pairs with BOTH
    endpoints inside the band (the E- candidate pool).
    """
    fg = id_map > 0
    same_h = id_map[:, :-1] == id_map[:, 1:]
    fg_h = fg[:, :-1] & fg[:, 1:]
    seam_h = np.zeros(id_map.shape, dtype=bool)
    neg_h = np.zeros(id_map.shape, dtype=bool)
    seam_h[:, :-1] = fg_h & ~same_h
    neg_h[:, :-1] = fg_h & same_h & band[:, :-1] & band[:, 1:]
    same_v = id_map[:-1, :] == id_map[1:, :]
    fg_v = fg[:-1, :] & fg[1:, :]
    seam_v = np.zeros(id_map.shape, dtype=bool)
    neg_v = np.zeros(id_map.shape, dtype=bool)
    seam_v[:-1, :] = fg_v & ~same_v
    neg_v[:-1, :] = fg_v & same_v & band[:-1, :] & band[1:, :]
    return seam_h, seam_v, neg_h, neg_v


def _edge_index_pool(
    mask_h: torch.Tensor, mask_v: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw the pooled h/v edge positions ONCE.

    mask_h / mask_v: bool (H, W-1) / (H-1, W) edge validity for
    horizontal / vertical pairs respectively (the last col / row of the
    record bitmap is already dropped by the caller). Sampling is
    uniform without replacement over the concatenated h-then-v nonzero
    index list, indices sorted for determinism; all edges are kept
    when the pool fits (k >= n).

    Returns (sel, idx_h, idx_v). Gathering ANY set of per-edge
    quantities with the same sel through _gather_edges keeps the i-th
    element of every gathered tensor on the same physical edge -- the
    caller MUST sample once and gather many (per-quantity independent
    sampling would decorrelate e.g. w(d+) from the g+ edge it must
    weight).
    """
    idx_h = torch.nonzero(mask_h.reshape(-1)).flatten()
    idx_v = torch.nonzero(mask_v.reshape(-1)).flatten()
    n = idx_h.numel() + idx_v.numel()
    if k >= n:
        sel = torch.arange(n, device=mask_h.device)
    else:
        sel, _ = torch.sort(torch.randperm(n, device=mask_h.device)[:k])
    return sel, idx_h, idx_v


def _gather_edges(
    sel: torch.Tensor,
    idx_h: torch.Tensor,
    idx_v: torch.Tensor,
    vals_h: torch.Tensor,
    vals_v: torch.Tensor,
) -> torch.Tensor:
    """Gather per-edge values at the pooled positions sel (see _edge_index_pool)."""
    fh = vals_h.reshape(-1)
    fv = vals_v.reshape(-1)
    h_sel = sel[sel < idx_h.numel()]
    v_sel = sel[sel >= idx_h.numel()] - idx_h.numel()
    return torch.cat([fh[idx_h[h_sel]], fv[idx_v[v_sel]]])


def seam_rank_loss(
    sem_logits: torch.Tensor,
    seam_h: torch.Tensor,
    seam_v: torch.Tensor,
    neg_h: torch.Tensor,
    neg_v: torch.Tensor,
    depth: torch.Tensor,
    *,
    margin: float = 1.0,
    tau_fg: float = 2.0,
    floor_w: float = 0.25,
    max_pairs: int = 4096,
) -> tuple[torch.Tensor, dict]:
    """E23 L_seam. Returns (loss, stats).

    sem_logits: (B, 1, H, W) full-resolution semantic logits.
    seam_h/seam_v/neg_h/neg_v: (B, H, W) record bitmaps (any numeric;
    nonzero = valid edge). depth: (B, 1, H, W) normalized depth, the
    exact channel the model consumes. Per image at most max_pairs E+
    edges are drawn (all if fewer); E- is drawn at equal count from
    that image's neg pool (fewer only if the pool is smaller).
    """
    z = sem_logits[:, 0]
    gh = (z[:, :, :-1] - z[:, :, 1:]).abs()
    gv = (z[:, :-1, :] - z[:, 1:, :]).abs()
    dh = (depth[:, 0, :, :-1] - depth[:, 0, :, 1:]).abs()
    dv = (depth[:, 0, :-1, :] - depth[:, 0, 1:, :]).abs()
    zh_min = torch.minimum(z[:, :, :-1], z[:, :, 1:])
    zv_min = torch.minimum(z[:, :-1, :], z[:, 1:, :])

    gp_l, dp_l, mp_l, gn_l = [], [], [], []
    for b in range(z.shape[0]):
        ph = seam_h[b, :, :-1] > 0
        pv = seam_v[b, :-1, :] > 0
        nh = neg_h[b, :, :-1] > 0
        nv = neg_v[b, :-1, :] > 0
        n_pos = int(ph.sum()) + int(pv.sum())
        n_neg = int(nh.sum()) + int(nv.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        k = min(max_pairs, n_pos, n_neg)
        # positive edges: draw the pooled indices ONCE, then gather every
        # per-edge quantity with the same sel so g+/d+/z_min[i] all
        # describe edge i (independent per-quantity sampling decorrelates
        # the depth-flat weight from the edge it must weight)
        sel, idx_h, idx_v = _edge_index_pool(ph, pv, k)
        gp_l.append(_gather_edges(sel, idx_h, idx_v, gh[b], gv[b]))
        dp_l.append(_gather_edges(sel, idx_h, idx_v, dh[b], dv[b]))
        mp_l.append(_gather_edges(sel, idx_h, idx_v, zh_min[b], zv_min[b]))
        # negative pool: independent unbiased Monte Carlo, own sampling
        nsel, nidx_h, nidx_v = _edge_index_pool(nh, nv, k)
        gn_l.append(_gather_edges(nsel, nidx_h, nidx_v, gh[b], gv[b]))

    if not gp_l:
        zero = sem_logits.sum() * 0.0
        return zero, dict(_EMPTY_STATS)

    g_plus = torch.cat(gp_l)
    d_plus = torch.cat(dp_l)
    z_min = torch.cat(mp_l)
    g_minus = torch.cat(gn_l)
    s = d_plus.median().clamp(min=1e-6)
    w = 1.0 / (1.0 + d_plus / s)
    w = w / w.mean().clamp(min=1e-8)
    rank = (w * F.softplus(margin + g_minus - g_plus)).mean()
    floor = F.softplus(tau_fg - z_min).mean()
    loss = rank + floor_w * floor
    stats = {
        "n_pos": int(g_plus.numel()),
        "n_neg": int(g_minus.numel()),
        "rank": float(rank.detach()),
        "floor": float(floor.detach()),
        "g_plus": float(g_plus.mean().detach()),
        "g_minus": float(g_minus.mean().detach()),
        "s_depth": float(s.detach()),
        "w_min": float(w.min().detach()),
        "w_max": float(w.max().detach()),
    }
    return loss, stats
