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


def _sample_pool(
    mask_h: torch.Tensor,
    mask_v: torch.Tensor,
    k: int,
    vals_h: torch.Tensor,
    vals_v: torch.Tensor,
) -> torch.Tensor:
    """Sample k edges from the pooled h/v candidate masks, return values.

    mask_h / mask_v: bool (H, W-1) / (H-1, W) edge validity for
    horizontal / vertical pairs respectively (the last col / row of the
    record bitmap is already dropped by the caller). vals_h / vals_v:
    per-edge values of the same shapes. Sampling is uniform without
    replacement, h-pool first
    then v-pool, indices sorted for determinism.
    """
    idx_h = torch.nonzero(mask_h.reshape(-1)).flatten()
    idx_v = torch.nonzero(mask_v.reshape(-1)).flatten()
    n = idx_h.numel() + idx_v.numel()
    if k >= n:
        sel = torch.arange(n, device=mask_h.device)
    else:
        sel, _ = torch.sort(torch.randperm(n, device=mask_h.device)[:k])
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
        n_pos = int(ph.sum()) + int(pv.sum())
        n_neg = int((neg_h[b, :, :-1] > 0).sum()) + int((neg_v[b, :-1, :] > 0).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        k = min(max_pairs, n_pos, n_neg)
        gp_l.append(_sample_pool(ph, pv, k, gh[b], gv[b]))
        dp_l.append(_sample_pool(ph, pv, k, dh[b], dv[b]))
        mp_l.append(_sample_pool(ph, pv, k, zh_min[b], zv_min[b]))
        gn_l.append(
            _sample_pool(neg_h[b, :, :-1] > 0, neg_v[b, :-1, :] > 0, k, gh[b], gv[b])
        )

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
