# Team C — algorithmic reduction route

## Result

- **Hot path: 130.5 ms/img median** (bench/timing.py, warmup 10 + 50 imgs,
  p90 158.9, mean 133.1). Reference on same machine/protocol: 673.75 (judge)
  / 687.3 (self) → **~5.1x faster**.
- Output is **byte-identical to the reference** on all 250 images
  (counts strings and scores equal), not just within thresholds.
- Correctness (bench/correctness.py, unit c2-team-c-corr2):
  - C1 zero-dev frac = 1.0000, max_dev = 0 → PASS
  - C2 mean IoU = 1.000000 → PASS
  - C3 probe AP ref 0.6909 vs usr 0.6909, |d| = 0 → PASS
  - C4 determinism → PASS

## Win-by-win contribution (per-step medians on this package)

| win | before | after | saves |
|---|---|---|---|
| elevation precompute (cache hit) | ~68 ms | ~3 ms (subsample-hash 0.5 + npy load 1.5) | ~65 |
| merge postprocess vectorized | 70.5 ms | 19.9 ms | ~50 |
| instance extract: bincount + find_objects (no per-label full-image bool masks) | 48.6 ms | ~11 ms | ~38 |
| to_results RLE from cropped patches (column-run counts in numpy, offset into full-image RLE, compressed via frPyObjects) | 290.2 ms | ~14 ms | ~276 |
| watershed, cn_markers | unchanged (exact) | 80 + 2 ms | 0 |

Baseline steps from PROBLEM §7 (uniform val); in-package reference is harder
(high-instance images), hence 687 total.

## Precompute (C5)

`precompute.py` stores sobel-magnitude(depth) per dump as exact float64
`.npy` under `team_c/cache/elev/`, 250 files, 26 s total ≈ **104 ms/img
amortized** (dominated by npz load + sobel; one-time, offline). Key =
`val_{image_id}_{md5(shape + depth[::4,::4])}` — keyed by
(split, image_id) plus a content hash of the depth input, so a depth or
split change re-computes instead of colliding. Hot path falls back to
inline sobel on any miss, so the entry runs on ANY val image. No model
output is ever cached.

## Rejected routes (measured, not assumed)

- **512-res watershed + 2x label upsample**: zero-dev count frac 0.044
  (C1 needs ≥0.95), max dev 18, mean IoU 0.805 (C2 needs ≥0.995). Fatal —
  instances are small (~50 px wide), 2 px boundary shifts destroy them.
- **to_results area-tail pruning**: any dropped instance changes n_pred,
  C1 requires exact count on ≥95%; rejected without further tuning since
  the patch-RLE route already removed 95% of to_results cost.

## Known risks

- merge reimplementation is semantics-equivalent, validated array-equal on
  all 250 images (unit c2-team-c-mergeval5); the reference's np.roll
  wraparound quirk is unreachable for connected regions < 32 px.
- Exactness depends on scipy/skimage versions matching the reference env
  (both import the same installed code, so judge rerun is consistent).
