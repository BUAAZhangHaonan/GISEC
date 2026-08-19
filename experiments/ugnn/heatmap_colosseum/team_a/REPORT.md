# Team A — offline centroid cache + precomputed Gaussian stamp

## Method

One key observation: the mask centroid of a COCO annotation never
changes across epochs. So the expensive part of exp06
`make_heatmap` — `ann_to_mask` polygon decode + full-frame
`np.nonzero` per instance, every epoch, forever — is pure recomputation.

- **Offline (`build_cache.py`)**: run the exact reference centroid
  computation (`ann_to_mask` → `np.nonzero` → `mean` → `int(round)`)
  once for every train annotation, 64-way multiprocessing, saved to
  `centroids_train.npz` (ann_id → cy, cx; 7.5 MB, committed).
- **Hot path (`solution.py::build_heatmap`)**: dict lookup per ann,
  then stamp a precomputed 25×25 float32 Gaussian kernel (r=12,
  σ=4) with `np.maximum` on clipped slices. Kernel arithmetic mirrors
  the reference float32 ops exactly, so output is bit-identical.
  Uncached ann ids fall back to the lazy reference path (correctness
  preserved for val/test or edited annotations).

## Numbers (bench.py, seed=42, 64 imgs, 3488 instances)

- Correctness: **PASS** — max|Δ| = 0.000e+00 (gate 1e-3),
  support-set mismatch 0/64, worst centroid deviation 0.0000 px.
- Timing (3 rounds, median):
  - reference full (decode + heatmap): 384.3 ms/img
  - reference `make_heatmap` only (the 278 ms baseline): 275.5 ms/img
  - **team_a hot: 0.487 ms/img** — **565x vs 278.3 baseline**
    (789x vs full path)

## Amortized account

- One-off precompute: 310.3 s for 1,398,374 annotations
  (0.22 ms/ann, 64 procs).
- Amortized over one E9-style run (25,654 × 20 epochs = 513,080
  fetches): 310.3 / 513,080 = **0.605 ms/img** one-off + 0.487 hot
  ≈ **1.09 ms/img amortized** — still ~255x vs the baseline, and the
  precompute is already done (npz is committed, zero setup at
  integration time).

## Integration notes

- Call `solution.init_cache()` once (dataset `__init__`), then
  `build_heatmap(anns, (h, w))` in `__getitem__` — anns are the raw
  LiteCOCO dicts; no mask decode needed for the heatmap channel.
- The semantic mask channel still needs `ann_to_mask`, so the
  dataloader keeps decoding masks — the win is that heatmap no longer
  adds a second full nonzero scan per instance.
- Cache is split-specific; regenerate with `build_cache.py val` etc.
  if val heatmaps are needed.
- No GPU used. exp06-08 files untouched.
