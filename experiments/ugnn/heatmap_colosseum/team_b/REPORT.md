# Team B — center heatmap synthesis (RLE arithmetic, CPU)

## Method (one line)

Rasterize polygons to RLE with pycocotools exactly as
`ann_to_mask` does, then compute each instance's pixel count,
sum(y) and sum(x) straight from the column-major RLE runs via
closed-form prefix sums (no mask decode, no `np.nonzero`), round to
the integer centroid, and stamp a precomputed 25x25 float32
sigma=4 Gaussian kernel with pixelwise max.

## Why CPU, not GPU

The contract call site is a CPU dataloader worker (per-sample
`build_heatmap`). A torch/CUDA path would add an H2D copy of the
annotation-derived inputs plus a stream sync per sample (~0.5-1 ms
overhead alone, and workers cannot safely share a CUDA context),
against a ~18 ms pure-CPU job. Per-image GPU conv (impulse image +
fixed 25x25 kernel) also requires materializing the same centroids
first, so the CPU path is the honest winner; a batched GPU
interface was therefore not pursued. `build_heatmap_batch` is
provided for API completeness (loop over `build_heatmap`).

## Correctness (seed=42, 64 train images, 3488 instances)

| check | gate | result |
|---|---|---|
| max abs pixel diff vs exp06 `make_heatmap` | <= 1e-3 | **0.0** (bit-exact) |
| support set (hm > 0) equality | identical | 64/64 identical |
| per-instance centroid (int pair) | <= 0.25 px | 3488/3488 exactly equal |

Exactness argument: (a) same `frPyObjects`+`merge` rasterization as
the reference; (b) run-length sums are exact integers, so
`sum/n == np.mean(nonzero coords)` bit-for-bit and Python `round`
gives the same banker's-rounded int; (c) the kernel is evaluated
once with the reference's exact float32 expression
(`exp(-(dy^2+dx^2)/(2*sigma^2))`), and max-composition is
order-independent, so the output is bit-identical. Verified
empirically: the vectorized LEB128 counts decoder reproduces the
ground-truth run lengths on all sampled instances.

## Speed (hot, per-sample, 3 rounds, median)

| impl | ms/img | speedup |
|---|---|---|
| reference (exp06 make_heatmap + ann_to_mask) | 376.3 | 1x |
| team B | 18.0 | **20.9x** |

Note: the reference re-measured 376 ms/img on this box (the 278.3 ms
headline was measured elsewhere / different load; GPU0 training is
running now). Both arms timed identically in the same process.

Cost breakdown (profile, 64 imgs): polygon->RLE rasterization
~5.7 ms (irreducible C work), counts-string decode ~6.2 ms,
prefix-sum centroid math ~3.7 ms, kernel stamping ~1 ms.

## Amortized cost

One-time precompute = the 25x25 Gaussian kernel (< 0.1 ms, done on
first call). Amortized over 25654 x 20 = 513,080 samples:
0.1 ms / 513080 ~ 2e-7 ms/img -> **amortized ~18.0 ms/img**, i.e.
the amortization term is zero for practical purposes.
