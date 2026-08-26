# Team A (GPU route) — final report

**Self-timed (bench/timing.py, warmup 10 + 50 imgs): median 96.1 ms/img,
p90 113.2, mean 96.7** vs judge baseline 673.75 (7.0x). Single process,
GPU 0 only.

## Per-step medians (steady state, 1024x1024)
| step | ms | note |
|---|---|---|
| cn_markers (CPU scipy) | 2.0 | reference-identical decode |
| elevation | ~5 | GPU separable sobel + D2H (gx,gy f32) + np.hypot on CPU |
| watershed (numba bucket queue) | 25-55 | radix rank (~10) + priority flood |
| merge (vectorized) | ~20 | |
| extract+RLE (vectorized) | ~19 | column-run pass + numba LEB128 |
| total | ~96 | |

## Correctness (bench/correctness.py, unit c2-team-a12)
- C1 zero-dev frac 0.9840, max dev 1 -> PASS
- C2 mean IoU 0.998384 -> PASS
- C3 probe AP ref 0.6909 / usr 0.6911, |d|=0.00012 -> PASS
- C4 determinism (two runs, json-identical) -> PASS

## Algorithm
1. **Elevation**: scipy sobel = derivative-then-smooth separable pass;
   replicated on GPU with the same float32 op order -> bit-identical
   gx/gy (np.array_equal vs ndi.sobel == True on all dumps). hypot on
   CPU keeps the float32 elevation bitwise identical to reference.
2. **Watershed**: skimage pops a (value, age) heap; the pushed key is
   the minimax max(parent_key, elev[child]), which is monotone
   non-decreasing, so rank-buckets + FIFO inside each bucket reproduce
   the exact pop order in O(N). Exact pruning: skip an entry whose key
   is not strictly below the best key already queued for that pixel.
   Stable 16-bit radix sort of in-mask keys only (~35% of pixels).
   Label agreement vs skimage: 99.8-100% per image (residual: FIFO age
   vs bucket order on exact-tie plateaus; C2 still 0.998).
   GPU relax variants (minimax / +hop-distance / top-2 lexicographic
   profile) were tried first and all fail C2 badly: 32% of in-mask
   pixels share exactly equal float32 elevation, so tie semantics
   dominate; they cannot be batched on GPU without the sequential age.
3. **RLE**: one column-run pass over the label image; per-label counts
   built vectorized, COCO LEB128 varint in numba (byte-identical to
   pycocotools encode, verified on random masks), counts as utf-8 str
   (same as gisec encode_binary_mask).

## H2D/D2H share
depth H2D 4 MB + gx/gy D2H 8 MB ~= 1.5-3 ms of the 96 (~3%). In
production the forward pass already holds depth/sem/hm on GPU, so the
H2D disappears; only gx/gy D2H (or doing hypot+rank on GPU) remains.

## Known risks
- Residual 0.02-0.2% pixel deviation vs skimage on tie-heavy images
  (C1 allows 1-instance dev; observed max dev = 1 on 4/250 images).
- One-GPU assumption; torch.cuda required (no CPU fallback).
- Numba JIT warmup ~10 s on first call (excluded by the warmup protocol).
