# Heatmap Colosseum — Judge Verdict (2026-08-19)

Independent rerun under uniform conditions: seed=42, 64 train images
(3488 instances) + 32 val images (1687 instances), fresh python
process per timing cell, 3-round medians, reference rebuilt in-place
from exp06 `make_heatmap` + `ann_to_mask`. Judge scripts:
`judge/rerun_bench.py`, `judge/integration_bench.py`. Team code was
not modified.

## 1. Correctness rerun (one-vote veto)

| split | impl | max abs delta | support mismatch | centroid mismatch |
|-------|------|---------------|------------------|-------------------|
| train | ref | 0 | 0/64 | 0 |
| train | team_a | 0 | 0/64 | 0 |
| train | team_b | 0 | 0/64 | 0 |
| train | team_c | 0 | 0/64 | 0 |
| val | ref | 0 | 0/32 | 0 |
| val | team_a | **1.0** | **32/32** | 0 |
| val | team_b | 0 | 0/32 | 0 |
| val | team_c | 0 | 0/32 | 0 |

**team_a FAILS val.** Root cause: train and val annotation id spaces
in `20260318_1K_32254` overlap almost completely (val has 181,712
anns, 181,691 of which collide with train ids 1..1.4M). A's
train-only npz therefore returns *train* centroids for val anns —
the "lazy fallback" never fires. Every val heatmap is silently
wrong (stamps at foreign locations). The centroid column shows 0
because the judge's centroid probe for A calls its uncached
`compute_centroid`; the heatmap path itself is poisoned. A's REPORT
claim ("correctness preserved for val/test") is false.

team_b and team_c are bit-exact on both splits, cold and warm.
Multi-worker (16 workers) output also bit-exact vs single-process
reference for a/b/c on the train split.

## 2. Speed rerun (ms/img, same 64 train images)

| impl | cold | warm (median) | 20-epoch amortized |
|------|------|---------------|--------------------|
| ref | 398.3 | 407.0 | 407.0 |
| team_a (init 858 ms npz load) | 0.472 | 0.447 | 0.470 |
| team_b | 18.44 | 18.12 | 18.32 |
| team_c | 13.73 (JIT cache hit) / 26.4 (fresh JIT) | 0.296 | 0.876 |

C's amortized number is higher than its claimed 0.495 because the
judge's epoch-1 in a fresh process pays first-call overhead
(13.7 ms/img vs claimed 5.56); epoch 2+ matches (0.248 ms/img).

## 3. Integration bench (real Dataset+DataLoader, E8 recipe)

batch 8, num_workers 16, pin_memory, persistent_workers, 64-image
subset, data loop only (no model). Floor = "none" (no heatmap).

| impl | epoch1 ms/step | epoch2 | epoch3 | batches/s (warm) |
|------|----------------|--------|--------|------------------|
| none (floor) | 1270 | 464 | 441 | 2.27 |
| ref | 1388 | 826 | 740 | 1.35 |
| team_a | 1055 | 462 | 479 | 2.09 |
| team_b | 992 | 499 | 464 | 2.15 |
| team_c | 943 | 415 | 400 | 2.50 |

Reading: at 16 workers the ~18 ms/img stateless cost of B is almost
entirely hidden behind image/depth IO (b and a converge to the
floor within noise); the reference's 400 ms/img is not — it costs
~300 ms/step even warm. C is the only impl measurably *below* the
floor runs, i.e. its warm path is effectively free. a/b/c epoch-1
differences are within run-to-run noise (GPU0 training's own
dataloader shares the box); the stable signal is epochs 2-3.

## 4. Ranking

1. **team_c** — correct on both splits, fastest warm (0.30 ms) and
   fastest integration (2.50 b/s, only impl beating the floor),
   zero artifacts, self-warming cache that works for any split.
2. **team_b** — correct, stateless, zero deps, 22x speedup that
   already disappears into the IO floor at 16 workers. The safe,
   boring, portable choice.
3. **team_a** — fastest train-side number, but the val failure is a
   silent wrong-label bug of the worst kind (plausible heatmaps at
   wrong centroids). Vetoed.

Caveat on C recorded by the judge: its numba `cache=True` artifacts
pickle the defining module *by name* (`solution`); importing the
file under any other module name makes the JIT cache unloadable
(ModuleNotFoundError observed when the judge first imported it as
`team_c.solution`). Integration must ship it as a top-level module
with a stable name.

## 5. Verdict

Team C wins. A bet on precomputation and lost correctness where the
id-space assumption broke; B bet on statelessness and won safety
but leaves 18 ms/img on the table in single-worker contexts; C's
"compute once, remember in-process" gets A's warm speed with B's
safety — no artifact, no split coupling, and the cold path is the
same exact kernel as the warm path. The numba dependency and the
module-name-anchored JIT cache are real but one-time costs.

## 6. E9 integration recommendation

Adopt team_c. Patch sketch (do NOT inline numba caches from the
colosseum dir; copy the file so the module name is stable):

1. `cp experiments/ugnn/heatmap_colosseum/team_c/solution.py \
      experiments/ugnn/exp08_scale_32254/heatmap_fast.py`
   (top-level module — required for numba's cache to load).
2. In `train_scale.py`:
   `from heatmap_fast import build_heatmap` and in
   `CenterDataset.__getitem__` replace the `insts` accumulation +
   `make_heatmap(insts, *img.shape[:2])` tail with
   `hm = build_heatmap(anns, (info["height"], info["width"]))`;
   keep the `ann_to_mask` loop only for the `gt` mask channel.
3. Keep `num_workers=16, persistent_workers=True` (E8 recipe) so
   the in-process centroid cache survives across epochs; with
   persistent workers each of the 16 workers warms its own cache in
   epoch 1 — that is the 20-epoch amortization measured above.
4. Pin `numba` in the env (0.67.0 present). Expected E9 effect:
   data loop goes from ~826 -> ~415 ms/step warm; with the model in
   the loop the heatmap cost drops out of the critical path
   entirely.
5. If the numba dependency is refused, fall back to team_b (same
   two-line integration, zero deps, still converges to the IO
   floor at 16 workers).

team_a must not be integrated as-is; fixing it requires per-split
npz keys (id -> (split, centroid)) or a (img_id, ann_id) composite
key, at which point C already provides the same warm speed without
the artifact.
