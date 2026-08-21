# E9 RESULT: CenterNet seeds close the gap to oracle

## Goal
Fix the E8 seed-placement bottleneck with a CenterNet seed head (stride-4 adaptive-sigma focal heatmap + offset regression + peak NMS), as a single-variable change against the E8 seed head.

## Method
CenterNet-style seed head reading the shared decoder via AvgPool(4); adaptive per-instance sigma, focal loss, offset regression, peak NMS at inference. Training recipe (3-stage resume: train -> train2 -> train3 with E9b compact GT records) in STATUS.md.

## Numbers (eval_report.json, 3276-image val)
- FINAL (centernet): segm AP 0.7254 (AP50 0.8518 / AP75 0.7319), bbox AP 0.6476
- Oracle GT centers: segm AP 0.7359 -> FINAL reaches 98.6% of oracle
- Bootstrap (210 scenes, 100x): segm 0.7261 [0.7075, 0.7475]
- Seed precision: median 2.35 px, p90 4.98 px, <8px rate 96.3%; 56.2 markers/img vs 55.5 GT/img
- Latency: 0.47 s/img full pipeline (forward 0.040 s)
- Undersplit piece rate 9.4% (oracle 9.0%)

## Verdict
PASS on all three pre-registered lines: seed median <15px (2.35), <8px rate >30% (96.3%), segm AP >=0.60 (0.7254).

vs E8 (0.4815): +0.244 segm AP. The seed-placement problem is SOLVED: the FINAL-vs-oracle gap is only 0.010, so there is essentially no headroom left in seed placement.

Caveat: oracle itself dropped from E8 0.7952 to 0.7359. The third head squeezes semantic capacity (val mIoU 0.9989 -> 0.9968) and this checkpoint knife/semantic ceiling binds. The bottleneck is now boundary-knife precision and semantics; against the 90.63 M2F ceiling there are still ~18 points to go.

## Post-processing colosseum (2026-08-21)

Champion: team_b (numba CPU post-processing), see
`../postproc_colosseum/ARENA.md` — 69.07 ms/img single-process judge
timing (9.8x over the 673.75 ms reference), all correctness gates
passed including unseen-image cache-miss probe. Rule note: the GPU
hot-path ban was lifted post-verdict (user directive); team_b stays
champion on pure numbers (team_a GPU 95.88 ms) and production fit.

Integrated as `postproc_fast.py` (module name frozen: numba njit
cache pickles by module name). Full-val rank cache lives in
`runs/postproc_cache/val` (13 GB, gitignored via `runs/`;
`GISEC_POSTPROC_CACHE` overrides the root). Build with
`python postproc_fast.py` before any full run. Cold-cache note: in a
fresh process the first image costs ~0.5 s (numba JIT), steady-state
cache-miss ~+0.2 s/img until the rank cache exists.

Full-val revalidation (3276 imgs, runs/best.pth unchanged,
eval_report_postproc_fast.json, unit c2-integ-eval):
- FINAL segm AP 0.72541 vs 0.7254 baseline — identical to 5 decimals
  (|dAP| = 0.00001); oracle 0.73586, n_pred 166788 all unchanged.
- Bootstrap (210 scenes, 100x): segm 0.7261 [0.7075, 0.7475] — same CI.
- Wall 0.299 s/img vs 0.470 before (1.57x end-to-end); forward 0.042 s.
  Pipeline stage ~16.3 min for 3276 imgs; bootstrap dominates the rest.
- Determinism: 200-img double run (two fresh processes), per-image
  CRC32 of instances bitwise identical (determinism_check.py).
