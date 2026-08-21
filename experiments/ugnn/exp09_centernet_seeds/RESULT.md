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
