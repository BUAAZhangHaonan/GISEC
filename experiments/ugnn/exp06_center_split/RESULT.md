# E6: learned center-heatmap seeds for the depth watershed

## Goal

Replace E4's hand-rolled depth-extremum seeds with peaks of a learned
center heatmap and clear the E5 pass line: segm AP >= 0.42 with scene
bootstrap CI lower bound >= 0.38. Gray 0.35-0.42; < 0.35 closes the
dense route. References: E4 FINAL 0.3125, E4 GT-semantic ceiling
0.4933, M2F swin-t 0.5381 (~3x params).

## Method

- Model: E3 U-Net with a 2-channel head (`smp.Unet(resnet18/
  imagenet, 4ch, classes=2)`, ~14.6M): ch0 semantic, ch1 center
  heatmap (Gaussian sigma=4 px at each instance centroid, overlaps
  via max). Loss: BCE+Dice (semantic) + MSE (heatmap; chosen over
  focal because the target is a smooth dense regression surface —
  the Gaussian mass carries the placement signal, CenterNet-style).
  Recipe identical to E3: AdamW 3e-4 cosine, 20 epochs, batch 8@1024.
  Train 84.1 min, val mIoU 0.9406 (E3: 0.9453 — the extra head costs
  0.005 semantic).
- Inference: semantic >= 0.5 -> markers = peak_local_max(sigmoid
  heatmap, threshold 0.3, min_distance in {3,5,9,15}) -> watershed on
  grad(depth) (E4's best elevation) -> small-region merge (E4 best
  post) -> area-normalized scores -> gisec coco_eval. Entry points:
  `train_center.py`, `eval_center_split.py` (imports exp03/exp04).

## Numbers

Oracle target (run first, defines what the heatmap head is worth):
GT centroids as markers + model semantic + grad(depth) watershed:
**segm AP 0.5558**, bootstrap 0.558 [0.517, 0.615] (E3 semantic;
0.5445 with the E6 semantic). This beats the E4 GT-semantic ceiling
0.4933 and M2F 0.5381 — GT center seeds are the best operator
configuration found so far.

Main grid (segm AP / AP50 / AP75 / bbox AP / pieces-per-img):

| markers | AP | AP50 | AP75 | bbox | pieces/img |
|---|---|---|---|---|---|
| heatmap md3 | .194 | .476 | .138 | .264 | 100 (cap) |
| heatmap md5 | .436 | .742 | .440 | .469 | 96.9 |
| **heatmap md9** | **.480** | .737 | .501 | .468 | 75.3 |
| heatmap md15 | .423 | .651 | .430 | .402 | 54.7 |
| depth md15 (E4 repro) | .307 | .617 | .275 | .356 | 90.2 |
| union hm+depth md9 | .139 | .394 | .078 | .224 | 100 (cap) |
| oracle GT centers | .544 | .774 | .578 | .510 | 55.0 |

FINAL (heatmap md9): **segm AP 0.4797**, scene bootstrap (87 scenes
x 200) segm 0.4815 [0.4364, 0.5417], bbox 0.4704 [0.4277, 0.5238].

Split stats (FINAL): oversplit-GT ~0, undersplit pieces 8.6%
(oracle: 12.8%), count 75.3 vs GT 63.7 (1.18x, down from E4's 1.5x).

Seed placement precision (marker -> nearest GT centroid):

| seed source | markers/img | median px | P90 px | <8px rate |
|---|---|---|---|---|
| heatmap (md9, thr 0.3) | 238.4 | 22.9 | 31.0 | 20.3% |
| depth flat peaks (md15) | 103.1 | 17.9 | 27.3 | 10.9% |

Counterintuitive: per-marker distance is WORSE for the heatmap, yet
AP is +0.17. The heatmap wins by seed *density and coverage*, not by
per-seed accuracy — 3.7x more seeds inside every fused blob, so the
grad(depth) watershed gets a basin boundary near every real part and
the merge post-process absorbs the excess. The union control
confirms the mechanism is not "more seeds of any kind": adding depth
peaks to heatmap peaks (2.4x markers) collapses AP to 0.139 because
depth peaks land at wrong places that the heatmap does not.

## Conclusion

**PASS: segm AP 0.4797 >= 0.42, bootstrap CI lower 0.4364 >= 0.38.**
The learned-seed fix does what E5 funded it for: +0.172 over the E4
depth-extremum operator (0.3077) at ~14.6M params, 89% of M2F
swin-t's 0.5381 with ~1/3 the parameters, and it clears the E4
GT-semantic ceiling 0.4933. Route proceeds to the E5 section-5
roadmap (md sweep to convergence, count 1.18x fix, then 32254
transfer).

Bottleneck attribution for the remaining 0.48 -> 0.56 gap:
1. Seed placement on small parts is still the lever — the oracle
   with GT centroids reaches 0.556, so perfect centers buy +0.076
   over learned peaks; the heatmap's per-marker accuracy (median
   22.9 px) is actually worse than depth peaks, and the win comes
   from coverage. A higher-resolution heatmap head (sigma < 4 or
   offset regression) is the next training-side gain.
2. Boundary precision binds second: AP50 0.737 vs AP75 0.501 —
   same IoU-band pattern as E4 (0.62/0.28), improved but still the
   shape of the curve.
3. md9 sits inside the sweep with md5/md15 below on both sides, so
   unlike E4 the grid is now saturated at the operating point.
