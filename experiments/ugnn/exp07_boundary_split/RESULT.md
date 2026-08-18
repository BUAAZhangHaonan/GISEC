# E7: learned instance-boundary elevation for the watershed

## Goal

E6's bottleneck ranking puts seed placement first (+0.076 to the
GT-center oracle) and boundary precision second (AP50 0.737 vs AP75
0.501 — the knife, grad(depth), is imprecise). E7 changes exactly
one variable: replace grad(depth) as the watershed elevation with a
learned instance-boundary map. Markers stay at E6 FINAL (heatmap
peaks, threshold 0.3, min_distance 9) and the merge post-process is
unchanged. Pass: segm AP >= 0.50 AND AP75 >= 0.60; gray 0.45-0.50
(best of E6/E7 goes to 32254); fail < 0.45 or no AP75 gain.

## Method

- Model: E6 U-Net with a 3rd output channel — instance boundary =
  union of per-instance 1-px contours (cv2.findContours external,
  CHAIN_APPROX_NONE). Adjacent instances each contribute a contour
  column on a contact seam, which is the knife to learn. Loss:
  BCE+Dice (sem) + MSE (heatmap) + BCE pos_weight=90 on boundary
  (inverse frequency at the measured 1.1% boundary-pixel rate, on
  50 val imgs; not tuned). Recipe identical to E6: AdamW 3e-4
  cosine, 20 epochs, batch 8@1024. Train 103.2 min, val mIoU 0.9402
  (E6: 0.9406 — the third head is semantically free).
- Elevation configs, same markers/post everywhere (watershed floods
  low->high; both maps are high on boundaries, no inversion):
  depth = grad(depth) per-image /max (E6 operator, new ckpt);
  bnd = boundary prob /max; fuse = max(bnd, grad /max).
- Entry points: `train_boundary.py`, `eval_boundary_split.py`
  (imports exp03/04/06; runs/ gitignored).

## Numbers

Grid (149 val imgs, markers = heatmap md9, merge post):

| elevation | AP | AP50 | AP75 | APs | APm | pieces/img | undersplit |
|---|---|---|---|---|---|---|---|
| depth (E6 op) | .4225 | .731 | .418 | .172 | .519 | 92.6 | 4.8% |
| bnd | .4521 | .746 | .459 | .143 | .564 | 79.4 | 7.2% |
| fuse | .4583 | .748 | .472 | .145 | .572 | 79.5 | 7.1% |

FINAL (fuse): segm AP 0.4583, scene bootstrap (87x200)
0.4589 [0.4223, 0.5098], bbox 0.4501 [0.4152, 0.4939].

Boundary quality vs the GT contour union (positives + 200k sampled
negatives/img; contact seam = touching-pair +-2px band):

| map | ROC-AUC | AP | seam AUC | seam recall@p90 |
|---|---|---|---|---|
| grad(depth) | 0.893 | 0.381 | 0.678 | 0.475 |
| learned bnd | 0.987 | 0.718 | 0.723 | 0.631 |

## Conclusion

**GRAY (0.45 <= 0.4583 < 0.50), and E6 wins the gray tie-break** —
E6's 0.4797 [0.436, 0.542] beats E7's 0.4583 [0.422, 0.510] with
overlapping CIs, and AP75 0.472 does not clear E6's 0.501. Per the
pre-registered rule, the **E6 configuration** (2-head model, heatmap
md9 seeds, grad(depth) elevation, merge post) is what goes to 32254;
the boundary head is not carried forward.

Two findings explain the outcome:

1. The learned knife is genuinely better than the depth knife —
   within this checkpoint, switching elevation depth -> fuse buys
   +0.036 AP and +0.055 AP75 with identical markers. The boundary
   map nearly saturates global contour detection (AUC 0.987 vs
   0.893, AP 0.718 vs 0.381). Learning the knife works.
2. But it barely helps where the knife actually matters: on the
   contact seams the margin is only 0.723 vs 0.678 AUC (recall 0.63
   vs 0.48). Soft depth edges are not why seams stay fused —
   seam pixels that look like interior to both touching parts are,
   which a contour-supervised head trained on the same fused-looking
   RGBD input cannot separate either. E6's attribution stands:
   seed placement (+0.076 to oracle), not the knife, is the lever.
3. The retrain itself cost ~0.06: the E7 checkpoint's heatmap/seed
   behavior degraded (92.6 pieces/img under identical markers+depth
   elevation vs E6's 75.3; AP75 0.418 vs 0.501 on the same
   operator). A third head on a 14.6M U-Net is not free at this
   scale — the within-checkpoint control is what makes the
   elevation comparison honest.

Not run: md6/md12 (budget consumed by the 103-min train and eval
restarts; md9 was E6's saturated optimum and the elevation variable
does not interact with it).
