# E1: identity signal — fragment-pair separability (1566 val)

## Goal

Quantify whether "connected components of the same GT instance" vs
"components of different instances" are separable with cheap features
(depth / spatial / appearance), before building any GNN. This is the
signal ceiling for the fragment-GNN input. Pass bar: pair classification
AUC >= 0.85, or pure depth-rule merge accuracy >= 0.9.

## Method

- Data: 1566 val split (149 frames, 87 scenes), GT masks via
  `gisec.datasets.coco_utils.LiteCOCO` + `ann_to_mask`.
- Each GT instance mask is decomposed into 8-connected components
  (cv2, components < 5 px dropped). All within-image component pairs are
  enumerated; label = same GT instance.
- Features:
  - depth (raw npy meters, never normalized): |diff| of mean / median /
    q10 / q90 / std;
  - spatial: centroid distance, nearest-pixel distance (bbox-cropped
    distance transform), bbox gap, log area ratio, size-normalized
    centroid distance;
  - appearance: mean-color L1, 16-bin per-channel histogram intersection.
- AUC: single features scored directly; combos via logistic regression
  (standardized, class-weight balanced) under 5-fold GroupKFold grouped
  by image. Depth rule: same iff |mean depth diff| < tau, tau swept
  0.001..0.200 m.
- Entry point: `run_identity_signal.py` (python, optional --data-root).
  Full numbers in `results.json`.

## Numbers

735,366 pairs from 12,789 components over 9,456 instances; 6,192 positive
pairs (pos rate 0.84%). Multi-component instances: 1,850 / 9,456 = 19.6%
(measured on 1566 val, vs 29.2% earlier estimate on the other split).

| Feature layer | AUC |
|---|---|
| depth (5 feats) | 0.910 |
| spatial (5 feats) | 0.987 |
| appearance (2 feats) | 0.584 |
| depth + spatial | **0.991** |
| depth + appearance | 0.913 |
| spatial + appearance | 0.987 |
| all | 0.991 |

Best single features: centroid_dist 0.982, min_dist 0.942, bbox_gap
0.936, d_median 0.910, d_mean 0.903. Appearance is near chance.

Depth rule: best tau hits the sweep edge (0.001 m), accuracy 0.9776.
Warning: the trivial "never merge" policy scores 0.9916 (1 - pos rate),
so the depth rule passes the 0.9 bar numerically but is worse than doing
nothing — depth alone cannot drive merging. Depth as a discriminative
feature (AUC 0.91) is fine; depth as a threshold rule is not.

## Conclusion

PASS, comfortably. Depth + spatial cheap features reach AUC 0.991
(grouped by image), far above the 0.85 bar; spatial alone 0.987, depth
alone 0.910. Appearance adds nothing. The identity signal the GNN needs
exists and is cheap; next: E2 scoring simulation to measure the pipeline
AP ceiling under a perfect fragment detector.
