# E4: depth-guided watershed instance split (1566 val, zero training)

## Goal

E3 left the dense route with one problem: the union semantic mask
fuses 91% of parts (864 CCs vs 9494 GT instances) and CC cannot split
them. Depth is a strong identity signal (E1: pair AUC 0.991; 26.4x
between/within group variance), so test whether a watershed on a
depth-derived elevation map, seeded by depth plateaus, splits the
fused mask back into instances. Pass: segm AP >= 0.35; gray 0.20-
0.35 (E5 decides); < 0.20 closes the route.

## Method

- Semantic mask: E3 checkpoint (`exp03_unet_dense/runs/best.pth`,
  mIoU 0.945), sigmoid >= 0.5, reused via `eval_pipeline` imports.
- Markers: `peak_local_max` on Gaussian-smoothed depth (sigma 2)
  inside the semantic mask, min_distance grid {3, 5, 9, 15}.
- Elevation: -depth (plateau flooding) or |sobel(depth)| (gradient
  barrier). Split: `skimage.segmentation.watershed`. Post: regions
  < 32 px dropped or merged into the longest-boundary neighbor.
- Score = area-normalized (E2). Top-100 pieces/image evaluated (COCO
  maxDets=100 protocol; identical AP). Split stats: piece covers a
  GT at >= 50% of GT area = claim; oversplit = GT claimed by >= 2
  pieces, undersplit = piece claiming >= 2 GTs.

## Numbers

Main grid (model semantic, segm AP, drop/merge):

| elevation | md3 | md5 | md9 | md15 |
|---|---|---|---|---|
| -depth | .003/.003 | .005/.006 | .014/.014 | .017/.017 |
| grad(depth) | .011/.012 | .036/.038 | .173/.175 | .312/.313 |

FINAL (grad-depth, md15, merge): **segm AP 0.3125** (AP50 0.6198,
AP75 0.2787), bbox AP 0.3566. 96.8 pieces/img vs GT 63.7. md15 is
the grid edge and AP was still rising - the true optimum lies beyond
15, so 0.3125 is a lower bound for this operator family.

Controls (all at grad-depth, md15, merge):

| config | segm AP | pieces/img | read |
|---|---|---|---|
| a. GT semantic + depth ws | 0.4933 | 97.2 |
| b. GT seed count (top-N peaks) | 0.1798 | 60.9 |
| c. RGB-gradient elevation | 0.0260 | 209.8 |

Reads: (a) depth expresses instance boundaries, but only to ~0.49,
not 0.99; (b) knowing N does not help - greedy top-N peak selection
picks worse markers than the natural peaks; (c) appearance is
useless as elevation, depth is the whole signal.

Split statistics (FINAL): fusion resolved - undersplit pieces (>= 2
GTs at >= 50% coverage each) drop from ~91% of parts fused in the
E3 union mask to 8.2% of pieces; oversplit GT rate ~0 by the 50%
claim criterion; count over-segments 1.5x (96.8 vs 63.7).

Scene bootstrap (87 scenes, 200x): segm 0.314 [0.287, 0.356],
bbox 0.357 [0.331, 0.399].

## Conclusion

GRAY ZONE: segm AP 0.3125, in [0.20, 0.35); E5 to decide. Attribution:
(i) the depth gradient is a real instance-boundary signal (RGB
elevation collapses to 0.026; depth lifts 0.0287 CC -> 0.3125 and
resolves 91% fusion -> 8.2% undersplit); (ii) the binding constraint
is boundary precision, not seed count - GT semantics only reach
0.493, and GT-count seeds actually hurt (0.18): boundaries land at
IoU 0.5-0.75 (AP50 0.62 vs AP75 0.28), so adjacent same-depth parts
and soft depth edges leak across instances; (iii) AP was still rising
at the md15 grid edge. To pass 0.35 this operator needs learned
seeds (e.g. center heatmaps) or boundary-aware refinement, which is
training work, not post-processing.
