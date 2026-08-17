# E2: scoring simulation and fragmentation cost (1566 val, zero training)

## 1. Purpose

Quantify, without any training, the two suspected killers of the March
U-Net route: (a) instance scoring, and (b) connected-component (CC)
fragmentation of multi-component GT masks. GT masks play a perfect
segmenter; only the scoring rule or the instance decomposition varies.

## 2. Method

- Data: `datasets/20260318_1K_1566`, val split. 149 images, 9494 GT
  instances, single category. Protocol: `gisec.eval.coco_eval
  .evaluate_json` (standard COCO, score floor 0.05, maxDets up to 100),
  predictions exported via `gisec.eval.coco_export.masks_to_coco_results`.
- Scoring (same GT instances, only scores change): const 0.5, random
  (seeded), area (normalized), compactness (4*pi*A/P^2), grad energy
  (mean Sobel magnitude inside mask), oracle 1.0, oracle + Gaussian
  noise sigma 0.1 / 0.3.
- Fragmentation: each GT mask split by `cv2.connectedComponents` into
  predictions (oracle confidence). Merge variants: by GT ownership
  (oracle identity) vs by unsupervised depth ordering (fragments sorted
  by mean depth, cut into GT-count groups at the largest depth gaps).
- Measured multi-CC GT fraction on this split: 3362/9494 = 35.4% (the
  29.2% figure circulating earlier comes from a different split or
  definition; this val number is what the simulation uses).
  19015 fragments from 9494 instances.

## 3. Results

Scoring on perfect GT instances (segm AP | bbox AP):

| scheme | segm AP | bbox AP | vs oracle (segm) |
|---|---|---|---|
| const 0.5 | 0.9901 | 0.8520 | 1.00x |
| random | 0.9901 | 0.8497 | 1.00x |
| area | 0.9901 | 0.9001 | 1.00x |
| compactness | 0.9901 | 0.8642 | 1.00x |
| grad energy | 0.9901 | 0.8468 | 1.00x |
| oracle 1.0 | 0.9901 | 0.8520 | 1.00x |
| oracle + N(0, 0.1) | 0.9901 | 0.8483 | 1.00x |
| oracle + N(0, 0.3) | 0.9901 | 0.8492 | 1.00x |

With perfect masks there are no false positives, so score ordering
barely matters; segm AP saturates at 0.9901 for every scheme (the
missing 0.0099 is images with more than 100 instances hitting the
maxDets=100 cap). Even sigma=0.3 ranking noise costs nothing on segm
and <= 0.005 on bbox. "area" even beats oracle on bbox (0.9001 vs
0.8520) because under the 100-dets cap preferring large instances is a
better cap policy.

Fragmentation (oracle confidence 0.9 everywhere):

| variant | segm AP | bbox AP |
|---|---|---|
| GT instances (reference) | 0.9901 | 0.8520 |
| CC split into fragments | 0.4083 | 0.3087 |
| merge by GT (oracle identity) | 0.9901 | 0.8520 |
| merge by depth ordering (wrong identity) | 0.3559 | 0.1644 |

## 4. Verdict

- (a) PASS. Cheap scoring exists trivially: every tested scheme,
  including constant 0.5 and random, reaches >= 0.9x oracle AP
  (in fact 1.00x on segm). Scoring is not the March death cause.
- (b) Fragmentation is the killer and it is recoverable. CC shredding
  alone drops segm AP by 0.582 (0.9901 -> 0.4083, -59%); oracle
  fragment merging recovers 100% of it. But merging with a wrong
  unsupervised identity rule (depth ordering) lands at 0.3559, worse
  than not merging at all. So the GNN in E4 does not need to be a
  scorer, it needs correct same-part fragment identity; its theoretical
  headroom is ~0.58 segm AP, and a wrong grouping actively hurts.
