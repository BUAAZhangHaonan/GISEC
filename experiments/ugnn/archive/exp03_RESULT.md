# E3: small U-Net dense baseline (1566 dataset, real training)

## Goal

Fix the two March death causes per E1/E2 (scoring, fragmentation) and
measure what a small dense U-Net actually reaches on 1566 val.
Pass: segm AP >= 0.35 (route alive, go E4). < 0.20: route closed.
Reference: M2F swin-t 0.5381 on the same data; E2 perfect-detector
ceiling 0.9901.

## Caliber alignment (E1 19.6% vs E2 35.4%)

Unified criterion from here on: per GT instance, count 8-connected
components with area > 16 px; multi-component iff count >= 2.
Measured on 1566 val: 1129 / 9494 = **11.9%** (E1's 19.6% used a 5 px
floor; E2's 35.4% used no area floor at all). Components per instance:
1.14. All E3 numbers use the unified caliber.

## Method

- Model: `smp.Unet(resnet18/imagenet, in_channels=4, classes=1)`
  (~14.5M). Channel 4 = depth globally calibrated
  `(d - 0.245) / (0.686 - 0.245)`, clipped to [-1, 2], fixed constants.
- Train: 1261 imgs, BCE + Dice, AdamW 3e-4, cosine decay, 20 epochs,
  batch 8 @1024 (GPU had 96GB free; 12GB used). 35.7 min total.
- Entry points: `train_unet.py`, `eval_pipeline.py`. Instances:
  sigmoid >= 0.5 -> cv2 CC (area > 16) -> conservative merge
  (centroid_dist < tau1 AND |depth_median diff| < tau2, union-find)
  -> score = area-normalized -> `gisec.eval.coco_export/coco_eval`.

## Numbers

Training: val mIoU 0.9473 (per-image, epoch 19, still rising ~0.0002/
epoch), dataset-level 0.9453. Semantic quality is not the problem.

| variant | segm AP | bbox AP | n_inst |
|---|---|---|---|
| no merge | 0.0287 | 0.0258 | 776 (5.2/img) |
| merged (tau1=15, tau2=0.01) | 0.0287 | 0.0258 | 775 |
| oracle semantic (GT mask, same recovery) | 0.0385 | 0.0368 | 862 |
| M2F swin-t reference | 0.5381 | - | 9494 GT |
| E2 perfect-detector ceiling | 0.9901 | 0.8520 | - |

- tau grid (tau1 15-90, tau2 0.01-0.08): AP flat at 0.0287 everywhere;
  at conservative thresholds only 1 mergeable pair exists in the whole
  val set, 0 wrong merges. Merge rule is irrelevant here.
- Scene bootstrap (87 scenes, 1000x): segm 0.029 [0.023, 0.036],
  bbox 0.027 [0.022, 0.034].

## Conclusion

**FAIL: segm AP 0.0287 << 0.20. Route closed at E3.**

Attribution (semantic vs instance recovery): swapping in the GT
semantic mask and running the identical instance-recovery pipeline
gives 0.0385 — only +0.010 over the model. So ~97% of the gap to
0.9901 is instance recovery, not segmentation. The mechanism: CC on
the *union* semantic mask yields 864 components > 16 px across all of
val (5.8/img) versus 9494 GT instances — 91% of parts are fused with
neighbors in the union mask and are invisible to CC.

This reframes E2: its fragmentation simulation split *each GT instance
separately* (19015 fragments), but a real dense segmenter produces a
union mask where distinct parts touch and fuse. The real pipeline has
the opposite problem E2 modeled: not fragments-to-merge but
fused-blobs-to-split. A fragment-merge GNN (E4) has no input — there
are no fragments. Any revival of the dense route needs instance-aware
or boundary-aware prediction (e.g. pixel embedding, center/keypoint
heatmap, or watershed seeds), not a post-hoc merge rule.
