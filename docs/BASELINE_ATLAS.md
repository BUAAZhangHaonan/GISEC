# Baseline atlas: every model ever run on our datasets

Compiled 2026-09-03 from the cross-server archaeology (k100 / 6401 / 6403 /
4029; 4028 is offline) plus the current official-implementation campaign.
Four layers, strictest caliber first. All "segm AP" are COCO instance
segmentation AP; percent scale on the 1566 caliber, fraction scale on 32254
(unless noted).

## L1 — 32254 full-val (3,276 images): clean baselines & GISEC reference

Protocol unless noted: full train split, 20 epochs / 64K iter / batch 8
@1024 (GISEC-matching budget), full-val COCO segm AP, multiplicity-aware
paired scene bootstrap vs the GISEC reference.

| Model | Params | Input | Budget | segm AP | Implementation | Where run | Status |
|---|---:|---|---|---:|---|---|---|
| **GISEC E25** (reference, canonical) | 16.851M | RGB-D | 128K / bs16 | **0.87350** | ours (src/gisec) | k100 | done 09-02 |
| GISEC E24 (same-budget reference for fair comparison) | 16.851M | RGB-D | 64K / bs8 | 0.86113 | ours | k100 | done 08-29 |
| GISEC E20 (lineage) | 16.851M | RGB-D | 64K / bs8 | 0.84880 | ours | k100 | done 08-27 |
| Mask R-CNN R18 (mrcnn16fix) | 16.99M | RGB | 64K / bs8 | **0.6638** | torchvision | 6401 GPU0 | done 08-31 |
| Mask2Former R18 (m2f16v2) | 16.54M | RGB | 64K / bs8 | **0.4305** | HF transformers | 6401 GPU1 | done 09-02 |
| MagFormer-16M (external family) | 17.45M (+2.6% over cap, noted) | RGB-D | 64K from scratch | **0.7088** | magformer repo | 6401 | done 08-30 |
| YOLOv8s-seg (official, COCO-pretrained FT) | 11.79M | RGB | 20ep full / bs8 | pending | ultralytics 8.4 | 4029 GPU7 | running (09-03) |
| Panoptic-DeepLab R50 (official d2, AMP) | ~45M (TBC at print) | RGB | 64K / bs8 | pending | bowenc0221 tools_d2 | 6401 GPU1 | running (09-03) |
| CellPose 3.1.1.1 (official lib) | 6.60M | RGB | 20ep full / bs16 | pending | official lib | 4029 GPU4 | queued (RAM guard) |
| StarDist 0.9.2 (official lib) | 1.41M | RGB | 20ep full / bs4 | pending | official lib | 4029 GPU6 | running |
| UCN (official NVlabs, OCID-pretrained FT) | ~14M | RGB-D | 20ep full | pending | UnseenObjectClustering | 4029 GPU5 | queued (weights en route) |
| UOIS-Net zero-shot (official TOD weights) | ~81M | RGB-D | 0 (zero-shot) | **0.0003** (first 500 val) | chrisdxie/uois | 4029 GPU7 | done 09-03 |

**Erratum**: earlier tables said GISEC leads MagFormer-16M by "+14.0pt" —
that was the E20-era delta. On the E25 canonical it is **+16.47pt**; on the
same-budget E24 caliber it is **+15.2pt**.

Officially closed negative cells (user decisions 2026-08-30): mrcnn16d
(RGB-D MRCNN) and m2f16catfix (RGB-D M2F) — dropped, never trained.
Deep Watershed Transform: **no official training code exists**
(min2209/dwt ships TF1 inference only) — reported, not run. Fast UOIS (Fu et
al., Actuators 2024): exists but **closed-source** — lineage represented by
UOIS-Net (zero-shot above) and UCN (fine-tune queued).

## L2 — 32254, own exploration arms (NOT baselines; ablation/备选)

| Arm | Input | Caliber | segm AP | Notes |
|---|---|---|---:|---|
| d2m2f m2f_swin_t_rgbd_concat (E23, 47M official M2F 4-ch adapted) | RGB-D | full val @265K | 90.73 (91.43 flip-TTA) | accuracy reference from the magformer line; exploration code |
| d2m2f depth_only (E24) | depth | full val @40K | 78.94 | modality study |
| d2m2f DPTD (E24) | depth+distilled RGB | full val @40K | 78.17 | own fusion design |
| d2m2f hybrid_ptd (E24) | RGB+distilled depth | full val @40K | 75.39 | own fusion design |
| d2m2f concat4ch anchor (E24) | RGB-D | subset-1000 @40K | 76.29 | weights deleted; numbers valid |
| CDTI W1 cdti_full | RGB-D | subset-1000 seed42 @40K | 0.8441 | magformer method audit, 53.3M |
| CDTI W4 cdti_shallow | RGB-D | subset-1000 @40K | 0.8344 | |
| CDTI W2 concat_ctrl | RGB-D | subset-1000 @40K | 0.8296 | |
| CDTI W3 cdti_nodistill | RGB-D | subset-1000 @40K | 0.8367 | final read; four arms closed W1>W3>W4>W2, distillation necessary (+0.74 vs W3) |
| E26a invproj (in-mask float centroid + out-mask p*, off_w=1) | RGB-D | 64K/bs8, full val 3276 | 0.87433 (scene CI 0.87374 [0.87029,0.87776]) | winner ep17@0.9; paired vs E20 +2.52pt [+2.26,+2.74] |
| E26b offw0 (projected anchor + offset loss weight 0) | RGB-D | 64K/bs8, full val 3276 | 0.87617 (scene CI 0.87586) | winner ep15@0.95; paired vs E20 +2.73pt [+2.45,+2.99]; highest GISEC number to date |

**E26 ablation final read (2026-09-03; both arms 64K/bs8/20ep, same budget as E24; full val 3276,
eval chain identical to E24 incl. G1 reproduction gate).** Anchor triplet: centroid 0.84880 <
projected 0.86113 < invproj 0.87433 — the entire E24 gain over E20 comes from repairing invalid
(out-of-mask) centroids (+2.52pt, CI [+2.26,+2.74]); discretizing valid in-mask centroids costs
-1.32pt. Offset auxiliary loss judged harmful: projected + off_w=1 (E24, 0.86113) vs projected +
off_w=0 (E26b, 0.87617) isolates -1.50pt of pure multi-task interference, consistent with the
decode_fix finding that the offset head does not aid stride-4 decoding. E26a/E26b scene CIs overlap
(no direct arm-vs-arm paired test run); both exceed the E25 canonical 0.87350. Implication for the
265K long-run decision: the candidate recipe is invproj or offw0, not E24 as-is. Artifacts:
`experiments/ugnn/exp24_proj_anchor/eval/eval_full_e26_{invproj,offw0}.json`.

## L3 — 1566 caliber (train 1,261 / val 149), historical, magformer-side

Full 19-model dual-resolution table: see
`experiments/ugnn/baselines16m/official_1566/results_6403/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.md`
(copied verbatim). Key rows (segm AP @1024 / @512, 20ep):

mgm_mask2former_depthnorm 72.80/69.90 · magformer 68.42/59.91 ·
lightdepth 64.5-65.45 · magformer_nodpth 59.52/53.88 · mask2former(44M)
58.76/43.08 · maskrcnn(44M) 54.10/38.78 · **yolov8_seg_l 40.56/42.68 ·
x 40.32/42.93 · m 40.08/42.11 · s 36.45/40.21 · n 32.24/35.88** · uoais
17.67/6.16 · unet_boundary 13.60/5.79 · unetpp 10.88/10.18 ·
unet_semantic 3.57/3.25 · ucn 1.03/0.05 · msmformer 0.00/8.57
(unreliable).

100ep repaired runs (official libs): **CellPose 49.88@512 / 43.08@1024 ·
StarDist 45.82@512 / 56.70@1024** · IAUNet 8.78 / 13.85.
4029 10K-iter batch: CellPose 57.67/58.98 · StarDist 43.04/48.14
(IAUNet killed, no metrics). Runner code + result files recovered into
`experiments/ugnn/baselines16m/official_1566/` (see its README).

Superseded (supervision-path bug, historical only): mrcnn16 0.6082 ·
m2f16 0.4339 · m2f16cat 0.2244 · m2f16fix 0.2345 — all 32254 caliber,
2026-08-24..27.

## L4 — Official-implementation survey (2026-09-02/03)

| Paper | Official repo | Trainable? | Status | Our disposition |
|---|---|---|---|---|
| Panoptic-DeepLab (CVPR20) | bowenc0221/panoptic-deeplab (d2 build) | yes | deprecated 2021, functional | running (R50) |
| Deep Watershed Transform (CVPR17) | min2209/dwt | **no (inference only)** | dead since 2018 | reported only |
| UOIS-Net (ICRA20/RA-L) | chrisdxie/uois | yes (notebooks) + TOD/oid weights | frozen 2024 | zero-shot done |
| Fast UOIS (Actuators 2024) | none found | — | closed-source | not run; note in paper |
| UCN (ICRA21) | NVlabs/UnseenObjectClustering | yes + OCID ckpt | frozen | fine-tune queued |
| CellPose | MouseLand/cellpose | yes, active (4.x) | maintained | running (3.1.1.1 per 1566 protocol) |
| StarDist | stardist/stardist | yes, active | maintained | running |
| YOLOv8-seg | ultralytics | yes, active | maintained | running |
| Mask R-CNN / M2F | torchvision / HF | yes | maintained | done |

Environment patches applied to official PDL (recorded, not method changes):
torchvision≥0.15 `load_state_dict_from_url` import shim (5 files); d2
`.jpg`-naming convention satisfied via symlink dirs; AMP enabled to fit
full-image 1024 × batch 8 into 24G. Official trainer itself unmodified.
