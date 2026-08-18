# U-Net + GNN Route Experiment Ledger

ROUTE VERDICT (E5): original dense+merge conception DEAD — merge has no input (E3: 91% fusion) and GNN is removed from the route name; survived as depth-first small-model pipeline, CONTINUE with one conditional experiment (learned center-heatmap seeds, pass line segm AP >= 0.42 on 1566 val). See exp05_verdict/VERDICT.md.

| Experiment | Goal | Status | Conclusion | Key Number | Date |
|---|---|---|---|---|---|
| E1 identity_signal | Measure whether fragment-pair identity features alone separate same-part pairs from different-part pairs | done | PASS: depth+spatial AUC 0.991 (centroid_dist alone 0.982); appearance ~chance; never-merge baseline 0.9916 so merge rules must be conservative | pair AUC 0.991 | 2026-08-15 |
| E2 scoring_sim | Simulate the full scoring pipeline on ground-truth fragments to get the AP upper bound under a perfect detector | done | PASS: scoring free (const 0.5 segm 0.9901, area best bbox 0.9001); CC fragmentation is the killer 0.9901->0.4083, oracle merge recovers fully, wrong merge 0.3559 worse than no merge | segm AP 0.4083 CC-split | 2026-08-15 |
| E3 unet_dense | Train a U-Net (SMP) on the 1566 dataset to produce fragment masks; evaluate mask quality | done | FAIL: segm AP 0.0287 << 0.20, route closed. Semantic is fine (mIoU 0.945, oracle-semantic control only 0.0385): union-mask CC fuses 91% of parts (864 CCs vs 9494 GT inst); merging has no input, splitting is the real problem | segm AP 0.0287 | 2026-08-18 |
| E4 instance_split | Depth-guided watershed split of the fused union semantic mask (E3 ckpt); controls: GT-semantic, GT seed-count, RGB-gradient | done | GRAY: segm AP 0.3125 (CI 0.287-0.356) in [0.20,0.35), E5 decides. Depth gradient is a real boundary signal (RGB elev 0.026, CC 0.0287, fusion 91%->8.2%) but boundary precision binds: GT-semantic control only 0.493, GT seed-count hurts (0.18); AP still rising at md15 grid edge | segm AP 0.3125 | 2026-08-18 |
| E5 verdict | Combine E1-E4 evidence and decide go/no-go for the U-Net+GNN route | done | Original conception DEAD (merge has no input, E3 fusion 91%; GNN removed from route name); continue as depth-first split route with one conditional experiment: center-heatmap seeds on E3 U-Net, pass segm AP >= 0.42 | pass line AP 0.42 | 2026-08-18 |

## Note: 1566 val scene structure

Parsed from the 149 val filenames in `datasets/20260318_1K_1566/images/val`
(pattern `{part}_{NN}_scene_{sssss}_{ffff}_v1.png`, key = part+scene):

- 149 frames, 87 independent scenes (62 scenes with 2 frames, 25 with 1 frame).
- Same-scene frame pairs: 62 out of 11026 total frame pairs (0.56%).
- Among the 62 same-scene pairs, frame-number difference < 5 in 100% of cases.

Implication: report clustered bootstrap CIs at the scene level (87 units), not
the image level; treating the 149 images as independent would slightly
underestimate variance.
| E6 center_split | Learned center-heatmap seeds for depth watershed (E5-approved conditional) | done | PASS: segm AP 0.4797 >= 0.42, CI lower 0.4364 >= 0.38 (md9). +0.172 over E4 depth seeds, 89% of M2F at 1/3 params; oracle GT centers 0.5558 defines remaining seed-placement headroom; heatmap wins by coverage (238 markers/img) not per-seed accuracy (median 22.9px worse than depth 17.9px) | segm AP 0.4797 | 2026-08-18 |
| E7 boundary_split | Learned instance-boundary elevation for watershed (single variable vs E6) | done | GRAY 0.4583 < 0.50, AP75 0.472 < E6 0.501 -> E6 config wins tie-break and goes to 32254. Learned knife is real (contour AUC 0.987 vs 0.893; +0.036 AP within-ckpt) but seams barely move (0.723 vs 0.678 AUC); third head degraded seeds (92.6 vs 75.3 pieces/img) | segm AP 0.4583 | 2026-08-18 |
