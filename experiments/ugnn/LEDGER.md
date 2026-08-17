# U-Net + GNN Route Experiment Ledger

| Experiment | Goal | Status | Conclusion | Key Number | Date |
|---|---|---|---|---|---|
| E1 identity_signal | Measure whether fragment-pair identity features alone separate same-part pairs from different-part pairs | pending | - | - | - |
| E2 scoring_sim | Simulate the full scoring pipeline on ground-truth fragments to get the AP upper bound under a perfect detector | pending | - | - | - |
| E3 unet_dense | Train a U-Net (SMP) on the 1566 dataset to produce fragment masks; evaluate mask quality | pending | - | - | - |
| E4 fragment_gnn | Train a GNN (PyG) over fragment graphs to score pair identity; evaluate end-to-end AP | pending | - | - | - |
| E5 verdict | Combine E1-E4 evidence and decide go/no-go for the U-Net+GNN route | pending | - | - | - |

## Note: 1566 val scene structure

Parsed from the 149 val filenames in `datasets/20260318_1K_1566/images/val`
(pattern `{part}_{NN}_scene_{sssss}_{ffff}_v1.png`, key = part+scene):

- 149 frames, 87 independent scenes (62 scenes with 2 frames, 25 with 1 frame).
- Same-scene frame pairs: 62 out of 11026 total frame pairs (0.56%).
- Among the 62 same-scene pairs, frame-number difference < 5 in 100% of cases.

Implication: report clustered bootstrap CIs at the scene level (87 units), not
the image level; treating the 149 images as independent would slightly
underestimate variance.
