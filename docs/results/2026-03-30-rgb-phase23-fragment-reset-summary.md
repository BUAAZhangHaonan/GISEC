> **Historical Naming Notice**: This document uses the pre-rename naming convention (abbreviated stage/variant/phase codes). For the current mapping, see the "Variant Naming Reference" table in README.md.

**Historical Naming Notice**: This document uses the pre-rename naming convention (abbreviated stage/variant/phase codes). For the current mapping, see the "Variant Naming Reference" table in README.md.

# 2026-03-30 RGB Phase 2/3 Fragment Reset Summary

## Scope

- Goal stays the same: beat Magformer with a smaller RGB GISEC first, then keep the path open past `AP 80`.
- This note covers the real full-dataset `rgb_phase23_fragment_reset` milestone on the frozen `Mask2Former RGB @1024` backbone.

## Stage 2 Gate

- gate_passed: `False`
- stage3_status: `gated_off`
- baseline segm/AP: `0.5451`
- train cache overflow rate: `0.9441` (69528 / 73648)
- val cache overflow rate: `0.9465` (8556 / 9040)

## Practical Read

- The reset either passes the Stage 2 fragment gate and unlocks Stage 3, or it stops honestly at the fragment-representation layer.
- In this run, Stage 3 is gated off because Stage 2 did not clear the required fragment-quality thresholds.
- The strongest failed gate is the fragment-space quality itself: `split_gt_rate = 0.0022`, `impure_fragment_rate = 0.6315`, `leakage_rate = 0.3833`, and `overflow_crop_rate = 0.9467`.
- So the current `K=6` explicit-fragment reset does not earn Stage 3 promotion on the real dataset. The blocker is upstream fragment design, not the local merger.
