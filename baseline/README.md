# GISEC Baselines

This folder holds the baseline benchmark stack used to compare `GISEC` against
standard RGB instance segmentation models and U-Net-family alternatives.

It is intentionally separate from `gisec/` so baseline code, adapters, and
framework-specific glue do not pollute the main method implementation.

The baseline benchmark stack will eventually cover:

- standard RGB instance segmentation baselines,
- strong standalone U-Net family instance baselines,
- RGB-D baseline extensions,
- shared experiment exports that can be compared directly against `GISEC`.

Current priority:

- Phase 1 proves a strong standalone backbone can solve the dataset without `reference` or `graph`.
- Only after that gate passes does the main `GISEC` line re-enter with `reference` and later `graph rescue`.
- The current split-first anchor is `configs/baseline/unet_rgb_full.yaml`, treated as a calibrated fragment proposal generator.
- `rgb_depth_wall` remains available for analysis, but it is paused as a mainline split candidate until it stops reintroducing merged-blob failures.
