# GISEC Baselines

This folder holds the baseline benchmark stack used to compare `GISEC` against
standard RGB instance segmentation models and U-Net-family alternatives.

It is intentionally separate from `gisec/` so baseline code, adapters, and
framework-specific glue do not pollute the main method implementation.

The baseline benchmark stack will eventually cover:

- standard RGB instance segmentation baselines,
- U-Net family baselines,
- RGB-D baseline extensions,
- shared experiment exports that can be compared directly against `GISEC`.
