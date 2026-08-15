# GISEC Configs

These YAML files only cover the standalone GISEC pipeline.

- `configs/data/` stores dataset roots and annotation defaults (`ecc_20260318_1k_32254.yaml` for the main 32254-scene set, `ecc_20260318_1k_1566.yaml` for the small 1566-scene set).
- `configs/model/` stores the named GISEC variants and their model settings.
- `configs/reference/` stores reference-bank defaults for variants that need them.

Later `--config` files override earlier ones, and direct CLI flags still win over YAML values.

Example:

```bash
gisec train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/model/base_rgbd_1024_refine_ref_graph.yaml \
  --reference-root datasets/20260318_1K_13440 \
  --output-dir output/gisec/base_rgbd_1024_refine_ref_graph
```
