# GISEC Configs

These YAML files only cover the standalone GISEC pipeline.

- `configs/data/` stores dataset roots and annotation defaults.
- `configs/model/` stores the named GISEC variants and their model settings.
- `configs/reference/` stores reference-bank defaults for variants that need them.

Later `--config` files override earlier ones, and direct CLI flags still win over YAML values.

Example:

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/model/base_rgbd_1024_refine_ref_graph.yaml \
  --output-dir output/experiments/gisec/base_rgbd_1024_refine_ref_graph
```
