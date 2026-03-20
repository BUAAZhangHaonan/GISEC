# GISEC Configs

These YAML files are thin defaults for the existing CLI.

- `common` applies to `train`, `eval`, and `infer`
- `train`, `eval`, and `infer` only apply to the matching command
- later `--config` files override earlier ones
- direct CLI flags still override YAML values

Example:

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/variant/a1.yaml \
  --config configs/train/smoke_1024.yaml \
  --output-dir output/experiments/gisec_v2_smoke/A1
```
