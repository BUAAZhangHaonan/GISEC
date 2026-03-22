# GISEC Configs

These YAML files are thin defaults for the existing CLI.

- `common` applies to `train`, `eval`, and `infer`
- `train`, `eval`, and `infer` only apply to the matching command
- later `--config` files override earlier ones
- direct CLI flags still override YAML values
- the default reference-pack policy is `pose_farthest` with `16` views for full runs
- `configs/train/smoke_1024.yaml` intentionally overrides that to `6` views so smoke runs stay cheap
- prototype routing can now be configured with `prototype_slot_count` and `prototype_topk`
- recovery-stage routing can be configured with `reference_conditioning_mode`, `reference_routing_mode`, and `reference_skip_margin`
- `configs/train/recovery_smoke_1024.yaml` is the default short-run recovery stack for `Q0/Q1/Q2`

Example:

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/variant/a1.yaml \
  --config configs/train/smoke_1024.yaml \
  --output-dir output/experiments/gisec_v2_smoke/A1
```

Recovery smoke example:

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/variant/q1.yaml \
  --config configs/train/recovery_smoke_1024.yaml \
  --output-dir output/experiments/gisec_recovery_smoke/Q1
```
