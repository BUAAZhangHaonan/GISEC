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
- `configs/train/recovery_smoke_1024.yaml` is the default short-run recovery stack for `legacy_query_mask_only_debug/legacy_query_mask_reference_routing_debug/legacy_query_mask_reference_graph_rescue_debug`

Example:

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/variant/legacy_rgbd_prototype_ownership_graph_cues.yaml \
  --config configs/train/smoke_1024.yaml \
  --output-dir output/experiments/gisec_v2_smoke/legacy_rgbd_prototype_ownership_graph_cues
```

Recovery smoke example:

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/variant/legacy_query_mask_reference_routing_debug.yaml \
  --config configs/train/recovery_smoke_1024.yaml \
  --output-dir output/experiments/gisec_recovery_smoke/legacy_query_mask_reference_routing_debug
```
