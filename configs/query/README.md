# GISEC Query Config Surface

This directory is reserved for the `GISEC query-alpha` object-first line.

## Current Defaults

- `query_small_resnet18`
- `query_medium_resnet34`
- `ResNet` encoder family only
- `early6` fusion only: `RGB + depth geometry`

## Official Dataset Config

- `configs/data/ecc_20260318_1k_32254.yaml`

## Official Full-Rank Configs

- `configs/query/train/query_small_resnet18_full_train.yaml`
- `configs/query/eval/query_small_resnet18_full_eval.yaml`
- `configs/query/train/query_medium_resnet34_full_train.yaml`
- `configs/query/eval/query_medium_resnet34_full_eval.yaml`

## Reserved Later Variants

The next reserved names are:

- `query_ref_resnet18`
- `query_ref_resnet34`
- `query_graph_resnet18`
- `query_graph_resnet34`
- `query_refgraph_resnet18`
- `query_refgraph_resnet34`

## Deferred Official Configs

These variants now have full train/eval config stubs under `configs/query/train/` and `configs/query/eval/`:

- `query_ref_resnet18`
- `query_ref_resnet34`
- `query_graph_resnet18`
- `query_graph_resnet34`
- `query_refgraph_resnet18`
- `query_refgraph_resnet34`

Reference-bearing variants use `common.prototype_root` and the shared reference bank at `datasets/20260318_1K_13440`.
Graph-bearing variants set `common.graph_rescue: true` so the query train loop can route the post-processing branch when that code lands.

## Legacy Boundary

Legacy names such as `legacy_rgbd_prototype_affinity_baseline`, `legacy_rgbd_prototype_ownership_graph_cues`, `legacy_query_mask_only_debug`, `legacy_query_mask_reference_routing_debug`, `legacy_query_mask_reference_graph_rescue_debug`, `legacy_heuristic_graph_merge_baseline`, and `legacy_prototype_unet_*` remain historical or debug-only identifiers and must not be reused as query-alpha model ids.
