# FragmentGraph Split Manifest

Target repo: standalone `fragment-graph` project with Python package `fraggraph` under `src/fraggraph/...`.

This manifest keeps the legacy fragment-first graph-merge stack, copies only the shared helpers that stack actually uses, and removes all active/query/router surfaces.

## Included source files and destination paths

### Core FragmentGraph code

| Current path | Destination path | Notes |
| --- | --- | --- |
| `gisec/__init__.py` | `src/fraggraph/__init__.py` | Package root |
| `gisec/cli/__init__.py` | `src/fraggraph/cli/__init__.py` | CLI package glue |
| `gisec/cli/train_legacy.py` | `src/fraggraph/cli/train.py` | Direct `fraggraph train` entrypoint |
| `gisec/cli/eval_legacy.py` | `src/fraggraph/cli/eval.py` | Direct `fraggraph eval` entrypoint |
| `gisec/cli/infer_legacy.py` | `src/fraggraph/cli/infer.py` | Direct `fraggraph infer` entrypoint |
| `gisec/config/__init__.py` | `src/fraggraph/config/__init__.py` | Config package glue |
| `gisec/config/io.py` | `src/fraggraph/config/io.py` | YAML/config loader helpers |
| `gisec/config/variants.py` | `src/fraggraph/config/variants.py` | Keep the fragment-first variant registry, with names rewritten to FragGraph identifiers |
| `gisec/datasets/__init__.py` | `src/fraggraph/datasets/__init__.py` | Dataset package glue |
| `gisec/datasets/prototype_bank.py` | `src/fraggraph/datasets/prototype_bank.py` | Prototype-bank loading and contract helpers |
| `gisec/datasets/ecc_query_dataset.py` | `src/fraggraph/datasets/ecc_query_dataset.py` | Copy the full `ECCGraphDataset` module, or an equivalent local copy of that module, together with the local helpers it depends on; do not retain query-alpha branding |
| `gisec/engine/__init__.py` | `src/fraggraph/engine/__init__.py` | Engine package glue |
| `gisec/engine/runtime.py` | `src/fraggraph/engine/runtime.py` | Runtime, evaluation, export, and reporting helpers trimmed to FragmentGraph-only imports |
| `gisec/graph_refiner.py` | `src/fraggraph/graph_refiner.py` | Graph rescue and merge orchestration |
| `gisec/models/__init__.py` | `src/fraggraph/models/__init__.py` | Model package glue |
| `gisec/models/fragment_bundle.py` | `src/fraggraph/models/fragment_bundle.py` | Fragment proposal bundle |
| `gisec/models/gisec_model.py` | `src/fraggraph/models/fragment_graph_model.py` | Rename main model class to `FragmentGraphModel` |
| `gisec/models/graph_head.py` | `src/fraggraph/models/graph_head.py` | Graph edge scorer |
| `gisec/models/graph_utils.py` | `src/fraggraph/models/graph_utils.py` | Graph building, fragment splitting, and merge logic |
| `gisec/models/prototype_cache.py` | `src/fraggraph/models/prototype_cache.py` | Prototype cache helpers |
| `gisec/models/prototype_unet.py` | `src/fraggraph/models/prototype_unet.py` | Prototype-conditioned U-Net backbone |
| `gisec/ops/__init__.py` | `src/fraggraph/ops/__init__.py` | Ops package glue |
| `gisec/ops/connected_components.py` | `src/fraggraph/ops/connected_components.py` | CUDA connected-components wrapper |
| `gisec/ops/csrc/buf.h` | `src/fraggraph/ops/csrc/buf.h` | CUDA extension source |
| `gisec/ops/csrc/buf_2d.cu` | `src/fraggraph/ops/csrc/buf_2d.cu` | CUDA extension source |
| `gisec/ops/csrc/buf_3d.cu` | `src/fraggraph/ops/csrc/buf_3d.cu` | CUDA extension source |
| `gisec/ops/csrc/registry.cu` | `src/fraggraph/ops/csrc/registry.cu` | CUDA extension source |
| `gisec/train/__init__.py` | `src/fraggraph/train/__init__.py` | Train package glue |
| `gisec/train/train_gisec.py` | `src/fraggraph/train/train_fragment_graph.py` | Main FragmentGraph train/eval/infer implementation |
| `gisec/utils/__init__.py` | `src/fraggraph/utils/__init__.py` | Utils package glue |
| `gisec/utils/logging.py` | `src/fraggraph/utils/logging.py` | Logging helpers |
| `gisec/utils/visualization.py` | `src/fraggraph/utils/visualization.py` | Preview and overlay rendering |

### Shared helpers copied with FragmentGraph

| Current path | Destination path | Notes |
| --- | --- | --- |
| `baseline/common/__init__.py` | `src/fraggraph/common/__init__.py` | Shared helper package glue |
| `baseline/common/dataset.py` | `src/fraggraph/common/dataset.py` | `BaselineInstanceDataset` and collate helpers |
| `baseline/common/boundary_metrics.py` | `src/fraggraph/common/boundary_metrics.py` | Boundary metrics used by graph cache and eval paths |
| `baseline/common/coco_export.py` | `src/fraggraph/common/coco_export.py` | COCO mask export helpers |
| `baseline/common/config.py` | `src/fraggraph/common/config.py` | Benchmark config defaults used by retained runners/tests |
| `baseline/common/contracts.py` | `src/fraggraph/common/contracts.py` | Run-summary and artifact contract constants |
| `baseline/common/export.py` | `src/fraggraph/common/export.py` | Run summary payload helpers |
| `baseline/common/fragment_graph_cache.py` | `src/fraggraph/common/fragment_graph_cache.py` | Offline fragment-graph cache builder |
| `baseline/common/fragment_quality.py` | `src/fraggraph/common/fragment_quality.py` | Fragment quality metrics used by retained baseline/unet tests |
| `baseline/common/instance_targets.py` | `src/fraggraph/common/instance_targets.py` | Instance target pack helpers |
| `baseline/common/pathology.py` | `src/fraggraph/common/pathology.py` | Prediction pathology summaries |
| `baseline/common/paths.py` | `src/fraggraph/common/paths.py` | Path helpers |
| `baseline/common/training_artifacts.py` | `src/fraggraph/common/training_artifacts.py` | Training-history and visualization helpers |
| `baseline/rgbd/__init__.py` | `src/fraggraph/rgbd/__init__.py` | RGB-D helper package glue |
| `baseline/rgbd/depth_cache.py` | `src/fraggraph/rgbd/depth_cache.py` | Depth-cache helpers |
| `baseline/rgbd/fusion.py` | `src/fraggraph/rgbd/fusion.py` | Input-fusion helpers used by unet and fragment cache code |
| `baseline/reference_graph/__init__.py` | `src/fraggraph/reference_graph/__init__.py` | Reference-graph package glue |
| `baseline/reference_graph/dataset.py` | `src/fraggraph/reference_graph/dataset.py` | FragmentGraph merge dataset |
| `baseline/reference_graph/eval.py` | `src/fraggraph/reference_graph/eval.py` | Reference-graph eval helpers |
| `baseline/reference_graph/eval_pipeline.py` | `src/fraggraph/reference_graph/eval_pipeline.py` | Reference-graph evaluation pipeline |
| `baseline/reference_graph/model.py` | `src/fraggraph/reference_graph/model.py` | Reference-graph model |
| `baseline/reference_graph/train.py` | `src/fraggraph/reference_graph/train.py` | Reference-graph training entrypoint |
| `baseline/unet/__init__.py` | `src/fraggraph/unet/__init__.py` | U-Net package glue |
| `baseline/unet/eval.py` | `src/fraggraph/unet/eval.py` | U-Net eval helpers |
| `baseline/unet/export.py` | `src/fraggraph/unet/export.py` | U-Net fragment-cache export helpers |
| `baseline/unet/model.py` | `src/fraggraph/unet/model.py` | U-Net model family |
| `baseline/unet/train.py` | `src/fraggraph/unet/train.py` | U-Net training helpers |

## Tests to keep

Keep these tests in the cleaned FragmentGraph repo, with `gisec` / `legacy` filenames renamed to FragGraph names during the split:

- `tests/conftest.py`
- `tests/test_baseline_contracts.py`
- `tests/test_baseline_dataset.py`
- `tests/test_baseline_dataset_collate.py`
- `tests/test_baseline_unet_family.py`
- `tests/test_baseline_unet_smoke.py`
- `tests/test_eval_contracts.py`
- `tests/test_eval_infer_gisec_minibatch.py`
- `tests/test_fragment_graph_cache.py`
- `tests/test_fragment_quality.py`
- `tests/test_gisec_model_forward.py`
- `tests/test_graph_batch_and_merge.py`
- `tests/test_graph_batch_regression.py`
- `tests/test_graph_builder_gpu.py`
- `tests/test_graph_builder_legacy.py`
- `tests/test_legacy_reference_mode.py`
- `tests/test_legacy_throughput.py`
- `tests/test_merged_blob_pathology.py`
- `tests/test_no_reference_ablations.py`
- `tests/test_prototype_bank_loader.py`
- `tests/test_prototype_cache_source.py`
- `tests/test_prototype_reference_routing.py`
- `tests/test_prototype_routing.py`
- `tests/test_reference_ablation_modes.py`
- `tests/test_reference_graph_eval.py`
- `tests/test_reference_graph_merge.py`
- `tests/test_runtime_export.py`
- `tests/test_train_gisec_minibatch.py`
- `tests/test_train_losses.py`
- `tests/test_training_artifacts.py`
- `tests/test_variant_spec.py`

Delete the rest of `tests/`, especially all `active`, `query`, `query-alpha`, `fragment_generator`, `instance_fragment`, `local_merger`, `reference_splitter`, `audit`, and benchmark-only coverage.

## Configs to keep

Keep and rename only the FragmentGraph-relevant configs:

- `configs/data/ecc_20260318_1k_1566.yaml`
- `configs/data/ecc_20260318_1k_32254.yaml`
- `configs/reference/reference_20260318_1k_13440.yaml`
- `configs/train/smoke_1024.yaml`
- `configs/train/recovery_smoke_1024.yaml`
- `configs/train/full_legacy_20ep.yaml` -> `configs/train/full_graph_merge_20ep.yaml`
- `configs/variant/legacy_rgbd_prototype_affinity_baseline.yaml` -> `configs/variant/fraggraph_rgbd_prototype_affinity_baseline.yaml`
- `configs/variant/legacy_rgbd_prototype_ownership_graph_cues.yaml` -> `configs/variant/fraggraph_rgbd_prototype_ownership_graph_cues.yaml`

Delete `configs/active/`, `configs/query/`, `configs/runtime/`, and all other configs that only serve the active or query projects or unrelated baseline branches.

## Scripts to keep

Keep only workflow scripts that FragmentGraph still needs, and rename the ones that still mention `gisec` or `legacy`:

- `scripts/experiments/common_runner.sh`
- `scripts/experiments/build_fragment_graph_cache.py` -> `scripts/experiments/build_fraggraph_cache.py`
- `scripts/experiments/run_gisec_legacy_smoke.sh` -> `scripts/experiments/run_fraggraph_smoke.sh`
- `scripts/experiments/run_legacy_1k_20ep_1024_gisec.sh` -> `scripts/experiments/run_fraggraph_1k_20ep_1024.sh`
- `scripts/experiments/run_legacy_1k_20ep_1024_gisec_all.sh` -> `scripts/experiments/run_fraggraph_1k_20ep_1024_all.sh`
- `scripts/experiments/run_legacy_1k_20ep_1024_gisec_eval.sh` -> `scripts/experiments/run_fraggraph_1k_20ep_1024_eval.sh`
- `scripts/experiments/run_legacy_1k_20ep_1024_gisec_infer.sh` -> `scripts/experiments/run_fraggraph_1k_20ep_1024_infer.sh`
- `scripts/experiments/train_reference_graph_merge.py`
- `scripts/experiments/eval_reference_graph_merge.py`

Delete `scripts/audit/`, `scripts/maintenance/`, `scripts/analysis/`, and all other one-off diagnostic, benchmark, query, active, legacy, or fragment-generator scripts.

## Delete categories relevant to FragmentGraph

- Active/query/router code: `gisec/active/`, `gisec/cli/_routing.py`, all `*_active.py`, `*_legacy.py`, and `*_query.py` entrypoints, `gisec/train/train_active.py`, `gisec/train/train_query.py`, `gisec/train/query_targets.py`, `gisec/config/query_models.py`, `gisec/engine/query_*.py`, `gisec/models/query_*.py`, `gisec/models/query_common.py`, `gisec/bridge/`, and any router-only glue.
- Configs for other projects: `configs/active/`, `configs/query/`, `configs/runtime/`, and any baseline config not explicitly listed in the keep set.
- Process and planning docs: `docs/plans/`, `docs/reviews/`, and any `docs/experiments/` file that is a plan, ladder, gate, or metric-design note.
- Audit and stale outputs: `output/audit/`, `output/experiments/` unless a checkpoint is explicitly retained for reproducibility, and any audit/performance/result artifact that is not part of FragmentGraph documentation.
- One-off helper scripts: `scripts/audit/`, `scripts/maintenance/`, `scripts/analysis/`, and all experiment runners/builders that are not in the keep list above.
- Cache and build noise: `__pycache__/`, `*.pyc`, `*.pyo`, `*.eggs`, `.pytest_cache/`, `*.egg-info/`, `dist/`, `build/`.
- Backup/temp files: anything matching `*_v1`, `*_old`, `*_backup`, `*_copy`, `*_bak`, or `*_temp`.
- Repo-wide docs that compare against the other projects instead of describing FragmentGraph on its own: active/query/legacy/performance/audit result notes, plus `Project_Summary_Report.md`.

## Rename map

- `gisec/__init__.py` package identity -> `fraggraph` package identity
- `gisec/cli/train_legacy.py` -> `src/fraggraph/cli/train.py`
- `gisec/cli/eval_legacy.py` -> `src/fraggraph/cli/eval.py`
- `gisec/cli/infer_legacy.py` -> `src/fraggraph/cli/infer.py`
- `gisec/models/gisec_model.py` -> `src/fraggraph/models/fragment_graph_model.py`
- `GISECModel` -> `FragmentGraphModel`
- `gisec/train/train_gisec.py` -> `src/fraggraph/train/train_fragment_graph.py`
- `gisec/config/variants.py` variant IDs `legacy_*` -> `fraggraph_*` equivalents
- `tests/test_eval_infer_gisec_minibatch.py` -> `tests/test_eval_infer_fraggraph_minibatch.py`
- `tests/test_gisec_model_forward.py` -> `tests/test_fragment_graph_model_forward.py`
- `tests/test_graph_builder_legacy.py` -> `tests/test_graph_builder_fraggraph.py`
- `tests/test_legacy_reference_mode.py` -> `tests/test_reference_mode.py`
- `tests/test_legacy_throughput.py` -> `tests/test_fraggraph_throughput.py`
- `tests/test_train_gisec_minibatch.py` -> `tests/test_train_fraggraph_minibatch.py`
- `legacy` wording in FragmentGraph-only code, comments, CLI help, docs, and filenames -> `fraggraph` or `graph-merge` wording
- `gisec` wording in FragmentGraph-only code, comments, CLI help, docs, and filenames -> `fraggraph`
- `prototype` terminology stays in FragmentGraph, because this project still uses prototype-conditioned and prototype-cache code

## Root files to add or rewrite

- `README.md`
- `docs/architecture.md`
- `docs/experiment-results.md`
- `pyproject.toml`
- `environment.yml`
- `.gitignore`

## Notes

- This repo keeps the legacy fragment-first graph-merge stack only.
- It does not keep the active Mask2Former stack, the query-alpha stack, or the CLI router layer.
- The copied `gisec/datasets/ecc_query_dataset.py` module is a full FragmentGraph-owned local dataset module, with query-alpha branding removed where necessary.
