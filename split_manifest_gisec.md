# GISEC Split Manifest

Target repo: cleaned standalone `gisec` project under `src/gisec/...`.

This manifest keeps the current active Mask2Former-based GISEC path, copies only the shared helpers that path actually uses, and drops the legacy fragment/query/router surfaces.

## Included source files and destination paths

### Core GISEC code

| Current path | Destination path | Notes |
| --- | --- | --- |
| `gisec/__init__.py` | `src/gisec/__init__.py` | Package root |
| `gisec/cli/__init__.py` | `src/gisec/cli/__init__.py` | Direct CLI package |
| `gisec/cli/train.py` | `src/gisec/cli/train.py` | Keep as direct `gisec train` entrypoint |
| `gisec/cli/eval.py` | `src/gisec/cli/eval.py` | Keep as direct `gisec eval` entrypoint |
| `gisec/cli/infer.py` | `src/gisec/cli/infer.py` | Keep as direct `gisec infer` entrypoint |
| `gisec/config/__init__.py` | `src/gisec/config/__init__.py` | Config package glue |
| `gisec/config/io.py` | `src/gisec/config/io.py` | YAML/config loader helpers |
| `gisec/active/config.py` | `src/gisec/config/variants.py` | GISEC variant spec and gating |
| `gisec/active/metrics.py` | `src/gisec/metrics.py` | GISEC-specific split/merge metrics |
| `gisec/active/model.py` | `src/gisec/models/gisec_model.py` | Main GISEC model |
| `gisec/active/runtime.py` | `src/gisec/runtime.py` | GISEC refinement/runtime helpers |
| `gisec/engine/__init__.py` | `src/gisec/engine/__init__.py` | Engine package glue |
| `gisec/engine/runtime.py` | `src/gisec/engine/runtime.py` | Shared eval/export/runtime utilities, trimmed to GISEC-only imports |
| `gisec/graph_refiner.py` | `src/gisec/graph_refiner.py` | Graph rescue/refinement logic |
| `gisec/models/__init__.py` | `src/gisec/models/__init__.py` | Model package glue |
| `gisec/models/graph_head.py` | `src/gisec/models/graph_head.py` | Graph head used by GISEC |
| `gisec/ops/__init__.py` | `src/gisec/ops/__init__.py` | Ops package glue |
| `gisec/ops/connected_components.py` | `src/gisec/ops/connected_components.py` | CUDA connected-components wrapper |
| `gisec/ops/csrc/buf.h` | `src/gisec/ops/csrc/buf.h` | CUDA extension source |
| `gisec/ops/csrc/buf_2d.cu` | `src/gisec/ops/csrc/buf_2d.cu` | CUDA extension source |
| `gisec/ops/csrc/buf_3d.cu` | `src/gisec/ops/csrc/buf_3d.cu` | CUDA extension source |
| `gisec/ops/csrc/registry.cu` | `src/gisec/ops/csrc/registry.cu` | CUDA extension source |
| `gisec/train/__init__.py` | `src/gisec/train/__init__.py` | Train package glue |
| `gisec/train/train_active.py` | `src/gisec/train/train_gisec.py` | Main GISEC train/eval/infer implementation |
| `gisec/utils/__init__.py` | `src/gisec/utils/__init__.py` | Utils package glue |
| `gisec/utils/logging.py` | `src/gisec/utils/logging.py` | Logging helpers |
| `gisec/utils/visualization.py` | `src/gisec/utils/visualization.py` | Preview/overlay rendering |
| `gisec/datasets/__init__.py` | `src/gisec/datasets/__init__.py` | Dataset package glue |
| `gisec/datasets/prototype_bank.py` | `src/gisec/datasets/reference_bank.py` | Extract and rename only the reference-view loading and sampling helpers needed by GISEC rescue stages; do not keep the FragmentGraph prototype-bank surface |

### Copied shared helpers used by GISEC

| Current path | Destination path | Notes |
| --- | --- | --- |
| `baseline/common/dataset.py` | `src/gisec/datasets/baseline_instance_dataset.py` | `BaselineInstanceDataset` and related sample loading |
| `baseline/common/instance_targets.py` | `src/gisec/datasets/instance_targets.py` | Instance target pack helpers; replace any query wording |
| `baseline/common/boundary_metrics.py` | `src/gisec/eval/boundary_metrics.py` | Boundary metrics used by GISEC eval paths |
| `baseline/common/coco_export.py` | `src/gisec/eval/coco_export.py` | COCO mask export helpers |
| `baseline/common/export.py` | `src/gisec/eval/export.py` | Run summary payload helpers |
| `baseline/common/training_artifacts.py` | `src/gisec/eval/training_artifacts.py` | Needed by Mask2Former smoke/train helpers |
| `baseline/rgbd/depth_cache.py` | `src/gisec/datasets/depth_cache.py` | RGB-D depth cache helpers |
| `baseline/mask2former/__init__.py` | `src/gisec/backbones/mask2former/__init__.py` | Mask2Former support package |
| `baseline/mask2former/adapter.py` | `src/gisec/backbones/mask2former/adapter.py` | Mask2Former input/model adapters |
| `baseline/mask2former/eval.py` | `src/gisec/backbones/mask2former/eval.py` | Smoke/eval helper retained with local imports |
| `baseline/mask2former/train.py` | `src/gisec/backbones/mask2former/train.py` | Smoke/train helper retained with local imports |

## Tests to keep

Keep these tests in the cleaned GISEC repo, with `active`-named files renamed to `gisec`-named equivalents during the split:

- `tests/test_active_backbone_inputs.py`
- `tests/test_active_checkpoint_loading.py`
- `tests/test_active_cli_minibatch.py`
- `tests/test_active_decode_contract.py`
- `tests/test_active_depth_mode.py`
- `tests/test_active_failure_metrics.py`
- `tests/test_active_graph_training.py`
- `tests/test_active_local_training.py`
- `tests/test_active_model_builder.py`
- `tests/test_active_pilot_summary.py`
- `tests/test_active_reference_depth.py`
- `tests/test_active_reference_inputs.py`
- `tests/test_active_refine_ranking.py`
- `tests/test_active_run_state.py`
- `tests/test_active_runner_dry_run.py`
- `tests/test_active_stage_init.py`
- `tests/test_active_stage_order.py`
- `tests/test_active_train_cli.py`
- `tests/test_active_variant_surface.py`
- `tests/test_config_io.py`
- `tests/test_project_metadata.py`
- `tests/test_eval_contracts.py`
- `tests/test_runtime_export.py`
- `tests/test_training_artifacts.py`
- `tests/test_baseline_dataset.py`
- `tests/test_baseline_dataset_collate.py`
- `tests/test_baseline_mask2former_smoke.py`

Delete the rest of `tests/`, especially routing, query, fragment, legacy, benchmark, and audit coverage.

## Configs to keep

Keep and rename only the GISEC-relevant configs:

- `configs/active/base_rgb_1024.yaml` -> `configs/model/base_rgb_1024.yaml`
- `configs/active/base_rgb_1024_refine.yaml` -> `configs/model/base_rgb_1024_refine.yaml`
- `configs/active/base_rgb_1024_refine_ref.yaml` -> `configs/model/base_rgb_1024_refine_ref.yaml`
- `configs/active/base_rgb_1024_refine_ref_graph.yaml` -> `configs/model/base_rgb_1024_refine_ref_graph.yaml`
- `configs/active/base_rgbd_1024.yaml` -> `configs/model/base_rgbd_1024.yaml`
- `configs/active/base_rgbd_1024_refine.yaml` -> `configs/model/base_rgbd_1024_refine.yaml`
- `configs/active/base_rgbd_1024_refine_ref.yaml` -> `configs/model/base_rgbd_1024_refine_ref.yaml`
- `configs/active/base_rgbd_1024_refine_ref_graph.yaml` -> `configs/model/base_rgbd_1024_refine_ref_graph.yaml`
- `configs/data/ecc_20260318_1k_1566.yaml`
- `configs/data/ecc_20260318_1k_32254.yaml`
- `configs/reference/reference_20260318_1k_13440.yaml`
- `configs/baseline/mask2former_rgb_smoke.yaml`

Delete `configs/query/`, `configs/train/`, `configs/runtime/`, and `configs/variant/`, plus all baseline configs that are only for the other two projects or one-off benchmarks.

## Scripts to keep

Keep only workflow scripts that GISEC still needs, and rename them to remove legacy branding:

- `scripts/experiments/common_runner.sh`
- `scripts/experiments/run_gisec_active.sh` -> `scripts/experiments/run_gisec.sh`
- `scripts/experiments/build_reference_split_cache.py` -> `scripts/experiments/build_reference_bank_cache.py`
- `scripts/experiments/precompute_baseline_depth_cache.py` -> `scripts/experiments/precompute_depth_cache.py`
- `scripts/experiments/precompute_baseline_instance_cache.py` -> `scripts/experiments/precompute_instance_cache.py`

Delete `scripts/audit/`, `scripts/maintenance/`, the whole one-off analysis tree under `scripts/analysis/`, and all other benchmark, fragment, query, legacy, and diagnostic scripts in `scripts/experiments/`.

## Delete categories relevant to GISEC

- CLI router and split entrypoints: `gisec/cli/_routing.py`, `gisec/cli/*_legacy.py`, `gisec/cli/*_query.py`, and any other project-router shims.
- Legacy fragment/query model code: `gisec/models/{gisec_model.py,fragment_bundle.py,graph_utils.py,prototype_unet.py,prototype_cache.py,query_common.py,query_depth_geometry.py,query_model.py,query_uq_backbone.py}`, `gisec/train/{train_query.py,query_targets.py}`, `gisec/engine/{query_factory.py,query_coarse_objects.py,query_object_split.py,query_reentry_contracts.py,query_runtime.py}`, `gisec/bridge/`, and `gisec/config/query_models.py`.
- Shared code that belongs to the other projects: `baseline/fragment_generator/`, `baseline/instance_fragment_generator/`, `baseline/local_merger/`, `baseline/reference_graph/` helper surfaces that only serve the legacy stack, and any other baseline subpackage that is not needed by the retained GISEC train/eval/infer path or the retained Mask2Former smoke helpers.
- Configs for other projects: `configs/query/`, `configs/train/`, `configs/runtime/`, `configs/variant/`, and all baseline configs except the GISEC smoke config listed above.
- Process and planning docs: delete any `docs/plans/`, `docs/reviews/`, and any `docs/experiments/` plan/lane/gate/metric-design doc if present in later cleanup work.
- Audit and stale outputs: `output/audit/`, `output/experiments/` unless a checkpoint is explicitly kept for reproducibility, and any audit/performance/report artifact that is not a final GISEC result note.
- Helper and one-off scripts: `scripts/audit/`, `scripts/maintenance/`, `scripts/analysis/*`, and non-workflow experiment runners/builders/diagnostics in `scripts/experiments/`.
- Cache and build noise: `__pycache__/`, `*.pyc`, `*.pyo`, `*.eggs`, `.pytest_cache/`, `dist/`, `build/`, `*.egg-info/`.
- Backup/temp variants: any file matching `*_v1`, `*_old`, `*_backup`, `*_copy`, `*_bak`, or `*_temp`.

## Rename map

- `ActiveVariantSpec` -> `GisecVariantSpec`
- `active_variant_names()` -> `gisec_variant_names()`
- `get_active_variant_spec()` -> `get_gisec_variant_spec()`
- `ActiveInstanceModel` -> `GisecModel`
- `train_active()` -> `train_gisec()`
- `eval_active()` -> `eval_gisec()`
- `infer_active()` -> `infer_gisec()`
- `prepare_active_input_*` -> `prepare_gisec_input_*`
- `_active_*` private helpers in `train_active.py` -> `_gisec_*`
- `prototype_bank` / `PrototypeBank` / `PrototypeBankSource` -> `reference_bank` / `ReferenceBank` / `ReferenceBankSource`
- `prototype_root` -> `reference_root`
- `prototype_cache` -> `reference_cache`
- `prototype_slot_count` -> `reference_slot_count`
- `prototype_topk` -> `reference_topk`
- `extract_query_part_key()` -> `extract_reference_part_key()`
- `query` wording in GISEC-only comments, logs, CLI help, and docs -> `reference` or `instance` wording
- `active` wording in GISEC-only comments, logs, CLI help, and docs -> `gisec` wording

## Root files to add or rewrite

- `README.md`
- `docs/architecture.md`
- `docs/experiment-results.md`
- `pyproject.toml`
- `environment.yml`
- `.gitignore`

## Notes

- This repo should not keep the CLI router layer.
- This repo should not keep query or legacy model families.
- Any remaining helper module that still imports query-only code must be rewritten to use the local GISEC copy before the split is finalized.
