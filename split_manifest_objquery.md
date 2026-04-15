# ObjQuery Split Manifest

This manifest covers the new standalone `object-query` repo with Python package `objquery`.

Key decision: generic `query_*` module names stay only where they describe the object-query domain itself. The leak names that must go away are `active`, `query-alpha`, `UQ`, and `legacy`.

## Repo Layout

New root files to create in the new repo:

- `README.md`
- `docs/architecture.md`
- `docs/experiment-results.md`
- `pyproject.toml`
- `environment.yml`
- `.gitignore`

Standard package layout under `src/objquery/`:

- `src/objquery/__init__.py`
- `src/objquery/cli/__init__.py`
- `src/objquery/config/__init__.py`
- `src/objquery/datasets/__init__.py`
- `src/objquery/engine/__init__.py`
- `src/objquery/models/__init__.py`
- `src/objquery/train/__init__.py`
- `src/objquery/common/__init__.py`
- `src/objquery/ops/__init__.py`

## Included Source Files

These current repo files are copied into the new `src/objquery/` tree:

### CLI

- `gisec/cli/train_query.py -> src/objquery/cli/train.py`
- `gisec/cli/eval_query.py -> src/objquery/cli/eval.py`
- `src/objquery/cli/infer.py` is new and should be built from the current eval/export path, since the current repo has no `infer_query.py`

### Config

- `gisec/config/io.py -> src/objquery/config/io.py`
- `gisec/config/query_models.py -> src/objquery/config/query_models.py`

### Dataset and prototype support

- `gisec/datasets/ecc_query_dataset.py -> src/objquery/datasets/ecc_query_dataset.py`
- `gisec/datasets/prototype_bank.py -> src/objquery/datasets/prototype_bank.py`

### Engine

- `gisec/engine/query_factory.py -> src/objquery/engine/query_factory.py`
- `gisec/engine/query_coarse_objects.py -> src/objquery/engine/query_coarse_objects.py`
- `gisec/engine/query_object_split.py -> src/objquery/engine/query_object_split.py`
- `gisec/engine/query_runtime.py -> src/objquery/engine/query_runtime.py`
- `gisec/engine/query_reentry_contracts.py -> src/objquery/engine/query_reentry_contracts.py`
- `gisec/engine/runtime.py -> src/objquery/engine/runtime.py` with only the evaluation/export helpers that the query stack uses

### Models

- `gisec/models/query_model.py -> src/objquery/models/query_model.py`
- `gisec/models/query_uq_backbone.py -> src/objquery/models/query_uq_backbone.py`
- `gisec/models/query_depth_geometry.py -> src/objquery/models/query_depth_geometry.py`
- `gisec/models/graph_head.py -> src/objquery/models/graph_head.py`
- `gisec/models/graph_utils.py -> src/objquery/models/graph_utils.py`
- `gisec/models/prototype_cache.py -> src/objquery/models/prototype_cache.py`

### Training

- `gisec/train/train_query.py -> src/objquery/train/train_query.py`
- `gisec/train/query_targets.py -> src/objquery/train/query_targets.py`

### Shared helpers copied into the package

- `baseline/common/coco_export.py -> src/objquery/common/coco_export.py`

### Ops

- `gisec/ops/connected_components.py -> src/objquery/ops/connected_components.py`
- `gisec/ops/csrc/* -> src/objquery/ops/csrc/*`

## Included Tests

Keep and migrate these tests into the new repo. Files that still say `query` can stay if the name is domain-specific; files that encode `alpha`, `UQ`, or `legacy` should be renamed.

### Keep as-is or with only import updates

- `tests/test_query_train_cli.py`
- `tests/test_query_eval_cli.py`
- `tests/test_query_runner_dry_run.py`
- `tests/test_query_cli_boundaries.py`
- `tests/test_query_factory.py`
- `tests/test_query_depth_geometry.py`
- `tests/test_query_targets.py`
- `tests/test_query_supervision_targets.py`
- `tests/test_query_coarse_objects.py`
- `tests/test_query_object_split.py`
- `tests/test_query_runtime.py`
- `tests/test_query_graph_variant_mapping.py`
- `tests/test_query_reentry_contracts.py`
- `tests/test_prototype_bank_loader.py`
- `tests/test_prototype_cache_source.py`
- `tests/test_prototype_routing.py`
- `tests/test_prototype_reference_routing.py`
- `tests/test_ownership_targets.py`
- `tests/test_ownership_supervision.py`
- `tests/test_eval_contracts.py`
- `tests/test_config_io.py`
- `tests/test_project_metadata.py`

### Rename because they carry alpha/UQ branding

- `tests/test_query_alpha_runner.py -> tests/test_objquery_runner.py`
- `tests/test_query_uq_backbone.py -> tests/test_objquery_backbone.py`
- `tests/test_query_uq_minibatch.py -> tests/test_objquery_minibatch.py`

### Delete from the new repo

- `tests/test_query_alpha_summary.py`
- `tests/test_query_experiment_docs.py`
- `tests/test_query_gates_doc.py`
- `tests/test_query_short_run_doc.py`
- `tests/test_query_full_run_entry_doc.py`
- `tests/test_query_metrics_doc.py`
- `tests/test_query_boundary_contract.py`
- `tests/test_query_reentry_boundaries.py`
- `tests/query/__init__.py`

## Included Configs

Keep the query config tree, but rename the alpha-only presets so they no longer say `alpha`.

### Keep

- `configs/query/README.md`
- `configs/query/model/query_small_resnet18.yaml`
- `configs/query/model/query_medium_resnet34.yaml`
- `configs/query/model/query_ref_resnet18.yaml`
- `configs/query/model/query_ref_resnet34.yaml`
- `configs/query/model/query_graph_resnet18.yaml`
- `configs/query/model/query_graph_resnet34.yaml`
- `configs/query/model/query_refgraph_resnet18.yaml`
- `configs/query/model/query_refgraph_resnet34.yaml`
- `configs/query/train/query_small_resnet18_full_train.yaml`
- `configs/query/train/query_medium_resnet34_full_train.yaml`
- `configs/query/train/query_ref_resnet18_full_train.yaml`
- `configs/query/train/query_ref_resnet34_full_train.yaml`
- `configs/query/train/query_graph_resnet18_full_train.yaml`
- `configs/query/train/query_graph_resnet34_full_train.yaml`
- `configs/query/train/query_refgraph_resnet18_full_train.yaml`
- `configs/query/train/query_refgraph_resnet34_full_train.yaml`
- `configs/query/eval/query_small_resnet18_full_eval.yaml`
- `configs/query/eval/query_medium_resnet34_full_eval.yaml`
- `configs/query/eval/query_ref_resnet18_full_eval.yaml`
- `configs/query/eval/query_ref_resnet34_full_eval.yaml`
- `configs/query/eval/query_graph_resnet18_full_eval.yaml`
- `configs/query/eval/query_graph_resnet34_full_eval.yaml`
- `configs/query/eval/query_refgraph_resnet18_full_eval.yaml`
- `configs/query/eval/query_refgraph_resnet34_full_eval.yaml`

### Rename

- `configs/query/train/alpha_short_run.yaml -> configs/query/train/objquery_short_run.yaml`
- `configs/query/eval/alpha_full_eval.yaml -> configs/query/eval/objquery_full_eval.yaml`

### Delete from the new repo

- `configs/active/*`
- `configs/baseline/*`
- `configs/data/*`
- `configs/reference/*`
- `configs/runtime/*`
- `configs/train/*`
- `configs/variant/*`

## Included Scripts

Keep only the normal workflow runner pieces for ObjQuery.

### Keep

- `scripts/experiments/common_runner.sh`

### Rename

- `scripts/experiments/run_gisec_query_uq.sh -> scripts/experiments/run_objquery.sh`

### Delete from the new repo

- `scripts/audit/*`
- `scripts/maintenance/*`
- `scripts/analysis/summarize_query_alpha_ladder.py`
- every other `scripts/analysis/*` helper
- every other `scripts/experiments/*` runner/builder that is not the shared runner or the ObjQuery runner

## Delete Categories

These categories should be removed from the object-query repo because they belong to the monorepo or to the other two projects:

- `docs/archive/plans/*`
- `docs/archive/reviews/*`
- `docs/archive/experiments/*`
- `docs/results/*` entries that are cross-project, audit, or process artifacts
- `output/audit/*`
- `output/experiments/*`
- `Project_Summary_Report.md`
- all caches and build noise: `__pycache__`, `*.pyc`, `*.pyo`, `.pytest_cache`, `*.egg-info`, `build/`, `dist/`
- backup/temp files matching `*_v1`, `*_old`, `*_backup`, `*_copy`, `*_bak`, `*_temp`
- all non-ObjQuery code outside the files listed above, especially:
  - `gisec/active/*`
  - `gisec/bridge/*`
  - `gisec/cli/train.py`
  - `gisec/cli/eval.py`
  - `gisec/cli/infer.py`
  - `gisec/cli/_routing.py`
  - `gisec/cli/*legacy*.py`
  - `gisec/config/variants.py`
  - `gisec/models/gisec_model.py`
  - `gisec/models/query_common.py`
  - `gisec/models/prototype_unet.py`
  - `gisec/train/train_active.py`
  - `gisec/train/train_gisec.py`
  - all unrelated `baseline/*` subprojects that are not copied into `src/objquery/common/`, `src/objquery/datasets/`, `src/objquery/models/`, `src/objquery/engine/`, `src/objquery/train/`, or `src/objquery/ops/`

## Rename Rules

These names should be rewritten to ObjQuery naming in code, configs, tests, docs, and shell scripts:

- `QueryModelSpec -> ObjectQueryModelSpec`
- `UQModel -> ObjectQueryModel`
- `UQBackbone -> ObjectQueryBackbone`
- `active_alpha_model_ids -> objquery_base_model_ids`
- `deferred_query_model_ids -> objquery_deferred_model_ids`
- `later_phase_model_ids -> objquery_deferred_model_ids`
- `is_alpha_enabled_model_id -> is_objquery_enabled_model_id`
- `get_query_model_spec -> get_objquery_model_spec`
- `build_query_model -> build_objquery_model`
- `run_uq_minibatch -> run_objquery_minibatch`
- `run_uq_eval -> run_objquery_eval`
- `UQRunSummary -> ObjQueryRunSummary`
- `query_alpha` wording in docs/help/logs -> `ObjQuery`
- `legacy_*` graph-mapping strings inside `query_runtime.py` -> ObjQuery-local names, not legacy identifiers
- variant/model ids that currently say `alpha` or `UQ`:
  - `alpha_short_run -> objquery_short_run`
  - `alpha_full_eval -> objquery_full_eval`
  - `query_alpha_official -> objquery_official`
- `query_alpha` result/docs headings -> `ObjQuery`
- keep or rewrite ObjQuery-owned `docs/results/*`; only delete entries that are cross-project, audit, or process artifacts

## Notes

- `configs/query/` stays as the config tree name because the public model ids are still query-domain ids; only the alpha-only presets and all brand leaks are renamed.
- `tests/test_eval_contracts.py`, `tests/test_config_io.py`, and `tests/test_project_metadata.py` stay because the ObjQuery repo still needs the copied config loader, evaluation helper, and package metadata checks.
- `gisec/models/prototype_cache.py` is included because `gisec/models/graph_utils.py` imports it.
- `gisec/engine/runtime.py` is not copied wholesale; only the evaluation/export helpers needed by query eval/infer should move into the new repo.
