# Project Summary Report

This report is based only on repository evidence: source code under `gisec/` and `baseline/`, configs under `configs/`, process documents under `docs/`, tests under `tests/`, dataset metadata under `datasets/`, and curated result notes under `docs/results/` and `docs/archive/experiments/`. Binary images, depth arrays, and checkpoints are described only through their surrounding manifests, configs, and summary files.

## 1. Project Overview and Background

The project is named **GISEC: Graph-based Instance Segmentation for Electronic Components**. That name appears in the main README and matches the `gisec` package name. The repository addresses instance segmentation of electronic components in an RGB-first research setting, and it includes training, evaluation, inference, experiment runners, and result exporters rather than a packaged end-user application. Sources: `README.md`; `pyproject.toml`; `gisec/cli/train.py`; `gisec/cli/eval.py`; `gisec/cli/infer.py`.

The broader domain is computer vision, specifically instance segmentation and COCO-style evaluation. The codebase contains three distinct research surfaces:

- a current staged Mask2Former-based active line
- an older prototype-conditioned fragment-first graph line kept as the legacy path
- an isolated `query-alpha` object-first branch

The repository README and results index both say that the active line is the current benchmark surface, while `query-alpha` is archived rather than the default benchmark surface. Sources: `README.md`; `docs/results/README.md`; `gisec/cli/_routing.py`; `gisec/config/query_models.py`.

The intended users visible in the checked-in materials are internal researchers or engineers. Every primary interface is a CLI, a YAML config, or a shell runner. The main beneficiaries are therefore users who need to train, compare, and analyze segmentation models for electronic-component scenes. Sources: `gisec/cli/*.py`; `scripts/experiments/*.sh`; `configs/**/*.yaml`.

The project motivation changed over time and is documented in the process files. `docs/archive/plans/research-context.md` says the team had a stable lightweight baseline inherited from earlier "magformer" style work, but more complex attention variants were not beating it. Early plans therefore pushed toward prototype-guided graph reasoning with stronger reference use. Later results and summaries show a second shift: the repository benchmarked stronger standard detectors, found Mask2Former RGB at 1024 to outperform the earlier RGB Mask R-CNN benchmark and the weak fragment stages, and promoted the staged Mask2Former line to the current official path. Sources: `docs/archive/plans/research-context.md`; `docs/archive/plans/stage1-research-plan.md`; `docs/results/2026-03-29-rgb-phase1-backbone-summary.md`; `docs/results/2026-03-29-rgb-phase1-backbone-summary.json`; `docs/archive/reviews/2026-04-09-gisec-project-summary.md`; `README.md`.

The project builds upon or responds to several concrete reference systems already implemented in the repository: Mask2Former, Mask R-CNN, U-Net-family dense predictors, a prototype bank and reference-conditioned graph pipeline, and the later `query-alpha` object-first branch. These are not abstract mentions; they correspond to runnable code under `baseline/`, `gisec/models/`, `gisec/train/`, and `configs/`. Sources: `baseline/mask2former/adapter.py`; `baseline/mask_rcnn/train.py`; `baseline/unet/model.py`; `gisec/models/prototype_unet.py`; `gisec/models/gisec_model.py`; `gisec/models/query_uq_backbone.py`.

## 2. Application Scenarios

The clearest application scenario is offline segmentation research on electronic-component datasets. A user points the CLI at a dataset root, optionally a prototype/reference root, selects a variant, and runs training, evaluation, or inference. The outputs are checkpoints, COCO metrics, failure summaries, overlays, and standardized `run_summary.json` files. Sources: `gisec/cli/train.py`; `gisec/cli/eval.py`; `gisec/cli/infer.py`; `gisec/train/train_active.py`; `gisec/train/train_gisec.py`; `gisec/engine/runtime.py`.

A second scenario is controlled benchmark comparison. The repository contains shell runners for baseline sweeps, official active ladders, and query-alpha runs. The tests for those runners show that stages are chained through saved checkpoints and that completed stages are skipped while resumable incomplete stages can be continued. This makes the repo suitable for repeatable ablation studies rather than only one-off training jobs. Sources: `scripts/experiments/run_baseline_benchmarks.sh`; `scripts/experiments/run_gisec_active.sh`; `scripts/experiments/run_gisec_query_uq.sh`; `tests/test_baseline_reset_ladder_scripts.py`; `tests/test_active_runner_dry_run.py`.

The environmental constraints are research-oriented rather than application-facing. Most official configs use `1024` pixel inputs, multi-epoch runs, disk-heavy datasets, and optional prototype banks with many reference views. The repository includes both `ddp_1gpu.yaml` and `ddp_6x3090.yaml`, which shows that the code is meant to scale from single-GPU to multi-GPU training. The curated result notes also document long wall times and throughput recovery work, which reinforces that this is an offline training pipeline. Sources: `configs/active/*.yaml`; `configs/train/full_legacy_20ep.yaml`; `configs/runtime/ddp_1gpu.yaml`; `configs/runtime/ddp_6x3090.yaml`; `docs/results/2026-04-06-throughput-recovery-and-g3-restart.md`; `docs/results/2026-04-08-active-rgb-resume-and-throughput-recovery.md`.

The code is explicitly designed to handle difficult separation cases such as touching instances, under-segmentation, over-segmentation, and uncertain splits inside one coarse object. The legacy line uses boundary logits, ownership offsets, contact edges, bridge edges, and constrained graph merges. The query-alpha line first finds coarse objects and then splits them with core peaks, boundary evidence, and ownership offsets. The instance-local documents also measure split and merge error counts directly. Sources: `gisec/models/graph_utils.py`; `gisec/graph_refiner.py`; `gisec/engine/query_coarse_objects.py`; `gisec/engine/query_object_split.py`; `docs/results/2026-03-30-rgb-phase23-instance-local-reset-summary.json`; `docs/archive/experiments/gisec-query-metrics.md`.

## 3. Tasks and Objectives

The repository supports both core research tasks and auxiliary infrastructure tasks. The table below summarizes the distinct tasks that are clearly implemented in code.

| Task | Main entry points | Inputs | Outputs | Success criteria visible in repo |
| --- | --- | --- | --- | --- |
| Dataset loading and target construction | `gisec/datasets/ecc_query_dataset.py`; `baseline/common/dataset.py`; `gisec/datasets/prototype_bank.py` | RGB images, optional depth maps, COCO annotations, prototype/reference banks | batched tensors and supervision targets such as `fg`, `boundary`, `core`, `affinity`, `ownership`, instance masks, and prototype caches | loaders run without contract errors and expose the targets required by the selected branch |
| Active staged training and evaluation | `gisec/cli/train.py`; `gisec/train/train_active.py`; `scripts/experiments/run_gisec_active.sh` | dataset root, active variant config, optional init checkpoint, optional prototype root | checkpoints, `metrics_log.jsonl`, `run_state.json`, `run_summary.json`, COCO metrics, eval artifacts | higher `segm/AP`, `bbox/AP`, and `boundary/IoU`; stage completion with success status |
| Legacy prototype-conditioned graph training and evaluation | `gisec/cli/train_legacy.py`; `gisec/train/train_gisec.py`; `gisec/models/gisec_model.py` | dataset root, prototype root, legacy variant spec, graph thresholds | checkpoints, graph diagnostics, run summaries, COCO metrics, merge previews | improved segmentation and controlled split/merge errors under the shared metric/export surface |
| Query-alpha object-first training and evaluation | `gisec/cli/train_query.py`; `gisec/cli/eval_query.py`; `gisec/train/train_query.py` | dataset root, `query_small_resnet18` or `query_medium_resnet34`, query configs | query checkpoints, eval-only summaries, failure summaries, COCO metrics, pathology diagnostics | gate-based improvement over legacy on the shared evaluation surface |
| Baseline benchmarking and ablation | `scripts/experiments/run_baseline_benchmarks.sh`; `baseline/*` | dataset root plus baseline model configs | per-baseline checkpoints and summaries | identify strong reference baselines before promoting custom modules |
| Result export and analysis | `gisec/engine/runtime.py`; `gisec/engine/query_runtime.py`; `docs/results/*.md`; `docs/results/*.json` | predictions, annotations, training/eval logs | `metrics.cocoeval.json`, `failure_summary.json`, rendered previews, markdown closeouts, summary JSONs | standardized comparison across runs and documented go/no-go decisions |

The core tasks are the first four rows: dataset/target construction, model training, model evaluation, and instance prediction. The baseline benchmark surface is also core in practice because the process documents repeatedly use it to decide whether custom modules should continue. The export, runner, and analysis surfaces are auxiliary, but they are important because the repository is organized around formal experiment ladders rather than ad hoc scripts. Sources: `docs/archive/plans/2026-03-19-gisec-master-plan.md`; `docs/archive/plans/2026-03-23-gisec-query-master-plan.md`; `docs/results/README.md`; `tests/test_active_run_state.py`; `tests/test_query_eval_cli.py`.

The project also has clear stage structure. Across the active, legacy, and query branches, the repeated stages are:

- prepare or load a dataset and, when needed, a prototype/reference bank
- train a model under a specific config
- run evaluation or inference on a split
- export COCO-style metrics and debugging artifacts
- decide whether the next experiment stage is allowed

This stage-wise pattern appears in the CLIs, the shell runners, the result summaries, and the formal experiment ladder documents. Sources: `scripts/experiments/run_gisec_active.sh`; `scripts/experiments/run_gisec_query_uq.sh`; `docs/archive/experiments/gisec-query-ladder.md`; `docs/results/2026-03-30-rgb-phase23-fragment-reset-summary.json`; `docs/results/2026-04-08-active-rgb-resume-and-throughput-recovery.md`.

## 4. Methods and Technical Approach

### 4.1 System Architecture

The repository is not one monolithic model. It is a shared experiment platform with multiple model families behind a routing layer. `gisec/cli/_routing.py` decides whether a generic `gisec.cli.train`, `gisec.cli.eval`, or `gisec.cli.infer` invocation should go to the current active line or the legacy line. Query-alpha uses separate CLI entry points and separate model specs so it cannot silently fall back to the legacy implementation. Sources: `gisec/cli/_routing.py`; `gisec/cli/train.py`; `gisec/cli/eval.py`; `gisec/cli/train_query.py`; `gisec/config/query_models.py`.

At a high level, the data flow is:

1. A loader reads RGB images, COCO annotations, and sometimes a prototype bank.
2. A model family produces dense outputs such as foreground masks, boundaries, ownership offsets, or query masks.
3. Some branches convert those dense outputs into instances through connected components, local refinement, graph scoring, or object splitting.
4. A shared export layer writes predictions, COCO metrics, failure summaries, and `run_summary.json`.

This pattern is visible across `baseline/common/dataset.py`, `gisec/datasets/ecc_query_dataset.py`, `gisec/train/train_active.py`, `gisec/train/train_gisec.py`, `gisec/train/train_query.py`, `gisec/engine/runtime.py`, and `gisec/engine/query_runtime.py`.

### 4.2 Active Mainline

The current active line is a staged Mask2Former-based system. `gisec/train/train_active.py` builds the active model by calling the repository's Mask2Former adapter and then wrapping it in `ActiveInstanceModel`. The active variant spec controls whether local refinement, reference rescue, or graph rescue are enabled. Sources: `gisec/train/train_active.py`; `gisec/active/config.py`; `gisec/active/model.py`; `baseline/mask2former/adapter.py`.

`ActiveInstanceModel` has three layers of behavior:

- a Mask2Former backbone for coarse instance prediction
- an optional `LocalRefinementModule` that refines a cropped query region using the coarse mask probability and projected features
- an optional `LocalGraphRescueHead` that scores local graph edges with `GraphEdgeScorer`

When reference rescue is enabled, the local refiner encodes query crops and reference views, ranks reference views by descriptor similarity, mixes the top views, and predicts both refined masks and reference-match logits. Sources: `gisec/active/model.py`; `gisec/models/graph_head.py`.

The active trainer also encodes control-plane behavior that matters to reproducibility. It supports stagewise initialization from earlier checkpoints, run-state tracking, standardized metrics logging, eval export, and sidecar stage locking. Tests confirm that it fails closed on stale or malformed locks and that eval-only query output must be kept separate from checkpoint directories. Sources: `gisec/train/train_active.py`; `tests/test_active_run_state.py`; `tests/test_query_eval_cli.py`.

### 4.3 Legacy Fragment-First Graph Line

The original GISEC line is implemented around `GISECModel`, `PrototypeConditionedUNetBackbone`, `GraphEdgeScorer`, and the graph-building utilities in `gisec/models/graph_utils.py`. The backbone is a U-Net-like encoder-decoder with RGB input, optional depth geometry features, and optional prototype-conditioning at the bottleneck and high-resolution stages. The prototype cache stores bottleneck, high-resolution, and depth-conditioned prototype slots together with shape statistics and routing metadata. Sources: `gisec/models/gisec_model.py`; `gisec/models/prototype_unet.py`; `gisec/models/prototype_cache.py`; `gisec/datasets/prototype_bank.py`.

The fragment-first pipeline works in two steps. First, `fragments_from_logits()` turns dense foreground and boundary predictions into fragments by running connected components on `foreground AND NOT boundary`. If ownership offsets are available, the same function can split one connected component into multiple fragments by voting for ownership landing points and assigning pixels to seed centers. Second, `build_graph_batch()` constructs graph nodes and edges from those fragments, attaches contact and bridge edges, and prepares edge features, edge targets, shape priors, and diagnostics. Sources: `gisec/models/graph_utils.py`; `gisec/models/gisec_model.py`.

Edge scoring uses a small message-passing MLP rather than a full graph neural network stack. `GraphEdgeScorer` projects node features, builds symmetric edge messages, aggregates them back to nodes, and predicts one merge logit per edge. After scoring, `merge_instances_from_edge_scores()` merges fragments by union-find. In constrained mode it rejects merges that violate empirical shape ranges or cross too much predicted boundary. This is one of the clearest custom algorithmic parts of the repository. Sources: `gisec/models/graph_head.py`; `gisec/models/graph_utils.py`.

### 4.4 Query-Alpha Object-First Branch

The query-alpha branch is intentionally isolated from the legacy variant system. `gisec/config/query_models.py` only enables `query_small_resnet18` and `query_medium_resnet34` in the current alpha stage, and reserves `query_ref_*`, `query_graph_*`, and `query_refgraph_*` for later phases that are not executable yet. `gisec/engine/query_factory.py` builds `UQModel`, which wraps `UQBackbone`. Sources: `gisec/config/query_models.py`; `gisec/engine/query_factory.py`; `gisec/models/query_model.py`.

`UQBackbone` is a fixed early-fusion encoder-decoder. It uses either `resnet18` (`query_small_resnet18`) or `resnet34` (`query_medium_resnet34`), replaces the first convolution with a 6-channel input layer, and concatenates RGB with depth geometry features. The decoder predicts four dense heads: foreground, boundary, core heatmap, and ownership offsets. This matches the process documents, which say alpha should stay narrow and only compare small versus medium scale under one encoder family and one fusion strategy. Sources: `gisec/models/query_uq_backbone.py`; `docs/archive/plans/2026-03-23-gisec-query-master-plan.md`; `docs/archive/plans/2026-03-23-02-gisec-query-uq-backbone.md`.

Instance formation in query-alpha is object-first. `build_coarse_objects()` first finds coarse connected foreground objects and can split large ones with boundary-based seeding. `split_coarse_object()` then detects core peaks, limits peak count dynamically by object area and span, and assigns pixels using a combined score based on geometric distance, ownership landing distance, and boundary-line cost. `predict_instance_map()` applies this per coarse object and produces query-specific pathology summaries such as object counts, split counts, and average cores per object. Sources: `gisec/engine/query_coarse_objects.py`; `gisec/engine/query_object_split.py`; `gisec/engine/query_runtime.py`.

### 4.5 Shared Data and Evaluation Stack

The dataset side is split by branch but shares the same semantic goal. `ECCGraphDataset` produces query RGB, instance masks, and dense supervision targets for the legacy line. `BaselineInstanceDataset` produces RGB tensors and instance targets for the active and baseline lines. `PrototypeBankSource` resolves the correct prototype bank for a query image, caches built prototype features, and can inject query-derived shape priors. Sources: `gisec/datasets/ecc_query_dataset.py`; `baseline/common/dataset.py`; `gisec/engine/runtime.py`; `gisec/datasets/prototype_bank.py`.

The evaluation/export layer is deliberately standardized. `RunSummary` in `gisec/engine/runtime.py` captures variant, checkpoint, metrics, runtime settings, and environment metadata. `evaluate_and_export()` writes COCO metrics, failure summaries, and rendered previews. Query-alpha has its own `UQRunSummary`, but it follows the same pattern. This shared export contract is why results from very different model families are still directly comparable in the stored summaries. Sources: `gisec/engine/runtime.py`; `gisec/engine/query_runtime.py`; `docs/archive/experiments/gisec-query-metrics.md`.

### 4.6 External Libraries and Technical Stack

The repository depends mainly on PyTorch, torchvision, Hugging Face Transformers, OpenCV, NumPy, SciPy, PyYAML, pycocotools, and Matplotlib. Their roles are visible in code:

- PyTorch and torchvision provide the training framework and base encoders
- Transformers provides Mask2Former
- OpenCV is used for connected components, distance transforms, morphology, and geometric line operations
- pycocotools provides COCO metrics
- PyYAML drives config loading
- NumPy and SciPy support tensor-independent numeric processing
- Matplotlib is present in the environment for visualization/reporting support

Sources: `pyproject.toml`; `environment.yml`; `baseline/mask2former/adapter.py`; `gisec/models/query_uq_backbone.py`; `gisec/models/graph_utils.py`.

## 5. Innovations and Contributions

The repository contains real algorithmic experimentation, but the novelty should be described carefully. The strongest empirical results in the curated result notes come from a staged Mask2Former line rather than from the fully custom graph pipeline. That means the repository's main contribution is best understood as a research and engineering platform that compared several candidate directions and retained the one that worked best. Sources: `docs/results/README.md`; `docs/results/2026-04-12-active-rgb-official-ladder-summary.md`; `docs/results/2026-04-06-phase-a-baseline-reset-closeout.md`.

The table below separates implemented contributions from planned or only partially validated ideas.

| Contribution | Evidence in repo | Assessment |
| --- | --- | --- |
| Prototype-conditioned U-Net backbone with reference slot routing | `PrototypeConditionedUNetBackbone` routes prototype bottleneck and high-resolution features into the query backbone and records routing metadata | Implemented and central to the legacy line, but not the current best-performing path |
| Fragment-to-graph instance assembly with contact and bridge edges | `fragments_from_logits()`, `build_graph_batch()`, `GraphEdgeScorer`, and `merge_instances_from_edge_scores()` implement a custom fragment graph pipeline | Implemented and algorithmically distinctive, but stored results show it underperformed the promoted active line |
| Object-first query-alpha backbone and split rule | `UQBackbone`, `build_coarse_objects()`, and `split_coarse_object()` implement a separate object-first branch | Implemented as an archived experimental branch; the repo does not contain stored full official result artifacts for it |
| Staged active line that adds local refine, reference rescue, and graph rescue on top of Mask2Former | active configs, active model wrapper, active runner, and curated result notes show a controlled stage ladder | Implemented and empirically central to the current repo state |
| Formal experiment gating and summary discipline | result markdown files, result JSONs, run summaries, and ladder docs encode explicit promotion rules and go/no-go decisions | Practical engineering contribution rather than a new segmentation algorithm |

The most defensible innovations are therefore practical and architectural:

- the repository makes multiple segmentation lines comparable under one result and export contract
- it implements a nontrivial prototype-conditioned graph pipeline instead of only benchmarking standard models
- it captures explicit stage gates so experimental promotion decisions are documented rather than implicit

Sources: `gisec/engine/runtime.py`; `gisec/engine/query_runtime.py`; `docs/archive/experiments/gisec-query-gates.md`; `docs/results/*.md`; `docs/results/*.json`.

At the same time, the code and results do not justify stronger claims such as "the project introduced a clearly superior new algorithm." The curated evidence says the strongest official result is a staged Mask2Former RGB model, while the legacy graph line remained below that level and the query-alpha line stayed gated or archived. Sources: `docs/results/2026-03-29-rgb-phase1-backbone-summary.json`; `docs/results/2026-03-30-rgb-phase23-fragment-reset-summary.json`; `docs/results/2026-03-31-rgb-phase23-instance-local-stage2-summary.md`; `docs/archive/experiments/gisec-query-ladder.md`; `docs/archive/experiments/gisec-query-gates.md`.

## 6. Experimental Design

### 6.1 Datasets and Experimental Assets

The repository contains at least three important data assets for experiments.

| Asset | Observed role | Concrete metadata found in repo | Source |
| --- | --- | --- | --- |
| `datasets/20260318_1K_1566` | smaller official ECC dataset used by many baseline and active runs | `build_stats.json` records `2264` total tasks, `2264` successful tasks, `1566` written images, and `4528` total views; the split directories contain `1261` train images, `149` val images, and `156` test images | `datasets/20260318_1K_1566/build_stats.json`; `datasets/20260318_1K_1566/dataset_info.json`; `datasets/20260318_1K_1566/images/train/`; `datasets/20260318_1K_1566/images/val/`; `datasets/20260318_1K_1566/images/test/` |
| `datasets/20260318_1K_32254` | larger ECC dataset used for later large-scale runs | `build_stats.json` records `2274` total tasks, `2221` successful tasks, `53` failed tasks, `32254` written images, and `35536` total views; the split directories contain `25654` train images, `3276` val images, and `3324` test images | `datasets/20260318_1K_32254/build_stats.json`; `datasets/20260318_1K_32254/dataset_info.json`; `datasets/20260318_1K_32254/images/train/`; `datasets/20260318_1K_32254/images/val/`; `datasets/20260318_1K_32254/images/test/` |
| `datasets/20260318_1K_13440` | prototype/reference bank for reference-conditioned experiments | manifest contains `48` parts; the reference config sets `reference_max_views: 16` and `reference_view_sampler: pose_farthest` | `datasets/20260318_1K_13440/manifest.json`; `configs/reference/reference_20260318_1k_13440.yaml` |

The dataset quality checks are partially documented. `alignment_report.json` and `scene_qc_report.json` exist for the ECC datasets, and `build_stats.json` plus `dataset_info.json` store build timestamps and git revisions. The exact semantic data-collection process and human annotation workflow are [Not Found], but mask-parity and alignment checks are clearly part of the dataset build pipeline. Sources: `datasets/20260318_1K_1566/alignment_report.json`; `datasets/20260318_1K_1566/scene_qc_report.json`; `datasets/20260318_1K_32254/alignment_report.json`; `datasets/20260318_1K_32254/scene_qc_report.json`.

### 6.2 Metrics and Promotion Rules

The common evaluation surface is COCO-style AP plus repository-specific diagnostics. Across the active and legacy paths, the most important reported metrics are `segm/AP`, `bbox/AP`, and `boundary/IoU`. The query-alpha docs add instance-count calibration, failure redistribution, and object pathology summaries such as `pred_count_mean`, `gt_count_mean`, `best_mask_iou_mean`, and `split_count_mean`. Sources: `gisec/engine/runtime.py`; `gisec/engine/query_runtime.py`; `docs/archive/experiments/gisec-query-metrics.md`.

Promotion is intentionally gate-based rather than open-ended. The fragment-reset and instance-local summaries record explicit `gate_passed` decisions. The query-alpha docs define `Gate A` and `Gate B` as relative performance gates, not vanity thresholds. This is a notable methodological choice because it keeps the repo focused on controlled comparisons instead of one-off best numbers. Sources: `docs/results/2026-03-30-rgb-phase23-fragment-reset-summary.json`; `docs/results/2026-03-31-rgb-phase23-instance-local-stage2-summary.json`; `docs/archive/experiments/gisec-query-gates.md`; `docs/archive/experiments/gisec-query-ladder.md`.

### 6.3 Main Configurations

The repository stores configuration files for each main experiment family. The most important ones are summarized below.

| Family | Key settings observed in config | Source |
| --- | --- | --- |
| Active official RGB ladder | `image_size: 1024`, `batch: 1`, `num_workers: 4`, `epochs: 20`, `learning_rate: 1e-4`; stage flags differ by `refine`, `ref`, and `graph` toggles | `configs/active/base_rgb_1024.yaml`; `configs/active/base_rgb_1024_refine.yaml`; `configs/active/base_rgb_1024_refine_ref.yaml`; `configs/active/base_rgb_1024_refine_ref_graph.yaml` |
| Legacy full run | `image_size: 1024`, `batch: 4`, `num_workers: 4`, `epochs: 20`, `lr: 1e-4`, `fragment_fg_threshold: 0.55`, `fragment_boundary_threshold: 0.7`, `min_area: 256` | `configs/train/full_legacy_20ep.yaml` |
| Legacy recovery smoke | reduced reference views and explicit routing/profiling controls for recovery experiments | `configs/train/recovery_smoke_1024.yaml` |
| Phase-A Mask2Former benchmark | `epochs: 20`, `amp: true`, `grad_accum_steps: 2`, `lr: 1e-4`, pretrained Swin-T Mask2Former | `configs/baselines/mask2former_swin_t_1024_phasea_full.yaml` |
| Phase-A Mask R-CNN benchmark | `epochs: 20`, `amp: true`, `grad_accum_steps: 2`, `lr: 1e-3`, `momentum: 0.9` | `configs/baselines/mask_rcnn_r50_1024_phasea_full.yaml` |
| Query-alpha short run | CPU-only smoke config with `image_size: 64`, `batch_size: 1`, `max_train_steps: 8`, `max_val_images: 4` | `configs/query/train/alpha_short_run.yaml` |
| Query-alpha full eval | CPU-only evaluation config with `image_size: 64`, `batch_size: 1`, `max_val_images: 16` | `configs/query/eval/alpha_full_eval.yaml` |

### 6.4 Software and Hardware Environment

The software environment is partially explicit. `pyproject.toml` declares the `gisec` package and a minimal dependency set, while `environment.yml` pins a richer research environment that includes Python `3.13`, PyTorch `2.10.0`, torchvision `0.25.0`, torchaudio `2.10.0`, Transformers `4.57.6`, OpenCV, pycocotools, SciPy `1.15.2`, Matplotlib `3.10.7`, and `ninja`. Sources: `pyproject.toml`; `environment.yml`.

The exact hardware used for every stored experiment is [Not Found]. What can be stated from repository evidence is that the code supports both single-GPU and multi-GPU distributed launches, and that one runtime config is explicitly named for six RTX 3090 GPUs. Sources: `configs/runtime/ddp_1gpu.yaml`; `configs/runtime/ddp_6x3090.yaml`; `gisec/train/train_gisec.py`.

## 7. Experimental Results

The stored results tell a clear story: the repository's strongest official result is a staged active RGB model, the original legacy line is much weaker, the fragment and instance-local exploratory stages exposed clear headroom but did not close it with learned models, and the query-alpha branch has protocols and tests but no stored full official result bundle.

### 7.1 Backbone Benchmark and Early Active Surface

The Phase-A backbone benchmark promoted Mask2Former over Mask R-CNN for RGB at `1024`. The benchmark summary JSON records Mask2Former as the winner.

| Experiment | segm/AP | bbox/AP | boundary/IoU | wall_time_sec | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Mask2Former RGB `1024` Phase A full | 0.5459 | 0.4933 | 0.1894 | 19390 | `docs/results/2026-03-29-rgb-phase1-backbone-summary.md`; `docs/results/2026-03-29-rgb-phase1-backbone-summary.json` |
| Mask R-CNN RGB `1024` Phase A full | 0.5194 | 0.4908 | 0.1470 | 6838 | `docs/results/2026-03-29-rgb-phase1-backbone-summary.md`; `docs/results/2026-03-29-rgb-phase1-backbone-summary.json` |

The earlier active pilot then reproduced a very similar level, with `base_rgb_1024` reaching `segm/AP = 0.5450772353135926`. Sources: `docs/results/2026-03-28-gisec-active-pilot.json`.

### 7.2 Current Official Active RGB Ladder

The strongest completed official results are summarized in `docs/results/2026-04-12-active-rgb-official-ladder-summary.md`. The best completed stage is `base_rgb_1024_refine`, not the later `ref` or `ref_graph` stages. The still-running 2026-04-13 rerun is reported separately below and is not included in this table.

| Stage | segm/AP | bbox/AP | boundary/IoU | train wall_time_sec | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| `base_rgb_1024` | 0.5495995386078752 | 0.5140047832669681 | 0.19392113294585225 | 32775 | `docs/results/2026-04-12-active-rgb-official-ladder-summary.md` |
| `base_rgb_1024_refine` | 0.5761366653940664 | 0.5155950306627068 | 0.25118819472440657 | 50120 | `docs/results/2026-04-12-active-rgb-official-ladder-summary.md` |
| `base_rgb_1024_refine_ref` | 0.5747495053887953 | 0.5141577902090311 | 0.25009854065035914 | 55372 | `docs/results/2026-04-12-active-rgb-official-ladder-summary.md` |
| `base_rgb_1024_refine_ref_graph` | 0.5745757912158308 | 0.5153391542700804 | 0.24883020894828495 | 75035 | `docs/results/2026-04-12-active-rgb-official-ladder-summary.md` |

These numbers support three measured conclusions:

- local refinement clearly helped, because `base_rgb_1024_refine` improved over `base_rgb_1024`
- the later reference and graph rescue stages did not beat the best refine-only stage in the stored official run
- the current best official result in the repository is the refine stage, with `segm/AP` slightly above `0.576`

### 7.3 2026-04-13 Active RGB Rerun Status

The 2026-04-13 active RGB rerun is still underway. The `base_rgb_1024` training state file reports `status = running` and `last_finite_checkpoint = resume_last.pth`. The launcher log shows an initial failed attempt on `NonFiniteActiveTrainingError` followed by a retry that is still progressing. There is no final `run_summary.json` yet, so the result fields remain empty here. Sources: `output/experiments/2026-04-13-rgb-full-rerun/active_official/active_rgb_official/train/base_rgb_1024/run_state.json`; `output/experiments/2026-04-13-rgb-full-rerun/recovery_launcher/launcher.log`.

| Stage | Status | segm/AP | bbox/AP | boundary/IoU | train wall_time_sec | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `base_rgb_1024` rerun | running | [Not Found] | [Not Found] | [Not Found] | [Not Found] | `output/experiments/2026-04-13-rgb-full-rerun/active_official/active_rgb_official/train/base_rgb_1024/run_state.json`; `output/experiments/2026-04-13-rgb-full-rerun/recovery_launcher/launcher.log` |

### 7.4 Legacy Baseline and Fragment/Instance-Local Branch

The repaired legacy baseline remained materially below the active official result. The legacy `legacy_prototype_unet_baseline_best_eval` summary reports `segm/AP = 0.4153300741961166` and `bbox/AP = 0.3647817709084885`. Source: `docs/results/2026-04-06-phase-a-baseline-reset-closeout.md`.

The fragment-reset branch produced a strong negative result. Its own summary says the Stage-2 fragment gate failed and Stage 3 stayed off. The baseline active model at that time was `segm/AP = 0.5451`, but the fragment-quality metrics showed very high overflow and impurity:

- `overflow_crop_rate = 0.9467429577464789`
- `impure_fragment_rate = 0.6314997102904792`
- `gate_passed = false`

Source: `docs/results/2026-03-30-rgb-phase23-fragment-reset-summary.json`.

The instance-local analysis exposed a large upper bound but also a large learning gap. The oracle owner-union result reached `segm/AP = 0.8489273379592495` and `boundary/IoU = 0.9227048670589119`, which shows that the stage had theoretical headroom. However, the later learned owner-union result only reached `segm/AP = 0.41991368748779717`, and the decision file explicitly kept Stage 3 paused. Sources: `docs/results/2026-03-30-rgb-phase23-instance-local-reset-summary.json`; `docs/results/2026-03-31-rgb-phase23-instance-local-stage2-summary.md`.

This legacy result family is important because it explains the later architectural shift. The repo did not abandon the fragment-first line arbitrarily; it documented that the learned stages were still far from the oracle and far from the active benchmark.

### 7.5 Query-Alpha Results

The repository contains detailed query-alpha plans, ladder documents, configs, and tests, but the stored full official result bundle is [Not Found]. What does exist is a clear protocol:

- compare only `v1.5 legacy`, `query_small_resnet18`, and `query_medium_resnet34` in alpha
- promote `query_ref_*`, `query_graph_*`, and `query_refgraph_*` only after the query-only base is proven
- use a fixed shared metric surface with count calibration and pathology summaries

Sources: `docs/archive/experiments/gisec-query-ladder.md`; `docs/archive/experiments/gisec-query-gates.md`; `docs/archive/experiments/gisec-query-metrics.md`; `docs/archive/plans/2026-03-23-gisec-query-master-plan.md`; `configs/query/*`; `tests/test_query_eval_cli.py`.

### 7.6 Overall Interpretation

Taken together, the results are internally consistent.

- Strong standard detectors won the early benchmark.
- Adding local refinement to the active RGB line produced the best official score in the repository.
- Later rescue stages were close but did not materially improve on that best score.
- The original graph-heavy line underperformed, even though its oracle analyses showed meaningful headroom.
- Query-alpha remained a structured experimental direction rather than a completed promoted mainline.

That is a practical research outcome, not a failure. The repository successfully identified which ideas were promising, which ones were not ready, and which ones needed to stay archived until they could clear explicit gates.

## 8. Conclusion and Future Work

The main takeaway is that the repository evolved from a custom prototype-conditioned graph segmentation program into a more pragmatic staged Mask2Former mainline, and the stored evidence supports that shift. The strongest completed official result currently present in the repository is `base_rgb_1024_refine` with `segm/AP = 0.5761366653940664`, `bbox/AP = 0.5155950306627068`, and `boundary/IoU = 0.25118819472440657`. The 2026-04-13 rerun is still underway and is excluded from the official results below. Sources: `docs/results/2026-04-12-active-rgb-official-ladder-summary.md`; `docs/results/README.md`; `output/experiments/2026-04-13-rgb-full-rerun/active_official/active_rgb_official/train/base_rgb_1024/run_state.json`; `output/experiments/2026-04-13-rgb-full-rerun/recovery_launcher/launcher.log`.

The current implementation has several clear limitations.

- The legacy graph line is implemented but empirically weaker than the promoted active line.
- The query-alpha branch is architecturally real but lacks stored full official result artifacts in the repository.
- Per-run hardware provenance is [Not Found].
- Some experiment families produce very large artifacts, which is why the repository also contains output-hygiene and pruning work.

Sources: `docs/results/2026-04-06-phase-a-baseline-reset-closeout.md`; `configs/runtime/*.yaml`; `docs/archive/plans/2026-03-27-output-hygiene-and-training-observability-design.md`; `scripts/maintenance/prune_output_artifacts.py`; `docs/results/2026-04-08-active-rgb-resume-and-throughput-recovery.md`.

The most concrete future-work directions are already written into the process documents.

- For the active line, continue the RGB-first staged path and treat reference or graph rescue as justified only if they beat the refine-only stage under the same evaluation surface.
- For the legacy instance-local line, Stage 3 should stay paused until learned owner-union becomes strong enough and remains clearly merge-limited.
- For query-alpha, only reopen `query_ref_*`, `query_graph_*`, and `query_refgraph_*` after `query_small_resnet18` and `query_medium_resnet34` clear the documented alpha gates.
- For the repository itself, keep improving artifact hygiene, monitoring, and reproducibility around long-running experiments.

Sources: `docs/results/2026-03-31-rgb-phase23-instance-local-stage2-summary.json`; `docs/archive/experiments/gisec-query-gates.md`; `docs/archive/plans/2026-03-23-05-gisec-query-reference-graph-reentry.md`; `docs/results/2026-04-08-active-rgb-resume-and-throughput-recovery.md`.

The most important unfinished items are therefore not hidden. They are explicitly documented in the repository: the query-only object-first branch still needs decisive promoted results, the legacy graph branch still needs a much smaller oracle-to-learned gap, and the active line still needs evidence before later rescue stages can be treated as upgrades rather than optional experiments.

---

## Document Revision Log

| Date | Section | Change | Source |
|------|---------|--------|--------|
| 2026-04-12 | Opening statement | Replaced raw experiment-artifact dependence with curated docs and source code evidence | `docs/results/README.md` |
| 2026-04-12 | Sections 1-2 | Removed out-of-scope privacy and deployment discussion and kept the report research-focused | `README.md`; `docs/results/2026-04-08-active-rgb-resume-and-throughput-recovery.md` |
| 2026-04-12 | Sections 5-8 | Replaced raw result citations with curated result notes, removed the depth-follow-up subsection, and added `[Not Found]` markers where needed | `docs/results/2026-04-12-active-rgb-official-ladder-summary.md`; `docs/results/2026-04-06-phase-a-baseline-reset-closeout.md`; `docs/archive/experiments/gisec-query-ladder.md` |
| 2026-04-12 | Sections 1-7 | Trimmed remaining depth-follow-up wording and kept the active results anchored to the RGB-first curated summaries | `docs/results/README.md`; `docs/results/2026-03-29-rgb-phase1-backbone-summary.md`; `docs/results/2026-04-12-active-rgb-official-ladder-summary.md` |
| 2026-04-13 | Sections 7-8 | Marked the 2026-04-13 active RGB rerun as still running, kept its result fields empty, and excluded it from the official completed results | `output/experiments/2026-04-13-rgb-full-rerun/active_official/active_rgb_official/train/base_rgb_1024/run_state.json`; `output/experiments/2026-04-13-rgb-full-rerun/recovery_launcher/launcher.log` |
