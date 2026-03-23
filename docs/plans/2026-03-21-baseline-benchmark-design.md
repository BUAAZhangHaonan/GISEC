# Baseline Benchmark Design

## Goal
Build a credible benchmark ladder that answers three questions before further expanding `GISEC`:

1. how strong common pure-RGB instance segmentation models already are on this dataset,
2. how much depth is really worth when the task is cluttered electronic-component instance segmentation,
3. whether `GISEC` is currently losing because the method idea is wrong or because the current implementation is simply underperforming basic baselines.

## Why This Matters
The current `GISEC` line is underperforming badly enough that it is no longer safe to evaluate it in isolation. Right now there is no trustworthy answer to:

- whether the task is intrinsically hard for all methods,
- whether a standard RGB baseline already outperforms the current graph/reference design,
- whether RGB-D should provide a large gain here,
- whether the current failure is mostly in mask formation, grouping, or reference usage.

Without a strong benchmark stack, `GISEC` risks becoming a moving target with no stable comparison anchor.

## Design Principles
- Baselines must be first-class citizens, not one-off scripts.
- All baselines must export the same run artifacts as `GISEC` where practical.
- The first benchmark layer should establish pure-RGB lower and upper bounds.
- Depth should be introduced only after a clean RGB baseline table exists.
- Baseline engineering must stay modular so later results can be quoted directly in a paper table.

## Baseline Ladder
The benchmark stack is intentionally split into layers.

### Layer 1: Standard RGB Instance Segmentation Baselines
These are the "everyone recognizes them" comparison anchors.

- `Mask R-CNN`
- `Mask2Former`
- `YOLOv8-seg`

Purpose:
- establish strong public RGB baselines,
- measure standard AP / speed / memory tradeoffs,
- check whether `GISEC` is already losing to ordinary object-instance segmenters.

### Layer 2: Strong Pixel Segmentation Baselines
These are the simplest models that often work surprisingly well in industrial settings.

- `U-Net`
- `UNet++`
- `Attention U-Net`

Purpose:
- test whether dense pixel segmentation alone already solves much of the problem,
- measure whether the current `GISEC` bottleneck is really grouping or still basic mask quality.

### Layer 3: Similar-Scenario U-Net Variants
These are not meant to be an exhaustive literature survey. They are meant to represent "reasonable U-Net-family models for cluttered industrial imagery".

Candidate pool:
- `ResUNet`
- `U2Net` or another stronger edge-aware U-Net-like variant if engineering cost stays low
- one lightweight industrial-style RGB-D U-Net adaptation

Purpose:
- see whether the current task prefers strong local mask formation over explicit graph reasoning,
- create a fair comparison against the current `GISEC` backbone family.

## RGB Before RGB-D
The benchmark order must be:

1. pure RGB baselines first,
2. RGB-D extensions second,
3. `GISEC` re-evaluation third.

Reason:
- if RGB-only baselines already outperform `GISEC`, adding depth to `GISEC` first hides the real problem,
- if RGB-D baselines do not beat RGB baselines by a clear margin, then the current depth formulation may not deserve complexity credit,
- if RGB-D gives a large gain on simpler models but not on `GISEC`, that isolates the failure to the `GISEC` design.

## Expected Depth Value
For this project, depth is not a decorative modality. It should help separate touching objects, handle occlusion ordering, and suppress false visual continuity.

The working benchmark expectation is deliberately strict:

- RGB-D variants should materially outperform their RGB-only parents,
- if depth adds less than a substantial gain, the depth design or usage is probably weak,
- if depth cannot beat RGB by a large margin on this task, the current claim that RGB-D is central becomes questionable.

The benchmark goal is therefore not merely "does depth help a little". The goal is "does depth help enough to justify method complexity".

## Repository Shape
The repository should gain a top-level `baseline/` package or folder with clear submodules:

- `baseline/common/`
  - shared config parsing
  - shared output/export adapters
  - shared dataset adapters
  - shared metrics and logging helpers
- `baseline/mask_rcnn/`
- `baseline/mask2former/`
- `baseline/yolo_seg/`
- `baseline/unet/`
- `baseline/unetpp/`
- `baseline/attention_unet/`
- `baseline/rgbd/`
  - shared RGB-D fusion adapters for U-Net-family baselines

This keeps baseline code physically separate from `gisec/`, which is important because baseline comparison code should not pollute the main method implementation.

## Unified Benchmark Contract
Every baseline should converge on the same reporting contract as far as reasonable:

- `run_summary.json`
- `metrics.cocoeval.json`
- `inference_speed.json`
- `params_trainable.txt`
- `wall_time_sec.txt`
- `peak_memory_mb.txt`
- `visualizations/overlay/`

If a framework has its own logging style, a thin adapter should translate its outputs into the common experiment contract instead of forcing the whole baseline through `GISEC` internals.

## Training and Evaluation Protocol
To keep the table honest, all baselines should share:

- the same train/val split,
- the same image size for direct comparisons unless the model fundamentally requires a different recipe,
- the same AP evaluation contract,
- the same smoke/full distinction,
- a clearly recorded config stack.

The recommended protocol order is:

1. smoke protocol for each baseline to prove the wiring works,
2. short-run protocol for quick sanity comparisons,
3. full protocol only after the model has passed smoke and short-run gates.

## RGB-D Extension Strategy
RGB-D should not be added to every public baseline immediately.

Recommended order:

1. first add RGB-D to `U-Net`,
2. then extend `UNet++` or `Attention U-Net`,
3. only then decide whether it is worth adapting a heavier baseline.

RGB-D variants should be split at least into:

- `RGB-only`
- `RGBD-early-fusion`
- `RGB + depth-geometry-channels`

This is enough to tell whether the depth gain comes from raw fusion or from more structured geometry cues.

## Benchmark Table Structure
The target comparison table should eventually have columns like:

- model
- modality
- AP
- AP50
- AP75
- APs / APm / APl
- FPS
- peak memory
- trainable params
- notes

Rows should be grouped into:

- standard RGB instance segmentation
- RGB U-Net family
- RGB-D U-Net family
- `GISEC v1`
- `GISEC Method current`

## Stop Rules
Benchmarking should change the research direction if the evidence demands it.

- If plain RGB `Mask R-CNN` or `Mask2Former` already beat current `GISEC`, then `GISEC` cannot continue as if the current line were competitive.
- If `U-Net` family baselines already match or beat current `GISEC`, then the immediate priority becomes mask-quality recovery, not graph complexity.
- If RGB-D gives no strong gain on simpler baselines, then depth handling needs rethinking before any stronger claim is made.
- If RGB-D strongly helps simpler baselines but not `GISEC`, then the current `GISEC` integration of depth/reference/graph is the likely failure point.

## Recommended First Implementation Order
1. Create the `baseline/` scaffold and shared experiment contract.
2. Land a minimal `U-Net` RGB baseline first because it is the fastest reliable anchor.
3. Land `YOLOv8-seg` next as a fast instance baseline.
4. Land `Mask R-CNN` next as the classic region-based baseline.
5. Land `Mask2Former` after the shared contract is stable.
6. Expand the U-Net family.
7. Add RGB-D variants.
8. Rebuild the main benchmark table and compare back to `GISEC`.

## Acceptance Criteria
- The repository has a stable `baseline/` area separated from `gisec/`.
- At least one baseline from each family can run through the shared experiment contract.
- There is a reproducible benchmark table comparing pure RGB and RGB-D baselines.
- There is enough evidence to say whether `GISEC` is failing against strong baselines or against weak ones.
