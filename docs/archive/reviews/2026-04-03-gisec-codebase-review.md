# GISEC Codebase Review

Review date: `2026-04-03`

## Bottom Line

The repo has real structure, a useful test surface, and a clearer active/legacy split than many research codebases. The main problem is that several core seams are now out of sync: the legacy graph path has at least one hard runtime break, the active path has a few silent-misconfiguration traps, and the shared runner/data-contract layer still has portability and documentation drift.

## High-Severity Findings

### 1. The legacy learned graph scorer cannot consume the graph batches that the current builder emits

Evidence:
- `gisec/models/graph_utils.py:22` defines `EDGE_FEATURE_DIM = 8`.
- `gisec/models/graph_utils.py:575-582` appends eight edge features.
- `gisec/models/gisec_model.py:45-48` still builds `GraphEdgeScorer(..., edge_dim=6, ...)`.
- `gisec/models/gisec_model.py:72-73` forwards `graph_batch.edge_features` directly into that scorer.

Direct reproduction:

```bash
python - <<'PY'
import torch, numpy as np
from gisec.models.gisec_model import GISECModel
from gisec.models.graph_utils import build_graph_batch
from gisec.config.variants import get_variant_spec
import gisec.models.graph_utils as graph_utils

feature_map = torch.ones((1, 8, 16, 16), dtype=torch.float32)
fg_logits = torch.full((1, 1, 16, 16), 4.0, dtype=torch.float32)
boundary_logits = torch.full((1, 1, 16, 16), -4.0, dtype=torch.float32)
boundary_logits[:, :, :, 7:9] = 4.0
ownership_offsets = torch.full((1, 2, 16, 16), 4.0, dtype=torch.float32)
depth_map = torch.ones((1, 1, 16, 16), dtype=torch.float32)

fragments = np.zeros((16, 16), dtype=np.int32)
fragments[4:12, 3:7] = 1
fragments[4:12, 8:12] = 2

orig = graph_utils.fragments_from_logits
graph_utils.fragments_from_logits = lambda *args, **kwargs: fragments.copy()
try:
    batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=ownership_offsets,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=None,
        variant=get_variant_spec("G1"),
        min_area=2,
    )
    print("edge_features_shape", tuple(batch.edge_features.shape))
    model = GISECModel(base_channels=8)
    model.forward_graph(batch)
finally:
    graph_utils.fragments_from_logits = orig
PY
```

Observed result:

```text
edge_features_shape (1, 8)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x136 and 134x64)
```

Why this matters:
- This is not just theoretical drift. The current legacy learned-graph path crashes at runtime once the model sees a real graph batch.
- The tests cover the pieces separately, but not the seam between `build_graph_batch(...)` and `GISECModel.forward_graph(...)`.

### 2. Constrained merge is not actually score-ordered, even though the method depends on score-ordered greedy merging

Evidence:
- `docs/method/gisec-method-fragment-first.md:63-69` defines `ConstrainedGreedyMerge` as descending-score greedy acceptance.
- `docs/method/gisec-method-fragment-first.md:116-118` repeats that inference attempts merges in descending score order.
- `gisec/models/graph_utils.py:832-851` iterates through `zip(edge_index.t().tolist(), edge_scores.tolist())` in input order and never sorts by score.

Why this matters:
- Once merge constraints depend on the current cluster state, edge order is part of the algorithm.
- Right now the result depends on the order in which edges were emitted upstream, not on the intended highest-confidence-first policy.
- `tests/test_graph_batch_and_merge.py:9-42` only exercises single-edge cases, so this regression is not caught.

### 3. The repo-hygiene tests are broken and not self-contained

Evidence:
- `tests/test_repo_hygiene_script.py:35` and `tests/test_repo_hygiene_script.py:64` invoke `/home/k100/zhn/electronic-components-grasp-and-segment/gnn-reference-prior/scripts/analysis/check_repo_hygiene.py`.
- This repo already contains `scripts/analysis/check_repo_hygiene.py`, but the test does not use it.

Validation:

```bash
pytest -q tests/test_prototype_bank_loader.py tests/test_config_io.py tests/test_project_metadata.py tests/test_repo_hygiene_script.py
```

Observed result:
- `25 passed, 2 failed`
- Both failures were in `tests/test_repo_hygiene_script.py`
- The external script path did not exist in this checkout, so the tests failed before they could validate behavior

Why this matters:
- A broken test at this level weakens confidence in repo hygiene exactly where the project is trying to enforce naming and layout discipline.
- It also makes the test suite depend on the contributor’s directory tree instead of the repo itself.

### 4. Active variant auto-detection only affects routing, not the actual active parser defaults

Evidence:
- `gisec/cli/_routing.py:28-59` can infer an active variant from `run_summary.json`.
- `gisec/cli/train.py:10-17` and `gisec/cli/eval.py:10-17` only use that result to decide active vs legacy routing.
- `gisec/train/train_active.py:123-164` still defaults `--variant` to `base_rgb_1024` unless the caller explicitly passes it.

Direct reproduction:

```bash
python - <<'PY'
import json, tempfile
from pathlib import Path
from gisec.cli._routing import resolve_cli_variant
from gisec.train.train_active import parse_eval_args

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "run"
    out.mkdir()
    (out / "run_summary.json").write_text(
        json.dumps({"variant": "base_rgbd_1024_refine_ref_graph"}),
        encoding="utf-8",
    )
    argv = [
        "--dataset-root", "/tmp/dataset",
        "--output-dir", str(out),
        "--checkpoint", str(out / "model_best.pth"),
    ]
    print("resolved_variant", resolve_cli_variant(argv))
    print("parsed_variant", parse_eval_args(argv).variant)
PY
```

Observed result:

```text
resolved_variant base_rgbd_1024_refine_ref_graph
parsed_variant base_rgb_1024
```

Why this matters:
- A user can point eval/infer at an existing active run directory, get correct routing to the active stack, and still silently parse the wrong active variant.
- That is especially risky because later-stage variants change depth mode, prototype-root requirements, and local-module behavior.

### 5. Active checkpoint loading is permissive and silent, so wrong checkpoints can still run

Evidence:
- Refine-stage init loading filters down to only compatible keys: `gisec/train/train_active.py:338-341`.
- Eval uses `model.load_state_dict(state_dict, strict=False)` at `gisec/train/train_active.py:1109-1111`.
- Infer does the same at `gisec/train/train_active.py:1176-1178`.

Why this matters:
- A mismatched checkpoint can degrade into partial loading or random initialization without a hard failure.
- That is the wrong failure mode for experiments: it produces outputs that look valid but may not correspond to the requested model.

## Medium-Severity Findings

### 6. The strict prototype-bank contract validates `views`, but the upstream spec says `num_views`

Evidence:
- The upstream data spec requires `meta/manifest.json` to include `num_views`: `../magformer/docs/plans/2026-03-10-reference-data-spec.md:170-182`.
- `gisec/datasets/prototype_bank.py:304-307` validates `meta.get("views")`.
- `tests/test_prototype_bank_loader.py:51`, `:75`, `:95`, and `:120` reinforce the same `views` field.

Why this matters:
- The strict validator is checking a field name that does not match the documented contract.
- The test suite currently preserves the mismatch instead of catching it.

### 7. “No-reference” legacy ablations are not actually no-reference by default

Evidence:
- `gisec/config/variants.py:118-146` marks `G1` and `G2` with `use_reference_conditioning=False`.
- `gisec/train/train_gisec.py:172-176` still defaults `--reference-conditioning-mode` to `full`.
- Training only disables conditioning if the caller explicitly sets `off`: `gisec/train/train_gisec.py:377-389`.
- The eval/export runtime always resolves and passes a prototype cache: `gisec/engine/runtime.py:853-860`.

Why this matters:
- These ablations are supposed to answer “what happens without reference conditioning?”
- Right now that answer depends on hidden CLI behavior and even differs between training and evaluation paths.

### 8. Active refine/reference/graph training is optimistic-only and weakly discriminative

Evidence:
- Inference refines predicted coarse masks: `gisec/train/train_active.py:621-638`.
- Training feeds a GT-derived blurred mask instead: `gisec/train/train_active.py:903-916`.
- The reference-match auxiliary loss uses all-ones targets: `gisec/train/train_active.py:925-927`.
- The graph-rescue auxiliary loss also uses all-ones targets: `gisec/train/train_active.py:729-734`.
- `tests/test_active_graph_training.py:9-26` only checks that the loss is positive, not that it separates good and bad edges.

Why this matters:
- The refiner never has to learn from the kinds of coarse-mask errors it will see at inference time.
- The auxiliary heads are rewarded for always saying “match” and “merge,” which weakens their ability to reject bad cases.

### 9. The core legacy scorer drops edge type before scoring, even though the repo treats edge type as a meaningful cue elsewhere

Evidence:
- `gisec/models/graph_utils.py:585-630` stores `edge_type` separately in `GraphBatch`.
- `gisec/models/gisec_model.py:72-73` only forwards `graph_batch.edge_features` into the scorer.
- `baseline/reference_graph/dataset.py:111-120` explicitly one-hot encodes `edge_type` into the scored feature vector on the baseline reference-graph path.

Why this matters:
- The method notes list edge type as a meaningful graph cue.
- The baseline reference-graph path already agrees with that idea.
- The main legacy GISEC scorer is the odd one out, so contact-vs-bridge semantics are lost at the scoring interface.

### 10. The query runner breaks the shared shell-runner contract and has unsafe quoting

Evidence:
- README says shell runners should honor `GISEC_CONDA_ENV` and `GISEC_PYTHON`: `README.md:110`.
- Shared runner support exists in `scripts/experiments/common_runner.sh:51-65`.
- `scripts/experiments/run_gisec_query_uq.sh:34` hard-codes `python -m ...`.
- `scripts/experiments/run_gisec_query_uq.sh:51` executes a single interpolated string via `eval`.

Direct reproductions:

```bash
GISEC_PYTHON=/tmp/custom-python \
bash scripts/experiments/run_gisec_query_uq.sh \
  --preset alpha-short-run \
  --model-scale s \
  --dataset-root /tmp/d \
  --output-root /tmp/o \
  --dry-run
```

Observed result:
- The printed command still began with `python -m`, not `/tmp/custom-python -m`

```bash
bash scripts/experiments/run_gisec_query_uq.sh \
  --preset alpha-short-run \
  --model-scale s \
  --dataset-root /tmp/d \
  --output-root "/tmp/o'ut" \
  --dry-run
```

Observed result:
- The printed `--output-dir` was malformed: `'/tmp/o'ut/UQ-s'`

Why this matters:
- This runner is the outlier relative to the repo’s shared runner contract.
- The current quoting is fragile enough to break on valid filesystem paths.

## Low-Severity Findings

### 11. Documentation and runner defaults still disagree about the active surface

Evidence:
- README frames the active face as the Mask2Former line and says Query Alpha is archival: `README.md:3-13`.
- `docs/method/README.md:17-19` still says `query-alpha object-first` is the active implementation target.
- README points to a missing method file at `README.md:48`.
- `scripts/experiments/run_gisec_active.sh:11` defaults `DATASET_ROOT` to `/home/k100/zhn/electronic-components-grasp-and-segment/datasets/0831_1K`, while README examples use `/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K`.
- README describes `run_gisec_active.sh` as the helper for the full active surface, but refine variants still require an explicit init checkpoint: `gisec/train/train_active.py:190-191`.

Why this matters:
- New contributors can follow the docs and still end up with the wrong mental model of what is active, what is archival, and what prerequisites later active stages actually have.

## Strengths

- The repo has stronger unit coverage than average for graph-builder behavior. `tests/test_graph_builder_legacy.py`, `tests/test_variant_spec.py`, and `tests/test_reference_graph_eval.py` cover bridge edges, purity filtering, threshold sweeps, and export contracts in a concrete way.
- The prototype-bank loader is explicit and inspectable. `gisec/datasets/prototype_bank.py` gives a real compat/strict contract, typed errors, and cached bank loading instead of silently guessing.
- The active/legacy split is real in code, not just in prose. `gisec/cli/train.py`, `gisec/cli/eval.py`, and the explicit `*_legacy.py` entrypoints make the repo easier to navigate than a single monolithic CLI would.

## Validation Performed

### Targeted pytest runs

```bash
pytest -q tests/test_active_train_cli.py tests/test_active_cli_routing.py tests/test_active_stage_order.py tests/test_active_graph_training.py tests/test_active_runner_dry_run.py
```

Observed result:
- `15 passed in 7.27s`

```bash
pytest -q tests/test_gisec_model_forward.py tests/test_graph_batch_and_merge.py tests/test_reference_graph_merge.py tests/test_reference_graph_eval.py tests/test_variant_spec.py
```

Observed result:
- `38 passed in 15.31s`

```bash
pytest -q tests/test_prototype_bank_loader.py tests/test_config_io.py tests/test_project_metadata.py tests/test_repo_hygiene_script.py
```

Observed result:
- `25 passed, 2 failed`
- Both failures were the hard-coded external-path issue in `tests/test_repo_hygiene_script.py`

### Direct reproductions

- Reproduced the legacy graph scorer dimension crash from the current graph-builder output.
- Reproduced active variant auto-detection returning `base_rgbd_1024_refine_ref_graph` while `parse_eval_args(...)` still defaulted to `base_rgb_1024`.
- Reproduced the refine-stage CLI error for missing `--init-checkpoint`.
- Reproduced `run_gisec_query_uq.sh` ignoring `GISEC_PYTHON`.
- Reproduced malformed dry-run output from `run_gisec_query_uq.sh` when the output path contained a single quote.

## Residual Risk

- I did not run the full test suite or any long training/evaluation jobs.
- I did not measure how often the medium-severity design issues hurt final metrics in practice.
- The review is strongest on interface seams, contract drift, and runtime correctness. It is lighter on model-quality claims that would require GPU runs.
