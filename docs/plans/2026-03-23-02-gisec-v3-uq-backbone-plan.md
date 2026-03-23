# GISEC v3 UQ Backbone Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the minimal `UQ-s` and `UQ-m` query-only object-first backbones so the new mainline can be evaluated without reference or graph rescue.

**Architecture:** This phase intentionally fixes almost every architectural degree of freedom. Both models use the same `ResNet` encoder family, the same six-channel early fusion strategy, the same decoder family, and the same output heads. The only structural difference between `UQ-s` and `UQ-m` is encoder depth plus matching decoder width. This keeps the first scale study interpretable.

**Tech Stack:** PyTorch 2.10, torchvision ResNet backbones, existing COCO training/eval/export stack, new `gisec_v3` package surface.

---

### Task 1: Define the `UQModelSpec` contract

**Files:**
- Create: `gisec_v3/config/model_spec.py`
- Test: `tests/test_v3_model_spec.py`

**Step 1: Write the failing test**

Add tests that require:
- `UQ-s` and `UQ-m` specs to exist,
- both specs to share the same encoder family and fusion mode,
- the only intentional differences to be scale-related fields.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_model_spec.py -v
```

Expected: FAIL because the new model spec does not exist.

**Step 3: Write minimal implementation**

Define a dataclass or registry object that fixes:
- `encoder_family = resnet`
- `fusion_mode = rgb_depth_geometry_early`
- `heads = fg, boundary, core, ownership_offsets`
- `UQ-s = resnet18`
- `UQ-m = resnet34`

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_model_spec.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/config/model_spec.py tests/test_v3_model_spec.py
git commit -m "feat: define uq model spec contract"
```

### Task 2: Implement fixed depth-geometry input generation

**Files:**
- Create: `gisec_v3/models/depth_geometry.py`
- Test: `tests/test_v3_depth_geometry.py`

**Step 1: Write the failing test**

Add tests that require the new depth helper to produce exactly:
- normalized depth
- depth gradient magnitude
- depth discontinuity

The test should also assert the stacked output has `3` channels and does not depend on any reference input.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_depth_geometry.py -v
```

Expected: FAIL because the helper does not exist.

**Step 3: Write minimal implementation**

Implement a pure query-side depth preprocessing helper. Do not introduce a second depth encoder. Do not add optional fusion modes.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_depth_geometry.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/models/depth_geometry.py tests/test_v3_depth_geometry.py
git commit -m "feat: add v3 depth geometry input"
```

### Task 3: Implement the fixed `UQ` backbone family

**Files:**
- Create: `gisec_v3/models/uq_backbone.py`
- Create: `gisec_v3/models/common.py`
- Test: `tests/test_v3_uq_backbone.py`

**Step 1: Write the failing test**

Add tests that require:
- `UQ-s` and `UQ-m` to instantiate,
- both models to accept `RGB + depth_geometry` early fusion input,
- both models to produce a shared decoder feature map plus the four fixed heads,
- the two models to differ in scale but not in head schema.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_uq_backbone.py -v
```

Expected: FAIL because the new backbone family does not exist.

**Step 3: Write minimal implementation**

Implement:
- a single shared U-Net-style decoder family,
- `ResNet18` encoder for `UQ-s`,
- `ResNet34` encoder for `UQ-m`,
- one shared feature-map contract:
  - `fg_logits`
  - `boundary_logits`
  - `core_heatmap`
  - `ownership_offsets`
  - `feature_map`

Do not add:
- `ownership_confidence`
- `uncertainty`
- reference conditioning
- graph interfaces

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_uq_backbone.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/models/uq_backbone.py gisec_v3/models/common.py tests/test_v3_uq_backbone.py
git commit -m "feat: add uq s and uq m backbones"
```

### Task 4: Add the `gisec_v3` train/eval model factory

**Files:**
- Create: `gisec_v3/models/model.py`
- Create: `gisec_v3/engine/factory.py`
- Test: `tests/test_v3_factory.py`

**Step 1: Write the failing test**

Add tests that require:
- `UQ-s` and `UQ-m` to be buildable from the new registry,
- `use_reference` and `use_graph_rescue` to be false in `alpha`,
- the returned model object not to import or depend on old fragment-first `GraphRefiner`.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_factory.py -v
```

Expected: FAIL because the new factory does not exist.

**Step 3: Write minimal implementation**

Implement a `build_v3_model(...)` path that only supports:
- `UQ-s`
- `UQ-m`

and rejects unsupported later families with a clear error.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_factory.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/models/model.py gisec_v3/engine/factory.py tests/test_v3_factory.py
git commit -m "feat: add gisec v3 model factory"
```

### Task 5: Lock the alpha exclusions in writing

**Files:**
- Modify: `docs/plans/2026-03-23-gisec-v3-alpha-master-plan.md`
- Test: `tests/test_v3_boundary_contract.py`

**Step 1: Write the failing test**

Extend the boundary-contract test so it asserts `alpha` excludes:
- dual encoders
- stage-wise fusion
- encoder-family search
- uncertainty head
- ownership-confidence head

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_boundary_contract.py -v
```

Expected: FAIL until the exclusions are written clearly enough.

**Step 3: Write minimal implementation**

Add explicit exclusions to the master plan so later implementation does not quietly expand scope.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_boundary_contract.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/plans/2026-03-23-gisec-v3-alpha-master-plan.md tests/test_v3_boundary_contract.py
git commit -m "docs: lock uq alpha exclusions"
```
