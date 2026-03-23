# GISEC v3 Reference Rescue Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `reference` back into the new `GISEC` design as a required future paper module, but restrict its first executable role to a single rescue-side enhancement path.

**Architecture:** `reference` is not removed from `GISEC`. It is postponed and narrowed. The alpha base must work without it, then the first `reference` integration is allowed only inside the rescue stage for uncertain objects. It does not enter the coarse object backbone, and it does not become a hidden dependency of the main query-only segmentation model.

**Tech Stack:** Existing per-part prototype-bank contract, `gisec_v3` model/runtime surface, route-and-mix reference cache utilities, current ECC reference dataset protocol.

---

### Task 1: Define the alpha reference role in code and docs

**Files:**
- Create: `gisec_v3/config/reference_spec.py`
- Modify: `docs/plans/2026-03-23-gisec-v3-alpha-master-plan.md`
- Test: `tests/test_v3_reference_spec.py`

**Step 1: Write the failing test**

Add tests that require the alpha reference role to be stated explicitly:
- `UQ-*` must not require reference
- `UR-*` and `UA-*` are reserved names
- alpha reference integration is rescue-only
- coarse object backbone must remain query-only

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_reference_spec.py -v
```

Expected: FAIL because the new reference spec is missing.

**Step 3: Write minimal implementation**

Implement a minimal spec or registry that encodes:
- `use_reference = false` for `UQ`
- `reference_entry = rescue_only` for the first `UR/UA` phase
- no support for backbone modulation in alpha

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_reference_spec.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/config/reference_spec.py docs/plans/2026-03-23-gisec-v3-alpha-master-plan.md tests/test_v3_reference_spec.py
git commit -m "docs: define v3 alpha reference role"
```

### Task 2: Define the rescue-side reference interface

**Files:**
- Create: `gisec_v3/models/reference_rescue.py`
- Create: `gisec_v3/models/reference_contract.py`
- Test: `tests/test_v3_reference_contract.py`

**Step 1: Write the failing test**

Add tests that require a rescue-side reference interface to accept:
- local object or piece features
- `part_key`
- routed reference pack

and return:
- `reference_context`
- `shape_quantiles`
- routing metadata

The test should explicitly reject an interface that conditions the main coarse object backbone.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_reference_contract.py -v
```

Expected: FAIL because the reference rescue contract does not exist.

**Step 3: Write minimal implementation**

Create the new contract and module surface only. Do not implement backbone modulation. Keep the first interface narrow and local.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_reference_contract.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/models/reference_rescue.py gisec_v3/models/reference_contract.py tests/test_v3_reference_contract.py
git commit -m "feat: define v3 rescue-side reference interface"
```

### Task 3: Lock the minimum reference-pack protocol

**Files:**
- Create: `docs/method/gisec-v3-alpha-reference-pack.md`
- Modify: `configs/v3/README.md`
- Test: `tests/test_v3_reference_pack_rules.py`

**Step 1: Write the failing test**

Add tests that require the alpha reference protocol to document:
- required assets: `RGB + depth + mask`
- per-part routing by `part_key`
- default `max_views`
- default `pose_farthest`
- default slot count and routing mode

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_reference_pack_rules.py -v
```

Expected: FAIL because the v3 reference-pack rules are not documented.

**Step 3: Write minimal implementation**

Document the initial defaults:
- `max_views = 16`
- `view_sampler = pose_farthest`
- `slot_count = 6`
- `routing = top-2 soft`

These are design defaults only; they do not yet make reference mandatory for the base model.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_reference_pack_rules.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/method/gisec-v3-alpha-reference-pack.md configs/v3/README.md tests/test_v3_reference_pack_rules.py
git commit -m "docs: define v3 reference pack rules"
```

### Task 4: Add the first rescue-only routing path

**Files:**
- Create: `gisec_v3/engine/reference_runtime.py`
- Test: `tests/test_v3_reference_runtime.py`

**Step 1: Write the failing test**

Add tests that require:
- `UQ` runtime to skip reference entirely,
- `UR` or `UA` reserved runtime to resolve reference only when rescue objects are passed in,
- routing metadata to be exported without mutating the coarse object backbone output.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_reference_runtime.py -v
```

Expected: FAIL because the rescue-side reference runtime path does not exist.

**Step 3: Write minimal implementation**

Implement a thin rescue-only runtime path:
- no whole-image conditioning,
- no modification of the coarse object proposal stage,
- explicit skip when no rescue object is present.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_reference_runtime.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/engine/reference_runtime.py tests/test_v3_reference_runtime.py
git commit -m "feat: add v3 rescue-only reference runtime"
```

### Task 5: Define the promotion rule for reference

**Files:**
- Create: `docs/results/README-v3-reference-gate.md`
- Test: `tests/test_v3_reference_gate_doc.py`

**Step 1: Write the failing test**

Add a test that requires the promotion rule to be written explicitly:
- `reference` only graduates from reserved module to formal enhancement if it improves appearance-ambiguous cases without becoming a coarse-backbone dependency.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_reference_gate_doc.py -v
```

Expected: FAIL because the promotion rule is undocumented.

**Step 3: Write minimal implementation**

Document the rule in plain language, including what counts as evidence:
- hard subset gain
- no regression on normal objects
- no hidden shift of the query-only mainline into a reference-required model

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_reference_gate_doc.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/results/README-v3-reference-gate.md tests/test_v3_reference_gate_doc.py
git commit -m "docs: add v3 reference promotion rule"
```
