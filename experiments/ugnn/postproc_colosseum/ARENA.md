# Post-processing Colosseum (Round 2) — Judge Verdict (2026-08-21)

Independent rerun, gisec env, serial per team. Judge scripts:
`bench/correctness.py`, `bench/timing.py`, plus two judge-authored
probes (full-250 determinism/byte-equality; 5-image unseen-val
generalization with fresh E9 GPU forward). Team code untouched.

## 0. Arena state, integrity, cheat sweep

- Leftover processes: none. The two `/tmp/debug4.py` ~70 GB RSS
  orphans reported by team_b were team_a's own stuck heap
  simulations; team_a killed them at 15:26 (per its STATUS.md).
  Nothing to kill at judge time. GPU idle, 377 GB RAM free.
- `md5sum -c data/MANIFEST.md5`: **252/252 OK** — data package
  untouched.
- Source review (all three `solution.py` + both `precompute.py`):
  no lookup-table patterns, no reads of `reference_outputs.json`,
  no hardcoded 250-image outputs. Caches are input-only elevation,
  keyed (split, image_id) + depth md5 (b) / md5 (c) — C5 compliant.
- Rule note: **team_a's hot path runs the elevation sobel on GPU**.
  PROBLEM.md §6 allows GPU "only for dump generation, not for the
  contestant hot path". Recorded as a rule conflict; it does not
  change the ranking (a also loses on latency, see §2).

## 1. Correctness rerun (one-vote veto)

bench/correctness.py, judge-run:

| team | C1 zero-dev / max | C2 mean IoU | C3 \|ΔAP\| | C4 | verdict |
|------|-------------------|-------------|------------|----|---------|
| team_a (self) | 0.9840 / 1 | 0.998384 | 0.00012 | PASS | — |
| **team_a (judge)** | 0.9840 / 1 | 0.998384 | 0.00012 | PASS | PASS |
| team_b (self) | 0.9840 / 1 | 0.998380 | 0.00012 | PASS | — |
| **team_b (judge)** | 0.9840 / 1 | 0.998380 | 0.00012 | PASS | PASS |
| team_c (self) | 1.0000 / 0 | 1.000000 | 0.00000 | PASS | — |
| **team_c (judge)** | 1.0000 / 0 | 1.000000 | 0.00000 | PASS | PASS |

Judge extras (250 images, not just the 50-image probe):

- Full-250 run-twice determinism: a/b/c all bitwise stable.
- Full-250 byte-equality vs reference outputs: **team_c identical
  on every image**; a/b differ (their documented watershed pop-order
  deviations: 4/250 images with instance count ±1).

### Unseen-image generalization (anti-lookup-table probe, new this round)

5 val images *not* in the 250-dump package, fresh E9 forward
(`exp09_centernet_seeds/runs/best.pth`), hm/off round-tripped
through f16 to match the dump dtype convention, fed to
`reference/postproc_ref.py` and to each solution **on its cache-miss
path** (b/c caches only cover the 250):

| img (n_ref) | team_a | team_b | team_c |
|------|------|------|------|
| 1 (72) | n=72 IoU .9985 | n=72 IoU .9985 | n=72 **byte-eq** |
| 2 (68) | n=68 IoU .9983 | n=68 IoU .9983 | n=68 **byte-eq** |
| 4 (73) | n=73 IoU .9989 | n=73 IoU .9989 | n=73 **byte-eq** |
| 5 (72) | n=72 IoU .9971 | n=72 IoU .9971 | n=72 **byte-eq** |
| 6 (64) | n=64 IoU .9998 | n=64 IoU .9998 | n=64 **byte-eq** |

No cache-miss failure, no silent wrong output. All three generalize.

## 2. Official timing (serial, systemd-run --user -p MemoryMax=32G -p CPUQuota=800%, units c2-judge-a2/b2/c2)

| team | self median | **judge median** | judge p90 | speedup vs baseline 673.75 |
|------|-------------|------------------|-----------|-----------------------------|
| team_a (GPU sobel) | 96.1 | **95.88 ms/img** | 115.27 | 7.0x |
| team_b (numba CPU) | 66.4 | **69.07 ms/img** | 78.48 | **9.8x** |
| team_c (algo reduction) | 130.5 | **132.12 ms/img** | 163.40 | 5.1x |

Throughput (secondary, judge rerun of team_b/throughput.py, 8 procs,
same systemd limits, unit c2-judge-btp): **73.44 imgs/s (13.6
ms/img wall)**, 250/250 images. team_a/c made no throughput claim.

## 3. Verdict

**Champion: team_b (numba CPU route).**

1. All judge gates pass, including the unseen-image cache-miss
   probe — and it is the fastest single-process hot path by a wide
   margin (69.07 vs 95.88 vs 132.12 ms/img; 9.8x over the reference
   on the same package).
2. Production fit: pure CPU, drops straight into the existing
   "GPU forward main process + CPU Pool" two-stage eval. team_a's
   GPU sobel serializes against the forward pass on the same card
   (and conflicts with PROBLEM §6's CPU-only hot-path rule);
   team_c is CPU but 1.9x slower than b.
3. Accuracy margin is real but immaterial: b's only documented
   deviation is the watershed marker-plateau tie order (4/250
   images ±1 instance, |ΔAP|=0.00012); its 8-proc throughput
   (73 imgs/s) beats the production Pool(6) 470 ms/img wall by ~3x
   headroom.

Precision tiebreaker (§4 of the criteria) never activates: c is
byte-exact but not at b's latency level.

### 3.1 Rule revision review (2026-08-21, user directive) — no rerun

After the verdict, the user LIFTED PROBLEM §6's GPU hot-path ban
(GPU now allowed, CPU no longer mandated). Verdict review under the
revised rules: **team_b remains champion.** team_a was in fact
measured on its actual GPU path (95.88 ms/img judge-run), so it
lost to team_b's 69.07 ms on pure numbers, not on the rule — the
rule conflict only made the margin worse. The production argument
also stands regardless of the rule: GPU post-processing serializes
against the 8 ms GPU forward on the same card, so the GPU route
*reduces* end-to-end throughput compared to the CPU pool. No rerun.

## 4. Combination suggestion (optional, post-arena)

b and c are complementary: c is byte-exact but slow (skimage
watershed 86.5 ms kept), b is fast but has the tie-order deviation.
Swapping b's merge+extract stages for c's vectorized-merge +
bincount/bbox-RLE stages (both already byte-exact in c) would
plausibly give b's watershed speed *and* c's byte-equality —
candidate for the integration task, to be re-gated before use.

## 5. Integration notes

- team_b cache: elevation-rank keyed (val, image_id)+md5(depth);
  cold-miss first image in a fresh process costs ~0.5 s (numba JIT
  + inline elevation), steady-state miss ≈ +0.2 s/img until the
  rank cache is built. Precompute (`team_b/precompute.py`) before
  any full-val run, and extend the cache to the full val split.
- team_a: GPU post-processing competes with the E9 forward pass on
  the same card in production; also a PROBLEM §6 rule conflict.
  Do not integrate as-is.
- team_c: exactness is the gold standard for regression testing;
  keep it as the reference-equivalence oracle even if not the
  production path.
- All judge units (c2-judge-a2/b2/c2/btp) finished and were
  reset; no stray processes left.
