# Rank colosseum archive (2026-09-04)

Three-team arena for the inference bottleneck (the value->rank sort
segment: sem_logit_rank 212 / mix_elevation_rank 163 / cold depth rank
189 ms per 1024^2 image on the CPU reference). Judge-verified results
(back-to-back bench, 20 imgs, median-of-3, ms/img):

| team | route | check (bitwise vs reference) | sem | mix | cold |
|---|---|---|---:|---:|---:|
| reference | argsort / unique+searchsorted | — | 212.9 | 163.5 | 189.0 |
| team_a | pure numpy (f64-mantissa packed value sort + bincount mix) | PASS | 65.6 | 36.0 | 66.1 |
| team_b | numba 11-bit LSD radix + counting rank | PASS | 49.2 (serial) | 27.9 | 47.4 |
| team_c | GPU torch.sort (sobel stays CPU) | PASS | 16.7 | 3.2 | 17.0 |

Verdict: **team_b serial kernels merged** into `gisec.postproc_fast`
(bitwise-equal, 300-img end-to-end CRC gate). team_b's PARALLEL
kernels were judged unusable under the eval chain's fork pool —
`fork_test.py` reproduces the libgomp `fork() ... unsafe` abort — and
stay archived here. team_c's GPU rank segment became the rank core of
`gisec.gpu_pipeline` (the gpu_fast path). team_a's numpy mix-bincount
idea is independently present in team_b's counting rank.

Contents: `arena/` (rules, harness, fork probe, full-val gate script,
pre-merge reference md5 snapshot), `team_{a,b,c}/` (solutions + notes;
team_c also carries its fork test and timing breakdown). Payloads
(40 cached sem_logit arrays) were ephemeral and are not archived.
