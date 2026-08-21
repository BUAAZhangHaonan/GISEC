# Team C STATUS

## systemd units used (all under --user, MemoryMax=32G, CPUQuota=800%)

- c2-team-c-pre.service .. c2-team-c-pre7.service — elevation precompute
  (final: c2-team-c-pre7, 250 maps, 24 s)
- c2-team-c-corr.service — aborted early run of correctness (stopped, was
  exercising a since-fixed RLE bug)
- c2-team-c-corr2.service — final correctness gate, C1-C4 all PASS
- c2-team-c-mergeval3/4/5.service — merge-equivalence validation, 250 imgs,
  0 mismatches (final: mergeval5)
- c2-team-c-timing / c2-team-c-timref / c2-team-c-tim2 — timing runs
- c2-team-c-ws512 — 512-res watershed experiment (rejected)

All units finished; none left running (verified with systemctl --user list-units).
Parallel processes at any time: 1. No GPU used. Scratch experiment scripts
(pm.py, ws512.py, validate_merge.py) removed after use.

## Self-timed numbers

- reference.postproc_ref: median 687.3 ms/img (n=20) / judge 673.75 (n=50)
- team_c.solution: median 130.5 ms/img (n=50), p90 158.9
- precompute amortized: 104 ms/img, one-time, offline
