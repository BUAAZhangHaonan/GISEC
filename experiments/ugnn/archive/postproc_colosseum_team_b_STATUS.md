# Team B STATUS

## systemd units used (all --user, MemoryMax=32G, CPUQuota=800%)
- c2-team-b-pre1, c2-team-b-pre2 (cache precompute)
- c2-team-b-corr1..corr8 (correctness gate runs)
- c2-team-b-t1..t5 (timing runs)
- c2-team-b-prof1 (profiling)
- c2-team-b-tp1, tp2 (throughput)
All transient (--wait --pipe), finished; stopped/cleaned (see below).

## Self-timed (advisory)
- bench/timing.py --module team_b.solution --n 50: median 66.4 ms/img
- bench/correctness.py --module team_b.solution: C1-C4 all PASS
- throughput (8 procs): 69.3 imgs/s

## Orphans check
`ps aux | grep python` before start: no stray user python processes.

## Cache (C5)
team_b/cache/val/<image_id>.rank.npy + .rank.md5 + .rank.nrank.npy,
keyed (split='val', image_id), validated by md5(depth). Input-only
elevation ranks; works on any val image (miss -> inline compute).

## Cleanup (post-run)
- All c2-team-b-* transient units stopped + reset-failed; none active.
- Orphans observed (NOT killed, not ours): team_c correctness unit
  c2-team-c-corr (running, judge/other team); two /tmp/debug4.py
  processes at ~70 GB RSS each (owner unknown, likely team_c/A debug).
