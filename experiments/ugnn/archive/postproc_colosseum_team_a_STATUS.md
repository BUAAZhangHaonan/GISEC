# Team A status

systemd units used (all --user -p MemoryMax=32G -p CPUQuota=800%, GPU 0 only):
- c2-team-a, c2-team-a2..a7  (prototypes / correctness attempts) - stopped
- c2-team-a8                 (unused) - stopped
- c2-team-a9, c2-team-a13    (bench/timing.py) - stopped after finish
- c2-team-a10                (correctness, hit stuck pre-fix RLE) - stopped
- c2-team-a11                (correctness, json-bytes error) - finished w/ error
- c2-team-a12                (correctness PASS: C1-C4) - finished

Final numbers:
- bench/timing.py (c2-team-a13): median 96.13 ms/img, p90 113.24
- bench/correctness.py (c2-team-a12): C1-C4 all PASS
  (C1 0.9840/max 1, C2 0.998384, C3 |dAP| 0.00012, C4 PASS)

Orphan check: two of my own stuck /tmp/debug4.py loops (160 GB RSS each,
from an early buggy heap simulation) were found and killed at 15:26;
no other orphans. All units above stopped; confirmed below.
