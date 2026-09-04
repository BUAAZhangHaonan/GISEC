import subprocess
import sys

PY = sys.executable
for team, track in [("team_trt", "fwd"), ("team_ws", "ws"), ("team_fuse", "fwd")]:
    r = subprocess.run(
        [PY, "harness.py", track, f"../{team}/solution.py"],
        capture_output=True,
        text=True,
        timeout=1500,
    )
    line = [ln for ln in r.stdout.splitlines() if "RESULT" in ln]
    print(
        f"[judge] {team}: {line[-1] if line else 'NO RESULT: ' + r.stderr[-300:]}",
        flush=True,
    )
