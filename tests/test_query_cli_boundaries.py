from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_query_train_cli_rejects_future_family_even_in_dry_run() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.train_query",
            "--model-family",
            "UR",
            "--model-scale",
            "s",
            "--dataset-root",
            "/tmp/dataset",
            "--output-dir",
            "/tmp/out",
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "not enabled in current query-alpha execution surface" in result.stderr
