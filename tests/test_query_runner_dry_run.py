from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


def test_query_runner_uses_gisec_python_and_handles_quoted_paths(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_query_uq.sh"
    wrapper_log = tmp_path / "wrapper.log"
    wrapper = tmp_path / "fake_python.sh"
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf '%s\\n' \"$@\" > \"$WRAPPER_LOG\"",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    dataset_root = tmp_path / "data's"
    output_root = tmp_path / "out's"

    env = os.environ.copy()
    env["GISEC_PYTHON"] = str(wrapper)
    env["WRAPPER_LOG"] = str(wrapper_log)

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--run",
        ],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert wrapper_log.exists()
    wrapper_args = wrapper_log.read_text(encoding="utf-8")
    assert "-m" in wrapper_args
    assert "gisec.cli.train_query" in wrapper_args
    assert str(dataset_root) in wrapper_args
    assert str(output_root / "query_small_resnet18") in wrapper_args
