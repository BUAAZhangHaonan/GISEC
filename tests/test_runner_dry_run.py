from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_gisec_runner_dry_run_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_gisec.sh"
    ref_root = tmp_path / "prototype_bank"
    for name in ["rgb", "depth", "mask", "meta"]:
        (ref_root / name).mkdir(parents=True)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(ref_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "G5",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode=dry-run" in res.stdout
    assert "variant=G5" in res.stdout
    assert "prototype_root=" in res.stdout
    assert "python -m gisec.cli.train" in res.stdout
    assert "conda run -n magformer" not in res.stdout


def test_gisec_runner_dry_run_supports_torchrun_launcher(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_gisec.sh"
    ref_root = tmp_path / "prototype_bank"
    for name in ["rgb", "depth", "mask", "meta"]:
        (ref_root / name).mkdir(parents=True)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(ref_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "G5",
            "--launcher",
            "torchrun",
            "--nproc-per-node",
            "6",
            "--master-port",
            "29610",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "torchrun --standalone --nnodes 1 --nproc-per-node 6 --master-port 29610 -m gisec.cli.train" in res.stdout


def test_gisec_runner_dry_run_supports_torchrun_launch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_gisec.sh"
    ref_root = tmp_path / "prototype_bank"
    for name in ["rgb", "depth", "mask", "meta"]:
        (ref_root / name).mkdir(parents=True)

    env = dict(os.environ)
    env["GISEC_TORCHRUN_NPROC_PER_NODE"] = "4"
    env["GISEC_TORCHRUN_MASTER_PORT"] = "29655"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(ref_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "G5",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "torchrun --standalone --nnodes 1 --nproc-per-node 4 --master-port 29655 -m gisec.cli.train" in res.stdout
    assert "--launcher 'torchrun'" in res.stdout or "--launcher torchrun" in res.stdout
