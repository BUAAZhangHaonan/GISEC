from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_gisec_runner_dry_run_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec.sh"
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
    assert "python -m gisec.cli.train_legacy" in res.stdout
    assert "conda run -n magformer" not in res.stdout


def test_gisec_runner_dry_run_accepts_a0_variant(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec.sh"
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
            "A0",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "variant=A0" in res.stdout


def test_gisec_runner_dry_run_forwards_config_argument(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec.sh"
    ref_root = tmp_path / "prototype_bank"
    config_path = tmp_path / "smoke.yaml"
    for name in ["rgb", "depth", "mask", "meta"]:
        (ref_root / name).mkdir(parents=True)
    config_path.write_text("common: {}\n", encoding="utf-8")

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
            "--config",
            str(config_path),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"--config '{config_path}'" in res.stdout or f"--config {config_path}" in res.stdout


def test_gisec_runner_dry_run_supports_torchrun_launcher(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec.sh"
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

    assert "torchrun --standalone --nnodes 1 --nproc-per-node 6 --master-port 29610 -m gisec.cli.train_legacy" in res.stdout


def test_gisec_runner_dry_run_supports_torchrun_launch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec.sh"
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

    assert "torchrun --standalone --nnodes 1 --nproc-per-node 4 --master-port 29655 -m gisec.cli.train_legacy" in res.stdout
    assert "--launcher 'torchrun'" in res.stdout or "--launcher torchrun" in res.stdout


def test_gisec_v2_smoke_runner_lists_a0_and_a1(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_legacy_smoke.sh"
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
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "variant=A0" in res.stdout
    assert "variant=A1" in res.stdout


def test_gisec_v2_smoke_runner_uses_default_config_stack(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_legacy_smoke.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "configs/data/ecc_20260318_1k_1566.yaml" in res.stdout
    assert "configs/reference/reference_20260318_1k_13440.yaml" in res.stdout
    assert "configs/train/smoke_1024.yaml" in res.stdout


def test_gisec_recovery_smoke_runner_lists_q0_q1_q2(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_legacy_recovery_smoke.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "variant=Q0" in res.stdout
    assert "variant=Q1" in res.stdout
    assert "variant=Q2" in res.stdout
    assert "configs/train/recovery_smoke_1024.yaml" in res.stdout
