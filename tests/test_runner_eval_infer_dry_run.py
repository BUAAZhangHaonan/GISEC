from __future__ import annotations

import subprocess
from pathlib import Path


def test_gisec_eval_runner_dry_run_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec_eval.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(tmp_path / "out"),
            "--checkpoint",
            str(tmp_path / "out" / "model_best.pth"),
            "--variant",
            "legacy_prototype_unet_with_rgbd_similarity_shape_stats",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode=dry-run" in result.stdout
    assert "python -m gisec.cli.eval_legacy" in result.stdout


def test_gisec_infer_runner_dry_run_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec_infer.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(tmp_path / "out"),
            "--checkpoint",
            str(tmp_path / "out" / "model_best.pth"),
            "--variant",
            "legacy_prototype_unet_with_rgbd_similarity_shape_stats",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode=dry-run" in result.stdout
    assert "python -m gisec.cli.infer_legacy" in result.stdout
