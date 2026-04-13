from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_gisec_all_runner_lists_full_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec_all.sh"
    text = script.read_text(encoding="utf-8")
    for variant in ["legacy_heuristic_graph_merge_baseline", "legacy_prototype_unet_baseline", "legacy_prototype_unet_refined", "legacy_prototype_unet_with_graph", "legacy_prototype_unet_with_rgbd_similarity", "legacy_prototype_unet_with_rgbd_similarity_shape_stats"]:
        assert variant in text


def test_gisec_all_runner_adds_post_train_eval_visualization_export(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec_all.sh"
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
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "python -m gisec.cli.train_legacy" in result.stdout
    assert "python -m gisec.cli.eval_legacy" in result.stdout
    assert "eval_vis" in result.stdout


def test_new_cli_modules_expose_train_eval_infer_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for module_name in ["gisec.cli.train", "gisec.cli.eval", "gisec.cli.infer"]:
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "base_rgb_1024" in result.stdout
        assert "base_rgbd_1024_refine_ref_graph" in result.stdout


def test_gisec_eval_and_infer_runner_scripts_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for script_name in [
        "run_legacy_1k_20ep_1024_gisec_eval.sh",
        "run_legacy_1k_20ep_1024_gisec_infer.sh",
    ]:
        assert (repo_root / "scripts" / "experiments" / script_name).exists()


def test_gisec_single_runner_accepts_a_variants_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_legacy_1k_20ep_1024_gisec.sh"

    for variant in ["legacy_rgbd_prototype_affinity_baseline", "legacy_rgbd_prototype_ownership_graph_cues"]:
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
                "--variant",
                variant,
                "--dry-run",
            ],
            cwd=str(tmp_path),
            check=True,
            capture_output=True,
            text=True,
        )

        assert f"--variant '{variant}'" in result.stdout
