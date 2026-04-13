from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _write_run_state(path: Path, *, status: str, allow_resume: bool = False, failure_reason: str | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "allow_resume": allow_resume,
                "failure_reason": failure_reason,
                "last_finite_step": 0,
                "last_finite_checkpoint": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_active_mainline_ladder_dry_run_lists_stage_chain(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_mainline.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(tmp_path / "active_official"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "conda run -n gisec python -m gisec.cli.train" in stdout
    assert "stage=base_mask2former_training" in stdout
    assert "stage=local_refinement_training" in stdout
    assert "stage=reference_conditioning_training" not in stdout
    assert "stage=graph_rescue_training" not in stdout
    refine_init = tmp_path / "active_official" / "train" / "base_mask2former_training" / "model_best.pth"
    assert f"--init-checkpoint '{refine_init}'" in stdout or f"--init-checkpoint {refine_init}" in stdout
    assert "gisec.cli.eval" in stdout


def test_active_rgb_mainline_ladder_dry_run_lists_stage_chain(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_rgb_mainline.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(tmp_path / "active_rgb_official"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "stage=base_mask2former_training" in stdout
    assert "stage=local_refinement_training" in stdout
    assert "stage=reference_conditioning_training" not in stdout
    assert "stage=graph_rescue_training" not in stdout
    refine_init = tmp_path / "active_rgb_official" / "train" / "base_mask2former_training" / "model_best.pth"
    assert f"--init-checkpoint '{refine_init}'" in stdout or f"--init-checkpoint {refine_init}" in stdout
    assert "conda run -n gisec python -m gisec.cli.train" in stdout
    assert "gisec.cli.eval" in stdout
    assert "--eval-every-epochs 0" in stdout


def test_active_rgb_mainline_dry_run_includes_experimental_rescue_stages_only_when_requested(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_rgb_mainline.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(tmp_path / "active_rgb_official"),
            "--include-experimental-rescue-stages",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "stage=reference_conditioning_training" in stdout
    assert "stage=graph_rescue_training" in stdout


def test_active_mainline_ladder_dry_run_skips_completed_stage(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_mainline.sh"
    output_root = tmp_path / "active_official"
    train_dir = output_root / "train" / "base_mask2former_training"
    eval_dir = output_root / "eval" / "base_mask2former_training"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (train_dir / "model_best.pth").write_text("stub\n", encoding="utf-8")
    (train_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
    _write_run_state(train_dir / "run_state.json", status="success")
    (eval_dir / "metrics.cocoeval.json").write_text("{}\n", encoding="utf-8")
    (eval_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "SKIP train base_mask2former_training" in stdout
    assert "SKIP eval base_mask2former_training" in stdout
    assert "stage=local_refinement_training" in stdout


def test_active_rgb_mainline_ladder_dry_run_skips_completed_stage(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_rgb_mainline.sh"
    output_root = tmp_path / "active_rgb_official"
    train_dir = output_root / "train" / "base_mask2former_training"
    eval_dir = output_root / "eval" / "base_mask2former_training"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (train_dir / "model_best.pth").write_text("stub\n", encoding="utf-8")
    (train_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
    _write_run_state(train_dir / "run_state.json", status="success")
    (eval_dir / "metrics.cocoeval.json").write_text("{}\n", encoding="utf-8")
    (eval_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "SKIP train base_mask2former_training" in stdout
    assert "SKIP eval base_mask2former_training" in stdout
    assert "stage=local_refinement_training" in stdout


def test_active_rgb_mainline_ladder_dry_run_resumes_incomplete_stage_only_when_run_state_allows_it(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_rgb_mainline.sh"
    output_root = tmp_path / "active_rgb_official"

    stage1_train_dir = output_root / "train" / "base_mask2former_training"
    stage1_eval_dir = output_root / "eval" / "base_mask2former_training"
    stage1_train_dir.mkdir(parents=True)
    stage1_eval_dir.mkdir(parents=True)
    (stage1_train_dir / "model_best.pth").write_text("stub\n", encoding="utf-8")
    (stage1_train_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
    _write_run_state(stage1_train_dir / "run_state.json", status="success")
    (stage1_eval_dir / "metrics.cocoeval.json").write_text("{}\n", encoding="utf-8")
    (stage1_eval_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")

    stage2_train_dir = output_root / "train" / "local_refinement_training"
    stage2_train_dir.mkdir(parents=True)
    resume_checkpoint = stage2_train_dir / "resume_last.pth"
    resume_checkpoint.write_text("stub\n", encoding="utf-8")
    _write_run_state(stage2_train_dir / "run_state.json", status="running", allow_resume=True)

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "stage=local_refinement_training" in stdout
    assert f"--resume-checkpoint '{resume_checkpoint}'" in stdout or f"--resume-checkpoint {resume_checkpoint}" in stdout


def test_active_rgb_mainline_ladder_dry_run_does_not_resume_without_run_state_allow_resume(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_rgb_mainline.sh"
    output_root = tmp_path / "active_rgb_official"

    stage1_train_dir = output_root / "train" / "base_mask2former_training"
    stage1_eval_dir = output_root / "eval" / "base_mask2former_training"
    stage1_train_dir.mkdir(parents=True)
    stage1_eval_dir.mkdir(parents=True)
    (stage1_train_dir / "model_best.pth").write_text("stub\n", encoding="utf-8")
    (stage1_train_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
    _write_run_state(stage1_train_dir / "run_state.json", status="success")
    (stage1_eval_dir / "metrics.cocoeval.json").write_text("{}\n", encoding="utf-8")
    (stage1_eval_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")

    stage2_train_dir = output_root / "train" / "local_refinement_training"
    stage2_train_dir.mkdir(parents=True)
    resume_checkpoint = stage2_train_dir / "resume_last.pth"
    resume_checkpoint.write_text("stub\n", encoding="utf-8")
    _write_run_state(stage2_train_dir / "run_state.json", status="failed", allow_resume=False, failure_reason="non-finite loss")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "stage=local_refinement_training" in stdout
    assert f"--resume-checkpoint '{resume_checkpoint}'" not in stdout
    assert f"--resume-checkpoint {resume_checkpoint}" not in stdout


def test_launch_tmux_queue_dry_run_prints_session_and_command(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "launch_tmux_queue.sh"
    output_root = tmp_path / "queue"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--session-name",
            "gisec_test_session",
            "--output-root",
            str(output_root),
            "--dry-run",
            "--",
            "echo",
            "hello",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "gisec_test_session" in result.stdout
    assert "echo hello" in result.stdout


def test_monitor_gpu_util_writes_jsonl_rows(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "monitor_gpu_util.sh"
    output_path = tmp_path / "gpu_monitor.jsonl"
    fake_smi = tmp_path / "fake_nvidia_smi.sh"
    fake_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '10,2048,1234,512\\n'\n",
        encoding="utf-8",
    )
    fake_smi.chmod(0o755)

    subprocess.run(
        [
            "bash",
            str(script),
            "--output",
            str(output_path),
            "--interval-sec",
            "0",
            "--sample-count",
            "1",
            "--nvidia-smi-bin",
            str(fake_smi),
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    rows = output_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert "gpu_util" in rows[0]


def test_legacy_support_ladder_dry_run_lists_g3_and_merge_order_steps(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_legacy_support.sh"
    g1_checkpoint = tmp_path / "backbone_benchmark" / "legacy" / "legacy_prototype_unet_baseline_train" / "model_best.pth"
    g1_checkpoint.parent.mkdir(parents=True)
    g1_checkpoint.write_text("stub\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(tmp_path / "baseline_reset"),
            "--g1-checkpoint",
            str(g1_checkpoint),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "conda run -n gisec python -m gisec.cli.train_legacy" in stdout
    assert "stage=legacy_prototype_unet_with_graph_train" in stdout
    assert "stage=legacy_prototype_unet_with_graph_best_eval" in stdout
    assert "--merge-order score" in stdout
    assert "--merge-order random" in stdout


def test_edge_type_ablation_ladder_dry_run_uses_hidden_worktree(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_edge_type_ablation.sh"
    worktree = tmp_path / ".worktree" / "edge-type-ablation"
    (worktree / "scripts" / "experiments").mkdir(parents=True)

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--worktree-root",
            str(worktree),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(tmp_path / "baseline_reset"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "worktree_root=" in stdout
    assert "stage=legacy_prototype_unet_baseline_edge_type_8d_train" in stdout
    assert "conda run -n gisec python -m gisec.cli.train_legacy" in stdout


def test_active_ablation_ladder_dry_run_lists_two_worktree_experiments(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_ablations.sh"

    gt_mask_worktree = tmp_path / ".worktree" / "active-gt-mask-ablation"
    all_ones_worktree = tmp_path / ".worktree" / "active-all-ones-ablation"
    for worktree in [gt_mask_worktree, all_ones_worktree]:
        (worktree / "scripts" / "experiments").mkdir(parents=True)

    mainline_root = tmp_path / "active_official"
    for variant in [
        "base_mask2former_training",
        "local_refinement_training",
        "reference_conditioning_training",
    ]:
        stage_dir = mainline_root / "train" / variant
        stage_dir.mkdir(parents=True)
        (stage_dir / "model_best.pth").write_text("stub\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--gt-mask-worktree",
            str(gt_mask_worktree),
            "--all-ones-worktree",
            str(all_ones_worktree),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--mainline-root",
            str(mainline_root),
            "--output-root",
            str(tmp_path / "baseline_reset"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "stage=gt_mask_refine_train" in stdout
    assert "stage=all_ones_refine_ref_train" in stdout
    assert "stage=all_ones_refine_ref_graph_train" in stdout
    assert "conda run -n gisec python -m gisec.cli.train" in stdout
