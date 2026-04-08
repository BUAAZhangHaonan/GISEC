from __future__ import annotations

import subprocess
from pathlib import Path


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
    assert "stage=base_rgbd_1024" in stdout
    assert "stage=base_rgbd_1024_refine_ref_graph" in stdout
    refine_init = tmp_path / "active_official" / "train" / "base_rgbd_1024" / "model_best.pth"
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
    assert "stage=base_rgb_1024" in stdout
    assert "stage=base_rgb_1024_refine_ref_graph" in stdout
    refine_init = tmp_path / "active_rgb_official" / "train" / "base_rgb_1024" / "model_best.pth"
    assert f"--init-checkpoint '{refine_init}'" in stdout or f"--init-checkpoint {refine_init}" in stdout
    assert "conda run -n gisec python -m gisec.cli.train" in stdout
    assert "gisec.cli.eval" in stdout
    assert "--eval-every-epochs 0" in stdout


def test_active_mainline_ladder_dry_run_skips_completed_stage(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_mainline.sh"
    output_root = tmp_path / "active_official"
    train_dir = output_root / "train" / "base_rgbd_1024"
    eval_dir = output_root / "eval" / "base_rgbd_1024"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (train_dir / "model_best.pth").write_text("stub\n", encoding="utf-8")
    (train_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
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
    assert "SKIP train base_rgbd_1024" in stdout
    assert "SKIP eval base_rgbd_1024" in stdout
    assert "stage=base_rgbd_1024_refine" in stdout


def test_active_rgb_mainline_ladder_dry_run_skips_completed_stage(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_rgb_mainline.sh"
    output_root = tmp_path / "active_rgb_official"
    train_dir = output_root / "train" / "base_rgb_1024"
    eval_dir = output_root / "eval" / "base_rgb_1024"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (train_dir / "model_best.pth").write_text("stub\n", encoding="utf-8")
    (train_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
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
    assert "SKIP train base_rgb_1024" in stdout
    assert "SKIP eval base_rgb_1024" in stdout
    assert "stage=base_rgb_1024_refine" in stdout


def test_active_rgb_mainline_ladder_dry_run_resumes_incomplete_stage_when_resume_checkpoint_exists(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_baseline_reset_active_rgb_mainline.sh"
    output_root = tmp_path / "active_rgb_official"

    stage1_train_dir = output_root / "train" / "base_rgb_1024"
    stage1_eval_dir = output_root / "eval" / "base_rgb_1024"
    stage1_train_dir.mkdir(parents=True)
    stage1_eval_dir.mkdir(parents=True)
    (stage1_train_dir / "model_best.pth").write_text("stub\n", encoding="utf-8")
    (stage1_train_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
    (stage1_eval_dir / "metrics.cocoeval.json").write_text("{}\n", encoding="utf-8")
    (stage1_eval_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")

    stage2_train_dir = output_root / "train" / "base_rgb_1024_refine"
    stage2_train_dir.mkdir(parents=True)
    resume_checkpoint = stage2_train_dir / "resume_last.pth"
    resume_checkpoint.write_text("stub\n", encoding="utf-8")

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
    assert "stage=base_rgb_1024_refine" in stdout
    assert f"--resume-checkpoint '{resume_checkpoint}'" in stdout or f"--resume-checkpoint {resume_checkpoint}" in stdout


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
    g1_checkpoint = tmp_path / "phase_a" / "legacy" / "G1_train" / "model_best.pth"
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
    assert "stage=G3_train" in stdout
    assert "stage=G3_best_eval" in stdout
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
    assert "stage=G1_edge_type_8d_train" in stdout
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
        "base_rgbd_1024",
        "base_rgbd_1024_refine",
        "base_rgbd_1024_refine_ref",
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
