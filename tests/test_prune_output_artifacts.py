from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "maintenance" / "prune_output_artifacts.py"
    spec = importlib.util.spec_from_file_location("prune_output_artifacts_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prune_output_artifacts_dry_run_lists_safe_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    output_root = tmp_path / "output"
    analysis_tmp = output_root / "analysis" / "eval_profile_overlays_tmp123"
    stale_experiment = output_root / "experiments" / "old_suite"
    stale_baseline = output_root / "experiments" / "baselines" / "old_baseline"
    analysis_tmp.mkdir(parents=True)
    stale_experiment.mkdir(parents=True)
    stale_baseline.mkdir(parents=True)

    monkeypatch.setattr(module, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(module, "EXPERIMENTS_ROOT", output_root / "experiments")
    monkeypatch.setattr(module, "BASELINES_ROOT", output_root / "experiments" / "baselines")
    monkeypatch.setattr(sys, "argv", ["prune_output_artifacts.py"])

    module.main()

    stdout = capsys.readouterr().out
    assert "Found 3 paths to remove:" in stdout
    assert str(analysis_tmp) in stdout
    assert str(stale_experiment) in stdout
    assert str(stale_baseline) in stdout
    assert analysis_tmp.exists()
    assert stale_experiment.exists()
    assert stale_baseline.exists()


def test_prune_output_artifacts_allows_safe_removals_within_output_root(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "output"
    analysis_tmp = output_root / "analysis" / "eval_profile_overlays_tmp123"
    stale_experiment = output_root / "experiments" / "old_suite"
    stale_baseline = output_root / "experiments" / "baselines" / "old_baseline"
    kept_experiment = output_root / "experiments" / "keep_me"
    kept_baseline = output_root / "experiments" / "baselines" / "keep_baseline"
    for path in [analysis_tmp, stale_experiment, stale_baseline, kept_experiment, kept_baseline]:
        path.mkdir(parents=True)

    removals = module._collect_removals(
        output_root=output_root,
        experiments_root=output_root / "experiments",
        baselines_root=output_root / "experiments" / "baselines",
        keep_experiment_dirs={"keep_me"},
        keep_baseline_dirs={"keep_baseline"},
    )
    approved = module._approve_removals(removals, output_root=output_root)

    assert analysis_tmp in approved
    assert stale_experiment in approved
    assert stale_baseline in approved
    assert kept_experiment not in approved
    assert kept_baseline not in approved

    module._remove_paths(approved)

    assert not analysis_tmp.exists()
    assert not stale_experiment.exists()
    assert not stale_baseline.exists()
    assert kept_experiment.exists()
    assert kept_baseline.exists()


def test_prune_output_artifacts_rejects_symlink_escape(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "output"
    experiments_root = output_root / "experiments"
    outside_root = tmp_path / "outside"
    outside_root.mkdir(parents=True)
    (outside_root / "payload.txt").write_text("outside\n", encoding="utf-8")
    experiments_root.mkdir(parents=True)
    (experiments_root / "escape").symlink_to(outside_root, target_is_directory=True)

    removals = module._collect_removals(
        output_root=output_root,
        experiments_root=experiments_root,
        baselines_root=experiments_root / "baselines",
        keep_experiment_dirs=set(),
        keep_baseline_dirs=set(),
    )

    with pytest.raises(ValueError, match="symlink"):
        module._approve_removals(removals, output_root=output_root)
