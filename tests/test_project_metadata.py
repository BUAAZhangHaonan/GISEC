from __future__ import annotations

from pathlib import Path
import subprocess


def test_project_metadata_uses_gisec_identity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    environment_text = (repo_root / "environment.yml").read_text(encoding="utf-8")
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    handoff_text = (repo_root / "docs" / "new-session-handoff.md").read_text(encoding="utf-8")
    reading_pack_text = (repo_root / "docs" / "reading-pack.md").read_text(encoding="utf-8")

    assert 'name = "gisec"' in pyproject_text
    assert 'include = ["gisec*"]' in pyproject_text
    assert 'description = "GISEC: Graph-based Instance Segmentation for Electronic Components"' in pyproject_text
    assert 'name: gisec' in environment_text
    assert "# GISEC: Graph-based Instance Segmentation for Electronic Components" in readme_text
    assert "affinigraph" not in pyproject_text
    assert "--prototype-root" in readme_text
    assert "prototype_bank_v1" not in readme_text
    assert "prototype_bank_v1" not in handoff_text
    assert "/home/k100/zhn/electronic-components-grasp-and-segment/gisec" in handoff_text
    assert "/home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference" in readme_text
    assert "/home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference" in handoff_text
    assert "../../magformer/docs/plans/" in reading_pack_text
    assert "../magformer/docs/plans/" in handoff_text


def test_project_metadata_includes_baseline_scaffold() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_init = repo_root / "baseline" / "__init__.py"
    baseline_readme = repo_root / "baseline" / "README.md"

    assert baseline_init.exists()
    assert baseline_readme.exists()
    readme_text = baseline_readme.read_text(encoding="utf-8")
    assert "baseline benchmark stack" in readme_text
    assert "separate from `gisec/`" in readme_text


def test_project_docs_freeze_v15_and_query_alpha_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    method_readme = (repo_root / "docs" / "method" / "README.md").read_text(encoding="utf-8")
    results_readme = (repo_root / "docs" / "results" / "README.md").read_text(encoding="utf-8")

    assert "GISEC v1.5 legacy" in readme_text
    assert "GISEC Query Alpha" in readme_text
    assert "fragment-first" in readme_text
    assert "object-first" in readme_text
    assert "base_rgb_1024" in readme_text
    assert "Mask2Former RGB @1024" in readme_text
    assert "Mask R-CNN RGB @1024" in readme_text
    assert "RGB-D is deferred" in readme_text
    assert "v1.5 legacy" in method_readme
    assert "query-alpha object-first" in method_readme
    assert "active benchmark surface" in results_readme
    assert "2026-03-28-active-surface-pilot-summary.md" in results_readme
    assert "2026-03-29-rgb-phase1-backbone-summary.md" in results_readme
    assert "2026-03-29-rgb-weekend-pipeline-summary.md" in results_readme
    assert "2026-03-29-phase3-prerequisite-diagnostics.md" in results_readme


def test_project_metadata_includes_formal_gisec_query_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "gisec" / "cli" / "train_query.py").exists()
    assert (repo_root / "gisec" / "cli" / "eval_query.py").exists()
    assert (repo_root / "gisec" / "config" / "query_models.py").exists()
    assert (repo_root / "gisec" / "engine" / "query_factory.py").exists()
    assert (repo_root / "gisec" / "engine" / "query_runtime.py").exists()
    assert (repo_root / "gisec" / "models" / "query_uq_backbone.py").exists()
    assert (repo_root / "gisec" / "train" / "query_targets.py").exists()
    assert (repo_root / "gisec" / "train" / "train_query.py").exists()
    assert (repo_root / "tests" / "query" / "__init__.py").exists()


def test_project_metadata_includes_query_alpha_runner_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner_path = repo_root / "scripts" / "experiments" / "run_gisec_query_uq.sh"

    assert runner_path.exists()
    runner_text = runner_path.read_text(encoding="utf-8")
    assert "gisec.cli.train_query" in runner_text
    assert "gisec.cli.eval_query" in runner_text
    assert "configs/query/model/uq_" in runner_text
    assert "[gisec-query-uq]" in runner_text


def test_tracked_source_paths_do_not_use_version_number_names() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    monitored_roots = ("gisec/", "configs/", "tests/", "scripts/", "docs/", "baseline/")
    versioned = [
        path
        for path in tracked
        if path.startswith(monitored_roots)
        and any(token in path for token in ("/v2", "_v2", "-v2", "/v3", "_v3", "-v3", "yolov8"))
    ]

    assert versioned == []
