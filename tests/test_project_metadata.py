from __future__ import annotations

from pathlib import Path


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
