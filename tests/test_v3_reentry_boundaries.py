from __future__ import annotations

from pathlib import Path


def test_v3_alpha_runner_surface_does_not_activate_later_phase_model_ids() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner_text = (repo_root / "scripts" / "experiments" / "run_gisec_v3_alpha_uq.sh").read_text(encoding="utf-8")

    assert "UR-s" not in runner_text
    assert "UG-s" not in runner_text
    assert "UA-s" not in runner_text


def test_v3_alpha_summary_script_filters_to_uq_only_table() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "scripts" / "analysis" / "summarize_v3_alpha_ladder.py").read_text(encoding="utf-8")

    assert "UQ-s" in source
    assert "UQ-m" in source
    assert "UR-s" not in source
    assert "UG-s" not in source
    assert "UA-s" not in source


def test_v3_docs_keep_ur_ug_ua_documented_but_not_alpha_activated() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ladder_text = (repo_root / "docs" / "experiments" / "gisec-v3-alpha-ladder.md").read_text(encoding="utf-8")
    short_run_text = (repo_root / "docs" / "experiments" / "gisec-v3-alpha-short-run-protocol.md").read_text(encoding="utf-8")
    full_entry_text = (repo_root / "docs" / "experiments" / "gisec-v3-alpha-full-run-entry.md").read_text(encoding="utf-8")

    assert "UR-*" in ladder_text
    assert "UG-*" in ladder_text
    assert "UA-*" in ladder_text
    assert "UR-*" not in short_run_text
    assert "UG-*" not in short_run_text
    assert "UA-*" not in short_run_text
    assert "UR" in full_entry_text
    assert "UG" in full_entry_text
