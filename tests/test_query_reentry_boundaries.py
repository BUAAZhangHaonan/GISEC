from __future__ import annotations

from pathlib import Path


def test_query_alpha_runner_surface_does_not_activate_later_phase_model_ids() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner_text = (repo_root / "scripts" / "experiments" / "run_gisec_query_uq.sh").read_text(encoding="utf-8")

    assert "query_ref_resnet18" not in runner_text
    assert "query_graph_resnet18" not in runner_text
    assert "query_refgraph_resnet18" not in runner_text


def test_query_alpha_summary_script_filters_to_uq_only_table() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "scripts" / "analysis" / "summarize_query_alpha_ladder.py").read_text(encoding="utf-8")

    assert "query_small_resnet18" in source
    assert "query_medium_resnet34" in source
    assert "query_ref_resnet18" in source
    assert "query_graph_resnet18" in source
    assert "query_refgraph_resnet18" in source


def test_query_docs_keep_ur_ug_ua_documented_but_not_alpha_activated() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ladder_text = (repo_root / "docs" / "archive" / "experiments" / "gisec-query-ladder.md").read_text(encoding="utf-8")
    short_run_text = (repo_root / "docs" / "archive" / "experiments" / "gisec-query-short-run-protocol.md").read_text(encoding="utf-8")
    full_entry_text = (repo_root / "docs" / "archive" / "experiments" / "gisec-query-full-run-entry.md").read_text(encoding="utf-8")

    assert "UQ-s" in ladder_text
    assert "UQ-m" in ladder_text
    assert "UR-*" in ladder_text
    assert "UG-*" in ladder_text
    assert "UA-*" in ladder_text
    assert "UQ-s" in short_run_text
    assert "UQ-m" in short_run_text
    assert "full runs are forbidden until the previous phase passes its relative gate" in full_entry_text
