from __future__ import annotations

from pathlib import Path


def test_v3_alpha_full_run_entry_doc_locks_promotion_conditions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "experiments" / "gisec-v3-alpha-full-run-entry.md").read_text(encoding="utf-8")

    assert "full runs are forbidden" in text
    assert "reference" in text
    assert "graph" in text
    assert "GPU is available" in text
    assert "previous phase passes its relative gate" in text
