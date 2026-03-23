from __future__ import annotations

from pathlib import Path


def test_query_alpha_gates_doc_uses_relative_promotion_rules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "experiments" / "gisec-query-gates.md").read_text(encoding="utf-8")

    assert "UQ-s" in text
    assert "v1.5 legacy" in text
    assert "UQ-m" in text
    assert "UR" in text
    assert "UG" in text
    assert "relative" in text
    assert "not vanity thresholds" in text
