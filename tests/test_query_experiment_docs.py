from __future__ import annotations

from pathlib import Path


def test_query_alpha_experiment_ladder_orders_phases_correctly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "archive" / "experiments" / "gisec-query-ladder.md").read_text(encoding="utf-8")

    assert "UQ-s" in text
    assert "UQ-m" in text
    assert "UR-*" in text
    assert "UG-*" in text
    assert "UA-*" in text
    assert text.index("UQ-s") < text.index("UR-*")
    assert text.index("UQ-m") < text.index("UG-*")
    assert text.index("UG-*") < text.index("UA-*")
