from __future__ import annotations

from pathlib import Path


def test_query_alpha_short_run_protocol_locks_core_settings() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "experiments" / "gisec-query-short-run-protocol.md").read_text(encoding="utf-8")

    required = [
        "image size",
        "training length",
        "max validation images",
        "seed",
        "mandatory diagnostics artifacts",
    ]
    for token in required:
        assert token in text
