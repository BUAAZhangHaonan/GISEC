from __future__ import annotations

from pathlib import Path


def test_query_alpha_official_baseline_summary_scaffold_has_required_gate_format() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_text = (repo_root / "docs" / "results" / "2026-04-13-query-alpha-official-baseline-summary.md").read_text(
        encoding="utf-8"
    )
    deferred_text = (repo_root / "docs" / "results" / "2026-04-13-query-alpha-deferred-variants-summary.md").read_text(
        encoding="utf-8"
    )

    baseline_required = [
        "query_small_resnet18",
        "query_medium_resnet34",
        "variant name",
        "segm/AP",
        "bbox/AP",
        "boundary/IoU",
        "train wall time",
        "Gate Evaluation",
        "Recommendation",
        "TBD",
        "NO-GO: official small and medium baseline runs have not been executed yet.",
    ]
    for token in baseline_required:
        assert token in baseline_text

    deferred_required = [
        "query_ref_resnet18",
        "query_ref_resnet34",
        "query_graph_resnet18",
        "query_graph_resnet34",
        "query_refgraph_resnet18",
        "query_refgraph_resnet34",
        "variant name",
        "segm/AP",
        "bbox/AP",
        "boundary/IoU",
        "train wall time",
        "Gate Evaluation",
        "Recommendation",
        "comparison vs active RGB refine-only baseline",
        "NO-GO: deferred query variants have not been executed yet.",
    ]
    for token in deferred_required:
        assert token in deferred_text
