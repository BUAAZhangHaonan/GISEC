from __future__ import annotations

from pathlib import Path


def test_v3_alpha_metrics_doc_defines_common_metrics_and_diagnostics() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "experiments" / "gisec-v3-alpha-metrics.md").read_text(encoding="utf-8")

    required = [
        "segm/AP",
        "bbox/AP",
        "pred_count_mean",
        "gt_count_mean",
        "best_mask_iou_mean",
        "best_bbox_iou_mean",
        "failure_summary",
    ]
    for token in required:
        assert token in text
