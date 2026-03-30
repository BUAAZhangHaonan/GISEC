from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_rgb_phase23_instance_local_stage2_summary_writes_learned_owner_union_bundle(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output"
    baseline_summary = tmp_path / "baseline_run_summary.json"
    oracle_summary = tmp_path / "oracle_summary.json"
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    out_table = tmp_path / "comparison.md"
    out_chart = tmp_path / "comparison.png"

    _write_json(
        baseline_summary,
        {
            "variant": "base_rgb_1024",
            "metrics": {
                "segm/AP": 0.5451,
                "bbox/AP": 0.4934,
                "boundary/IoU": 0.1894,
                "split_gt_count": 593,
                "merge_pred_count": 668,
            },
        },
    )
    _write_json(
        oracle_summary,
        {
            "oracles": {
                "oracle_fragments_no_merge": {
                    "metrics": {"segm/AP": 0.1434, "boundary/IoU": 0.7240},
                    "split_gt_count": 3708,
                    "merge_pred_count": 4,
                },
                "oracle_owner_union": {
                    "metrics": {"segm/AP": 0.8489, "boundary/IoU": 0.9227},
                    "split_gt_count": 2,
                    "merge_pred_count": 1,
                },
            },
        },
    )
    _write_json(
        output_root / "instance_fragment_generator_rgb_stage2" / "eval_val" / "eval_summary.json",
        {
            "covered_instance_rate": 0.93,
            "split_instance_rate": 0.34,
            "singleton_instance_rate": 0.50,
            "impure_fragment_rate": 0.09,
            "leakage_rate": 0.04,
            "fragments_per_covered_instance": 1.8,
            "negative_anchor_empty_precision": 0.95,
            "negative_anchor_false_fragment_mean": 0.05,
            "query_overflow_rate": 0.0,
            "truncated_fragment_total": 0,
            "owner_union_segm/AP": 0.6123,
            "owner_union_boundary/IoU": 0.2500,
            "owner_union_split_gt_count": 420,
            "owner_union_merge_pred_count": 390,
            "learned_fragments_no_merge_segm/AP": 0.2011,
            "learned_fragments_no_merge_boundary/IoU": 0.3000,
            "learned_fragments_no_merge_split_gt_count": 900,
            "learned_fragments_no_merge_merge_pred_count": 30,
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_rgb_phase23_instance_local_stage2.py",
            "--output-root",
            str(output_root),
            "--baseline-run-summary",
            str(baseline_summary),
            "--oracle-summary-json",
            str(oracle_summary),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-table-md",
            str(out_table),
            "--output-chart",
            str(out_chart),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["learned"]["owner_union"]["segm/AP"] == 0.6123
    assert payload["decision"]["stage3_reentry_allowed"] is False
    table = out_table.read_text(encoding="utf-8")
    assert "learned_owner_union" in table
    assert "oracle_owner_union" in table
    assert out_chart.exists()
