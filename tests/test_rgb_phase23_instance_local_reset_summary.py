from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_rgb_phase23_instance_local_reset_summary_writes_cache_and_oracle_bundle(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output"
    baseline_summary = tmp_path / "baseline_run_summary.json"
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    out_cache = tmp_path / "cache_table.md"
    out_oracle = tmp_path / "oracle_table.md"
    out_fragment_chart = tmp_path / "fragment_counts.png"
    out_oracle_chart = tmp_path / "oracles.png"

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
        output_root / "instance_fragment_cache_pred" / "val" / "manifest.json",
        {
            "num_samples": 9040,
            "positive_anchor_count": 8740,
            "negative_anchor_count": 300,
            "matchable_gt_count": 8688,
            "total_gt_instances": 9040,
            "matchable_gt_rate": 0.9602,
            "raw_fragment_count_mean": 4.8,
            "raw_fragment_count_p50": 4.0,
            "raw_fragment_count_p75": 6.0,
            "raw_fragment_count_p90": 8.0,
            "raw_fragment_count_p95": 9.0,
            "raw_fragment_count_max": 14,
        },
    )
    _write_json(
        output_root / "instance_fragment_cache_gt" / "val" / "manifest.json",
        {
            "num_samples": 9040,
            "raw_fragment_count_mean": 4.9,
            "raw_fragment_count_p50": 4.0,
            "raw_fragment_count_p75": 6.0,
            "raw_fragment_count_p90": 8.0,
            "raw_fragment_count_p95": 9.0,
            "raw_fragment_count_max": 14,
        },
    )
    _write_json(
        output_root / "instance_fragment_oracles" / "val" / "oracle_fragments_no_merge" / "eval_summary.json",
        {
            "split_gt_count": 820,
            "merge_pred_count": 120,
            "metrics": {
                "segm/AP": 0.4020,
                "boundary/IoU": 0.2110,
            },
        },
    )
    _write_json(
        output_root / "instance_fragment_oracles" / "val" / "oracle_owner_union" / "eval_summary.json",
        {
            "split_gt_count": 410,
            "merge_pred_count": 500,
            "metrics": {
                "segm/AP": 0.5820,
                "boundary/IoU": 0.2250,
            },
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_rgb_phase23_instance_local_reset.py",
            "--output-root",
            str(output_root),
            "--baseline-run-summary",
            str(baseline_summary),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-cache-md",
            str(out_cache),
            "--output-oracle-md",
            str(out_oracle),
            "--output-fragment-chart",
            str(out_fragment_chart),
            "--output-oracle-chart",
            str(out_oracle_chart),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["baseline"]["segm/AP"] == 0.5451
    assert payload["cache_pred"]["raw_fragment_count_p95"] == 9.0
    assert payload["decision"]["oracle_gate_passed"] is True
    markdown = out_md.read_text(encoding="utf-8")
    assert "Instance-Local Reset Summary" in markdown
    assert "oracle_owner_union" in out_oracle.read_text(encoding="utf-8")
    assert out_cache.exists()
    assert out_fragment_chart.exists()
    assert out_oracle_chart.exists()
