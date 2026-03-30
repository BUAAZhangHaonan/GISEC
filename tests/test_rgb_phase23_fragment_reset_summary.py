from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_rgb_phase23_fragment_reset_summary_handles_stage2_gate_stop(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output"
    baseline_summary = tmp_path / "baseline_run_summary.json"
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    out_stage2 = tmp_path / "stage2_table.md"
    out_stage3 = tmp_path / "stage3_table.md"
    out_stage2_chart = tmp_path / "stage2.png"
    out_stage3_outcome = tmp_path / "stage3_outcome.png"
    out_stage3_failures = tmp_path / "stage3_failures.png"

    _write_json(
        baseline_summary,
        {
            "variant": "base_rgb_1024",
            "metrics": {
                "segm/AP": 0.5459,
                "bbox/AP": 0.4933,
                "boundary/IoU": 0.1894,
                "split_gt_count": 593,
                "merge_pred_count": 668,
            },
            "inference_speed": {"throughput_fps": 11.69},
        },
    )
    _write_json(
        output_root / "fragment_generator_cache" / "train" / "manifest.json",
        {
            "num_samples": 72000,
            "num_negative_samples": 0,
            "num_overflow_crops": 68000,
        },
    )
    _write_json(
        output_root / "fragment_generator_cache" / "val" / "manifest.json",
        {
            "num_samples": 8500,
            "num_negative_samples": 0,
            "num_overflow_crops": 8100,
        },
    )
    _write_json(
        output_root / "fragment_generator_rgb_stage2" / "train_summary.json",
        {
            "loss_total": 1.23,
            "covered_gt_rate": 0.80,
            "split_gt_rate": 0.10,
            "singleton_gt_rate": 0.90,
            "impure_fragment_rate": 0.40,
            "leakage_rate": 0.30,
            "fragments_per_covered_gt": 1.1,
            "empty_slot_rate": 0.02,
            "overflow_crop_rate": 0.94,
        },
    )
    _write_json(
        output_root / "fragment_generator_rgb_stage2" / "val_summary.json",
        {
            "covered_gt_rate": 0.79,
            "split_gt_rate": 0.09,
            "singleton_gt_rate": 0.91,
            "impure_fragment_rate": 0.42,
            "leakage_rate": 0.31,
            "fragments_per_covered_gt": 1.05,
            "empty_slot_rate": 0.03,
            "overflow_crop_rate": 0.95,
        },
    )
    _write_json(
        output_root / "fragment_generator_exports" / "val" / "eval_summary.json",
        {
            "covered_gt_rate": 0.79,
            "split_gt_rate": 0.09,
            "singleton_gt_rate": 0.91,
            "impure_fragment_rate": 0.42,
            "leakage_rate": 0.31,
            "fragments_per_covered_gt": 1.05,
            "empty_slot_rate": 0.03,
            "overflow_crop_rate": 0.95,
            "gate_passed": False,
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_rgb_phase23_fragment_reset.py",
            "--output-root",
            str(output_root),
            "--baseline-run-summary",
            str(baseline_summary),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-stage2-md",
            str(out_stage2),
            "--output-stage3-md",
            str(out_stage3),
            "--output-stage2-chart",
            str(out_stage2_chart),
            "--output-stage3-outcome-chart",
            str(out_stage3_outcome),
            "--output-stage3-failure-chart",
            str(out_stage3_failures),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["stage2_gate_passed"] is False
    assert payload["stage3_status"] == "gated_off"
    assert payload["baseline"]["segm/AP"] == 0.5459
    markdown = out_md.read_text(encoding="utf-8")
    assert "Stage 2 Gate" in markdown
    assert "gated off" in markdown
    assert out_stage2.exists()
    assert out_stage3.exists()
    assert out_stage2_chart.exists()
    assert out_stage3_outcome.exists()
    assert out_stage3_failures.exists()
