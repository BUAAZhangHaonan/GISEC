from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_run_summary(path: Path, *, variant: str, segm_ap: float, split_gt_count: int, merge_pred_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "variant": variant,
                "metrics": {
                    "segm/AP": segm_ap,
                    "bbox/AP": segm_ap + 0.1,
                    "boundary/IoU": 0.1,
                    "split_gt_count": split_gt_count,
                    "merge_pred_count": merge_pred_count,
                    "refinement_invocation_rate": 0.0,
                    "local_graph_invocation_rate": 0.0,
                },
                "inference_speed": {
                    "throughput_fps": 10.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_active_pilot_summary_writes_json_markdown_and_charts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary_a = tmp_path / "base_rgb" / "run_summary.json"
    summary_b = tmp_path / "base_rgbd" / "run_summary.json"
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    out_ap = tmp_path / "ap.png"
    out_failures = tmp_path / "failures.png"
    _write_run_summary(summary_a, variant="base_rgb_1024", segm_ap=0.54, split_gt_count=10, merge_pred_count=20)
    _write_run_summary(summary_b, variant="base_rgbd_1024", segm_ap=0.23, split_gt_count=30, merge_pred_count=40)

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_active_pilot.py",
            "--run-summary",
            str(summary_a),
            "--run-summary",
            str(summary_b),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-ap-chart",
            str(out_ap),
            "--output-failure-chart",
            str(out_failures),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["best_variant"] == "base_rgb_1024"
    assert len(payload["rows"]) == 2
    markdown = out_md.read_text(encoding="utf-8")
    assert "base_rgb_1024" in markdown
    assert "base_rgbd_1024" in markdown
    assert out_ap.exists()
    assert out_failures.exists()
