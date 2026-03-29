from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_run_summary(
    path: Path,
    *,
    variant: str,
    model_family: str,
    resolution: int,
    segm_ap: float,
    bbox_ap: float,
    boundary_iou: float,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": model_family,
                "variant": variant,
                "modality": "rgb",
                "benchmark": {
                    "model_family": model_family,
                    "resolution": resolution,
                    "input_mode": "rgb",
                },
                "metrics": {
                    "segm/AP": segm_ap,
                    "bbox/AP": bbox_ap,
                    "boundary/IoU": boundary_iou,
                },
                "inference_speed": {
                    "throughput_fps": fps,
                },
            }
        ),
        encoding="utf-8",
    )


def test_rgb_phase1_summary_writes_json_markdown_and_charts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    short_a = tmp_path / "mask_rcnn_256" / "run_summary.json"
    short_b = tmp_path / "mask2former_256" / "run_summary.json"
    short_c = tmp_path / "mask_rcnn_1024" / "run_summary.json"
    short_d = tmp_path / "mask2former_1024" / "run_summary.json"
    full_a = tmp_path / "mask_rcnn_full" / "run_summary.json"
    full_b = tmp_path / "mask2former_full" / "run_summary.json"
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    out_short = tmp_path / "short.png"
    out_full = tmp_path / "full.png"

    _write_run_summary(
        short_a,
        variant="mask_rcnn_r50_256_phasea_short",
        model_family="mask_rcnn",
        resolution=256,
        segm_ap=0.01,
        bbox_ap=0.02,
        boundary_iou=0.03,
        fps=8.0,
    )
    _write_run_summary(
        short_b,
        variant="mask2former_swin_t_256_phasea_short",
        model_family="mask2former",
        resolution=256,
        segm_ap=0.04,
        bbox_ap=0.05,
        boundary_iou=0.06,
        fps=12.0,
    )
    _write_run_summary(
        short_c,
        variant="mask_rcnn_r50_1024_phasea_short",
        model_family="mask_rcnn",
        resolution=1024,
        segm_ap=0.10,
        bbox_ap=0.11,
        boundary_iou=0.12,
        fps=7.0,
    )
    _write_run_summary(
        short_d,
        variant="mask2former_swin_t_1024_phasea_short",
        model_family="mask2former",
        resolution=1024,
        segm_ap=0.25,
        bbox_ap=0.26,
        boundary_iou=0.27,
        fps=10.0,
    )
    _write_run_summary(
        full_a,
        variant="mask_rcnn_r50_1024_phasea_full",
        model_family="mask_rcnn",
        resolution=1024,
        segm_ap=0.52,
        bbox_ap=0.49,
        boundary_iou=0.14,
        fps=11.4,
    )
    _write_run_summary(
        full_b,
        variant="mask2former_swin_t_1024_phasea_full",
        model_family="mask2former",
        resolution=1024,
        segm_ap=0.55,
        bbox_ap=0.49,
        boundary_iou=0.19,
        fps=11.7,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_rgb_phase1_backbones.py",
            "--short-run-summary",
            str(short_a),
            "--short-run-summary",
            str(short_b),
            "--short-run-summary",
            str(short_c),
            "--short-run-summary",
            str(short_d),
            "--full-run-summary",
            str(full_a),
            "--full-run-summary",
            str(full_b),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-short-chart",
            str(out_short),
            "--output-full-chart",
            str(out_full),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["phase1_winner"] == "Mask2Former"
    assert len(payload["short_rows"]) == 4
    assert len(payload["full_rows"]) == 2
    markdown = out_md.read_text(encoding="utf-8")
    assert "Mask R-CNN" in markdown
    assert "Mask2Former" in markdown
    assert out_short.exists()
    assert out_full.exists()
