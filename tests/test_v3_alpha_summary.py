from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_run(
    root: Path,
    *,
    variant: str,
    segm_ap: float,
    bbox_ap: float,
    pred_count_mean: float,
    gt_count_mean: float,
    best_mask_iou_mean: float,
    best_bbox_iou_mean: float,
    object_count_mean: float,
    split_count_mean: float,
    failure_counts: dict[str, int],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_summary.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "model_id": variant,
                "metrics": {"segm/AP": segm_ap, "bbox/AP": bbox_ap},
                "inference_speed": {"throughput_fps": 5.0},
                "params_trainable": 1000,
                "wall_time_sec": 12,
            }
        ),
        encoding="utf-8",
    )
    (root / "match_diagnostics_summary.json").write_text(
        json.dumps(
            {
                "pred_count_mean": pred_count_mean,
                "gt_count_mean": gt_count_mean,
                "best_mask_iou_mean": best_mask_iou_mean,
                "best_bbox_iou_mean": best_bbox_iou_mean,
            }
        ),
        encoding="utf-8",
    )
    (root / "object_pathology_summary.json").write_text(
        json.dumps(
            {
                "object_count_mean": object_count_mean,
                "split_count_mean": split_count_mean,
                "avg_cores_per_object_mean": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (root / "failure_summary.json").write_text(
        json.dumps({"total_images": sum(failure_counts.values()), "counts": failure_counts}),
        encoding="utf-8",
    )


def test_v3_alpha_summary_script_writes_json_and_markdown(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "suite"
    output_json = tmp_path / "alpha_summary.json"
    output_md = tmp_path / "alpha_summary.md"
    _write_run(
        suite_root / "legacy",
        variant="v1.5 legacy",
        segm_ap=0.11,
        bbox_ap=0.12,
        pred_count_mean=1.5,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.25,
        best_bbox_iou_mean=0.35,
        object_count_mean=1.5,
        split_count_mean=0.1,
        failure_counts={"normal": 2, "empty": 1},
    )
    _write_run(
        suite_root / "uq_s",
        variant="UQ-s",
        segm_ap=0.21,
        bbox_ap=0.23,
        pred_count_mean=1.9,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.45,
        best_bbox_iou_mean=0.55,
        object_count_mean=1.9,
        split_count_mean=0.2,
        failure_counts={"normal": 3},
    )
    _write_run(
        suite_root / "uq_m",
        variant="UQ-m",
        segm_ap=0.28,
        bbox_ap=0.31,
        pred_count_mean=2.0,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.52,
        best_bbox_iou_mean=0.63,
        object_count_mean=2.0,
        split_count_mean=0.3,
        failure_counts={"normal": 3},
    )
    _write_run(
        suite_root / "ur_s",
        variant="UR-s",
        segm_ap=0.99,
        bbox_ap=0.99,
        pred_count_mean=2.0,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.99,
        best_bbox_iou_mean=0.99,
        object_count_mean=2.0,
        split_count_mean=0.1,
        failure_counts={"normal": 3},
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_v3_alpha_ladder.py",
            "--suite-root",
            str(suite_root),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_json.exists()
    assert output_md.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["rows"][0]["variant"] == "v1.5 legacy"
    assert payload["rows"][1]["variant"] == "UQ-s"
    assert payload["rows"][2]["variant"] == "UQ-m"
    assert all(row["variant"] != "UR-s" for row in payload["rows"])
    assert payload["gates"]["gate_a_pass"] is True
    assert payload["gates"]["gate_b_pass"] is True
    markdown = output_md.read_text(encoding="utf-8")
    assert "v1.5 legacy" in markdown
    assert "UQ-s" in markdown
    assert "UQ-m" in markdown
    assert "UR-s" not in markdown


def test_v3_alpha_summary_script_rejects_duplicate_official_variants(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "suite"
    output_json = tmp_path / "alpha_summary.json"
    output_md = tmp_path / "alpha_summary.md"
    _write_run(
        suite_root / "uq_s_a",
        variant="UQ-s",
        segm_ap=0.21,
        bbox_ap=0.23,
        pred_count_mean=1.9,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.45,
        best_bbox_iou_mean=0.55,
        object_count_mean=1.9,
        split_count_mean=0.2,
        failure_counts={"normal": 3},
    )
    _write_run(
        suite_root / "uq_s_b",
        variant="UQ-s",
        segm_ap=0.22,
        bbox_ap=0.24,
        pred_count_mean=1.9,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.46,
        best_bbox_iou_mean=0.56,
        object_count_mean=1.9,
        split_count_mean=0.2,
        failure_counts={"normal": 3},
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_v3_alpha_ladder.py",
            "--suite-root",
            str(suite_root),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
