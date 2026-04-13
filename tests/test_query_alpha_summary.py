from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_official_run(
    suite_root: Path,
    *,
    variant: str,
    segm_ap: float,
    bbox_ap: float,
    boundary_iou: float,
    train_wall_time_sec: float,
    eval_wall_time_sec: float,
    pred_count_mean: float,
    gt_count_mean: float,
    best_mask_iou_mean: float,
    best_bbox_iou_mean: float,
    object_count_mean: float,
    split_count_mean: float,
    failure_counts: dict[str, int],
    use_reference: bool | None = False,
    use_graph_rescue: bool | None = False,
) -> None:
    train_root = suite_root / "train" / variant
    eval_root = suite_root / "eval" / variant
    train_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "variant": variant,
        "model_id": variant,
        "metrics": {"segm/AP": segm_ap, "bbox/AP": bbox_ap, "boundary/IoU": boundary_iou},
        "inference_speed": {"throughput_fps": 5.0},
        "params_trainable": 1000,
        "wall_time_sec": train_wall_time_sec,
    }
    if use_reference is not None:
        train_payload["use_reference"] = use_reference
    if use_graph_rescue is not None:
        train_payload["use_graph_rescue"] = use_graph_rescue
    (train_root / "run_summary.json").write_text(json.dumps(train_payload), encoding="utf-8")

    eval_payload = {
        "variant": variant,
        "model_id": variant,
        "metrics": {"segm/AP": segm_ap, "bbox/AP": bbox_ap, "boundary/IoU": boundary_iou},
        "inference_speed": {"throughput_fps": 5.0},
        "params_trainable": 1000,
        "wall_time_sec": eval_wall_time_sec,
    }
    if use_reference is not None:
        eval_payload["use_reference"] = use_reference
    if use_graph_rescue is not None:
        eval_payload["use_graph_rescue"] = use_graph_rescue
    (eval_root / "run_summary.json").write_text(json.dumps(eval_payload), encoding="utf-8")
    (eval_root / "match_diagnostics_summary.json").write_text(
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
    (eval_root / "object_pathology_summary.json").write_text(
        json.dumps(
            {
                "object_count_mean": object_count_mean,
                "split_count_mean": split_count_mean,
                "avg_cores_per_object_mean": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (eval_root / "failure_summary.json").write_text(
        json.dumps({"total_images": sum(failure_counts.values()), "counts": failure_counts}),
        encoding="utf-8",
    )


def test_query_alpha_summary_script_writes_json_and_markdown(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "20260413-query-alpha-official"
    output_json = tmp_path / "alpha_summary.json"
    output_md = tmp_path / "alpha_summary.md"
    _write_official_run(
        suite_root,
        variant="v1.5 legacy",
        segm_ap=0.11,
        bbox_ap=0.12,
        boundary_iou=0.07,
        train_wall_time_sec=33.0,
        eval_wall_time_sec=11.0,
        pred_count_mean=1.5,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.25,
        best_bbox_iou_mean=0.35,
        object_count_mean=1.5,
        split_count_mean=0.1,
        failure_counts={"normal": 2, "empty": 1},
        use_reference=None,
        use_graph_rescue=None,
    )
    _write_official_run(
        suite_root,
        variant="query_small_resnet18",
        segm_ap=0.21,
        bbox_ap=0.23,
        boundary_iou=0.18,
        train_wall_time_sec=44.0,
        eval_wall_time_sec=12.0,
        pred_count_mean=1.9,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.45,
        best_bbox_iou_mean=0.55,
        object_count_mean=1.9,
        split_count_mean=0.2,
        failure_counts={"normal": 3},
    )
    _write_official_run(
        suite_root,
        variant="query_medium_resnet34",
        segm_ap=0.28,
        bbox_ap=0.31,
        boundary_iou=0.24,
        train_wall_time_sec=55.0,
        eval_wall_time_sec=13.0,
        pred_count_mean=2.0,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.52,
        best_bbox_iou_mean=0.63,
        object_count_mean=2.0,
        split_count_mean=0.3,
        failure_counts={"normal": 3},
    )
    _write_official_run(
        suite_root,
        variant="query_ref_resnet18",
        segm_ap=0.33,
        bbox_ap=0.35,
        boundary_iou=0.26,
        train_wall_time_sec=58.0,
        eval_wall_time_sec=14.0,
        pred_count_mean=2.1,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.57,
        best_bbox_iou_mean=0.66,
        object_count_mean=2.1,
        split_count_mean=0.35,
        failure_counts={"normal": 2},
        use_reference=True,
        use_graph_rescue=False,
    )
    _write_official_run(
        suite_root,
        variant="query_ref_resnet34",
        segm_ap=0.36,
        bbox_ap=0.39,
        boundary_iou=0.29,
        train_wall_time_sec=60.0,
        eval_wall_time_sec=15.0,
        pred_count_mean=2.2,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.60,
        best_bbox_iou_mean=0.69,
        object_count_mean=2.2,
        split_count_mean=0.38,
        failure_counts={"normal": 1},
        use_reference=True,
        use_graph_rescue=False,
    )
    _write_official_run(
        suite_root,
        variant="query_graph_resnet18",
        segm_ap=0.31,
        bbox_ap=0.33,
        boundary_iou=0.25,
        train_wall_time_sec=57.0,
        eval_wall_time_sec=14.5,
        pred_count_mean=2.05,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.54,
        best_bbox_iou_mean=0.64,
        object_count_mean=2.05,
        split_count_mean=0.33,
        failure_counts={"normal": 2},
        use_reference=False,
        use_graph_rescue=True,
    )
    _write_official_run(
        suite_root,
        variant="query_graph_resnet34",
        segm_ap=0.34,
        bbox_ap=0.37,
        boundary_iou=0.27,
        train_wall_time_sec=59.0,
        eval_wall_time_sec=15.0,
        pred_count_mean=2.1,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.58,
        best_bbox_iou_mean=0.67,
        object_count_mean=2.1,
        split_count_mean=0.34,
        failure_counts={"normal": 1},
        use_reference=False,
        use_graph_rescue=True,
    )
    _write_official_run(
        suite_root,
        variant="query_refgraph_resnet18",
        segm_ap=0.38,
        bbox_ap=0.41,
        boundary_iou=0.31,
        train_wall_time_sec=63.0,
        eval_wall_time_sec=16.0,
        pred_count_mean=2.25,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.62,
        best_bbox_iou_mean=0.71,
        object_count_mean=2.25,
        split_count_mean=0.4,
        failure_counts={"normal": 1},
        use_reference=True,
        use_graph_rescue=True,
    )
    _write_official_run(
        suite_root,
        variant="query_refgraph_resnet34",
        segm_ap=0.4,
        bbox_ap=0.44,
        boundary_iou=0.33,
        train_wall_time_sec=65.0,
        eval_wall_time_sec=16.5,
        pred_count_mean=2.3,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.65,
        best_bbox_iou_mean=0.74,
        object_count_mean=2.3,
        split_count_mean=0.42,
        failure_counts={"normal": 1},
        use_reference=True,
        use_graph_rescue=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_query_alpha_ladder.py",
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
    assert payload["rows"][1]["variant"] == "query_small_resnet18"
    assert payload["rows"][2]["variant"] == "query_medium_resnet34"
    assert payload["rows"][3]["variant"] == "query_ref_resnet18"
    assert payload["rows"][5]["variant"] == "query_graph_resnet18"
    assert payload["rows"][7]["variant"] == "query_refgraph_resnet18"
    assert payload["rows"][1]["boundary/IoU"] == 0.18
    assert payload["rows"][1]["train_wall_time_sec"] == 44.0
    assert payload["gates"]["gate_a_pass"] is True
    assert payload["gates"]["gate_b_pass"] is True
    markdown = output_md.read_text(encoding="utf-8")
    assert "boundary/IoU" in markdown
    assert "train wall time (s)" in markdown
    assert "query_small_resnet18" in markdown
    assert "query_medium_resnet34" in markdown
    assert "query_ref_resnet18" in markdown
    assert "query_graph_resnet18" in markdown
    assert "query_refgraph_resnet18" in markdown


def test_query_alpha_summary_script_rejects_duplicate_official_variants(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "suite"
    output_json = tmp_path / "alpha_summary.json"
    output_md = tmp_path / "alpha_summary.md"
    _write_official_run(
        suite_root / "first",
        variant="query_small_resnet18",
        segm_ap=0.21,
        bbox_ap=0.23,
        boundary_iou=0.18,
        train_wall_time_sec=44.0,
        eval_wall_time_sec=12.0,
        pred_count_mean=1.9,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.45,
        best_bbox_iou_mean=0.55,
        object_count_mean=1.9,
        split_count_mean=0.2,
        failure_counts={"normal": 3},
    )
    _write_official_run(
        suite_root / "second",
        variant="query_small_resnet18",
        segm_ap=0.22,
        bbox_ap=0.24,
        boundary_iou=0.19,
        train_wall_time_sec=45.0,
        eval_wall_time_sec=13.0,
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
            "scripts/analysis/summarize_query_alpha_ladder.py",
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


def test_query_alpha_summary_script_allows_legacy_row_without_alpha_module_flags(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "suite"
    output_json = tmp_path / "alpha_summary.json"
    output_md = tmp_path / "alpha_summary.md"
    _write_official_run(
        suite_root,
        variant="v1.5 legacy",
        segm_ap=0.11,
        bbox_ap=0.12,
        boundary_iou=0.07,
        train_wall_time_sec=33.0,
        eval_wall_time_sec=11.0,
        pred_count_mean=1.5,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.25,
        best_bbox_iou_mean=0.35,
        object_count_mean=1.5,
        split_count_mean=0.1,
        failure_counts={"normal": 2, "empty": 1},
        use_reference=None,
        use_graph_rescue=None,
    )
    _write_official_run(
        suite_root,
        variant="query_small_resnet18",
        segm_ap=0.21,
        bbox_ap=0.23,
        boundary_iou=0.18,
        train_wall_time_sec=44.0,
        eval_wall_time_sec=12.0,
        pred_count_mean=1.9,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.45,
        best_bbox_iou_mean=0.55,
        object_count_mean=1.9,
        split_count_mean=0.2,
        failure_counts={"normal": 3},
    )
    _write_official_run(
        suite_root,
        variant="query_medium_resnet34",
        segm_ap=0.28,
        bbox_ap=0.31,
        boundary_iou=0.24,
        train_wall_time_sec=55.0,
        eval_wall_time_sec=13.0,
        pred_count_mean=2.0,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.52,
        best_bbox_iou_mean=0.63,
        object_count_mean=2.0,
        split_count_mean=0.3,
        failure_counts={"normal": 3},
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_query_alpha_ladder.py",
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

    assert result.returncode == 0


def test_query_alpha_summary_script_includes_deferred_variants_in_official_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "suite"
    output_json = tmp_path / "alpha_summary.json"
    output_md = tmp_path / "alpha_summary.md"
    _write_official_run(
        suite_root,
        variant="query_ref_resnet18",
        segm_ap=0.33,
        bbox_ap=0.35,
        boundary_iou=0.26,
        train_wall_time_sec=58.0,
        eval_wall_time_sec=14.0,
        pred_count_mean=2.1,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.57,
        best_bbox_iou_mean=0.66,
        object_count_mean=2.1,
        split_count_mean=0.35,
        failure_counts={"normal": 2},
        use_reference=True,
        use_graph_rescue=False,
    )
    _write_official_run(
        suite_root,
        variant="query_graph_resnet18",
        segm_ap=0.31,
        bbox_ap=0.33,
        boundary_iou=0.25,
        train_wall_time_sec=57.0,
        eval_wall_time_sec=14.5,
        pred_count_mean=2.05,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.54,
        best_bbox_iou_mean=0.64,
        object_count_mean=2.05,
        split_count_mean=0.33,
        failure_counts={"normal": 2},
        use_reference=False,
        use_graph_rescue=True,
    )
    _write_official_run(
        suite_root,
        variant="query_refgraph_resnet18",
        segm_ap=0.38,
        bbox_ap=0.41,
        boundary_iou=0.31,
        train_wall_time_sec=63.0,
        eval_wall_time_sec=16.0,
        pred_count_mean=2.25,
        gt_count_mean=2.0,
        best_mask_iou_mean=0.62,
        best_bbox_iou_mean=0.71,
        object_count_mean=2.25,
        split_count_mean=0.4,
        failure_counts={"normal": 1},
        use_reference=True,
        use_graph_rescue=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_query_alpha_ladder.py",
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

    assert result.returncode == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert [row["variant"] for row in payload["rows"]] == [
        "query_ref_resnet18",
        "query_graph_resnet18",
        "query_refgraph_resnet18",
    ]
    markdown = output_md.read_text(encoding="utf-8")
    assert "query_ref_resnet18" in markdown
    assert "query_graph_resnet18" in markdown
    assert "query_refgraph_resnet18" in markdown
