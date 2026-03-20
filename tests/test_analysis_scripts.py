from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_run_summary(run_dir: Path, *, variant: str, ap: float, ap50: float, fps: float, peak_memory_mb: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant,
        "contract_mode": "compat",
        "checkpoint": str(run_dir / "model_best.pth"),
        "results_json": str(run_dir / "coco_instances_results.json"),
        "dataset_root": "/tmp/dataset",
        "prototype_root": "/tmp/prototypes",
        "split": "val",
        "image_size": 1024,
        "batch": 4,
        "num_workers": 4,
        "min_area": 10,
        "edge_threshold": 0.5,
        "device": "cpu",
        "metrics": {
            "iteration": 20,
            "segm/AP": ap,
            "segm/AP50": ap50,
            "segm/AP75": max(ap - 5.0, 0.0),
            "segm/APs": 0.0,
            "segm/APm": ap,
            "segm/APl": 0.0,
        },
        "inference_speed": {
            "status": "ok",
            "timed_images": 10,
            "latency_ms_mean": 1000.0 / fps,
            "latency_ms_p50": 1000.0 / fps,
            "latency_ms_p90": 1000.0 / fps,
            "throughput_fps": fps,
            "inference_peak_memory_mb": peak_memory_mb,
        },
        "params_trainable": 123456,
        "wall_time_sec": 42,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.cocoeval.json").write_text(json.dumps(payload["metrics"], ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "inference_speed.json").write_text(json.dumps(payload["inference_speed"], ensure_ascii=False) + "\n", encoding="utf-8")


def _write_baseline_run_summary(
    run_dir: Path,
    *,
    model: str,
    variant: str,
    modality: str,
    ap: float,
    ap50: float,
    fps: float,
    peak_memory_mb: float,
    params_trainable: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "variant": variant,
        "modality": modality,
        "artifact_root": str(run_dir),
        "metrics": {
            "iteration": 1,
            "bbox/AP": ap - 1.0,
            "bbox/AP50": ap50 - 1.0,
            "segm/AP": ap,
            "segm/AP50": ap50,
        },
        "inference_speed": {
            "status": "ok",
            "timed_images": 8,
            "latency_ms_mean": 1000.0 / fps,
            "latency_ms_p50": 1000.0 / fps,
            "latency_ms_p90": 1000.0 / fps,
            "throughput_fps": fps,
            "inference_peak_memory_mb": peak_memory_mb,
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
    (run_dir / "wall_time_sec.txt").write_text("17\n", encoding="utf-8")


def test_write_extended_metrics_table(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "suite"
    _write_run_summary(suite_root / "B0", variant="B0", ap=71.1, ap50=90.0, fps=2.5, peak_memory_mb=1200.0)
    _write_run_summary(suite_root / "G5", variant="G5", ap=74.2, ap50=92.5, fps=2.2, peak_memory_mb=1350.0)
    out_path = tmp_path / "extended_metrics_table.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/write_extended_metrics_table.py",
            "--suite-root",
            str(suite_root),
            "--output",
            str(out_path),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    text = out_path.read_text(encoding="utf-8")
    assert "Model" in text
    assert "segm/AP" in text
    assert "throughput_fps" in text
    assert "G5" in text


def test_summarize_suite_reports_best_variant(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "suite"
    _write_run_summary(suite_root / "B0", variant="B0", ap=71.1, ap50=90.0, fps=2.5, peak_memory_mb=1200.0)
    _write_run_summary(suite_root / "G3", variant="G3", ap=72.6, ap50=91.0, fps=2.4, peak_memory_mb=1250.0)
    _write_run_summary(suite_root / "G5", variant="G5", ap=74.2, ap50=92.5, fps=2.2, peak_memory_mb=1350.0)
    out_json = tmp_path / "suite_summary.json"
    out_md = tmp_path / "suite_summary.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_suite.py",
            "--suite-root",
            str(suite_root),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["best_variant"] == "G5"
    assert summary["num_runs"] == 3
    assert "best segm/AP" in out_md.read_text(encoding="utf-8")


def test_summarize_baseline_matrix_writes_markdown_table(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "baselines"
    _write_baseline_run_summary(
        suite_root / "unet_rgb_smoke",
        model="unet",
        variant="rgb_smoke",
        modality="rgb",
        ap=42.5,
        ap50=68.1,
        fps=18.0,
        peak_memory_mb=512.0,
        params_trainable=117041,
    )
    _write_baseline_run_summary(
        suite_root / "unet_depth_geometry_smoke",
        model="unet",
        variant="depth_geometry_smoke",
        modality="rgbd",
        ap=58.4,
        ap50=80.2,
        fps=16.2,
        peak_memory_mb=640.0,
        params_trainable=117553,
    )
    out_path = tmp_path / "baseline_matrix.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_baseline_matrix.py",
            "--input-root",
            str(suite_root),
            "--output",
            str(out_path),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    text = out_path.read_text(encoding="utf-8")
    assert "Baseline Benchmark Matrix" in text
    assert "depth_geometry_smoke" in text
    assert "58.4000" in text
    assert "rgbd" in text
