from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def _write_cache_sample(
    cache_dir: Path,
    *,
    image_id: int,
    gt_count: int,
    fragment_gt_instances: list[int],
    purities: list[float],
    same_instance_pairs_total: int,
    same_instance_pairs_covered: int,
    positive_edge_count: int,
    valid_edge_count: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "gt_count": gt_count,
            "same_instance_pairs_total": same_instance_pairs_total,
            "same_instance_pairs_covered": same_instance_pairs_covered,
            "positive_edge_count": positive_edge_count,
            "valid_edge_count": valid_edge_count,
        },
        "fragment_stats": [
            {"gt_instance": int(gt), "purity": float(purity)}
            for gt, purity in zip(fragment_gt_instances, purities)
        ],
    }
    torch.save(payload, cache_dir / f"{image_id:06d}.pt")


def test_graph_cache_prerequisites_summary_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"
    output_chart = tmp_path / "summary.png"

    _write_cache_sample(
        cache_a,
        image_id=1,
        gt_count=4,
        fragment_gt_instances=[1, 1, 2, 3],
        purities=[1.0, 1.0, 0.8, 0.9],
        same_instance_pairs_total=1,
        same_instance_pairs_covered=1,
        positive_edge_count=1,
        valid_edge_count=4,
    )
    _write_cache_sample(
        cache_b,
        image_id=1,
        gt_count=4,
        fragment_gt_instances=[1, 2, 3],
        purities=[1.0, 0.7, 0.8],
        same_instance_pairs_total=0,
        same_instance_pairs_covered=0,
        positive_edge_count=0,
        valid_edge_count=3,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_graph_cache_prerequisites.py",
            "--cache-dir",
            str(cache_a),
            "--label",
            "A",
            "--cache-dir",
            str(cache_b),
            "--label",
            "B",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--output-chart",
            str(output_chart),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    rows = {row["label"]: row for row in payload["rows"]}
    assert rows["A"]["covered_gt_rate"] == 0.75
    assert rows["A"]["split_gt_rate"] == 0.25
    assert rows["A"]["singleton_gt_rate"] == 0.5
    assert rows["A"]["same_instance_recall"] == 1.0
    assert rows["B"]["covered_gt_rate"] == 0.75
    assert rows["B"]["split_gt_rate"] == 0.0
    assert rows["B"]["impure_fragment_rate"] == 2.0 / 3.0
    assert output_md.exists()
    assert output_chart.exists()
