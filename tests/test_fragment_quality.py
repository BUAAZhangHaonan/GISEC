from __future__ import annotations

import numpy as np

from baseline.common.fragment_quality import (
    build_fragment_pair_records,
    build_fragment_records,
    summarize_fragment_quality,
)


def test_fragment_quality_reports_purity_and_same_instance_recall() -> None:
    label_map = np.zeros((12, 12), dtype=np.int32)
    label_map[2:8, 1:4] = 1
    label_map[2:8, 4:7] = 2
    label_map[2:8, 8:11] = 3

    instance_map = np.zeros((12, 12), dtype=np.int64)
    instance_map[2:8, 1:7] = 1
    instance_map[2:8, 8:11] = 2

    fragment_records = build_fragment_records(label_map, instance_map)
    pair_records = build_fragment_pair_records(label_map, fragment_records, max_gap=4.0)
    summary = summarize_fragment_quality(fragment_records, pair_records)

    assert [int(row["gt_instance"]) for row in fragment_records] == [1, 1, 2]
    assert [float(row["purity"]) for row in fragment_records] == [1.0, 1.0, 1.0]
    positive_pairs = [row for row in pair_records if int(row["gt_same_instance"]) == 1]
    assert len(positive_pairs) == 1
    assert summary["fragment_count"] == 3
    assert summary["fragment_purity_mean"] == 1.0
    assert summary["fragment_purity_median"] == 1.0
    assert summary["same_instance_total_pairs"] == 1
    assert summary["same_instance_recalled_pairs"] == 1
    assert summary["same_instance_recall"] == 1.0


def test_fragment_quality_drops_same_instance_recall_for_far_apart_fragments() -> None:
    label_map = np.zeros((16, 16), dtype=np.int32)
    label_map[2:5, 2:5] = 1
    label_map[10:13, 10:13] = 2

    instance_map = np.zeros((16, 16), dtype=np.int64)
    instance_map[2:5, 2:5] = 3
    instance_map[10:13, 10:13] = 3

    fragment_records = build_fragment_records(label_map, instance_map)
    pair_records = build_fragment_pair_records(label_map, fragment_records, max_gap=2.0)
    summary = summarize_fragment_quality(fragment_records, pair_records)

    assert summary["same_instance_total_pairs"] == 1
    assert summary["same_instance_recalled_pairs"] == 0
    assert summary["same_instance_recall"] == 0.0
