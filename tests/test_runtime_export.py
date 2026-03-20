from __future__ import annotations

import numpy as np

from gisec.engine.runtime import (
    _classify_mask_failure,
    _prepare_overlay_dir,
    _summarize_reference_routing,
    masks_to_results,
)


def test_masks_to_results_derives_non_uniform_scores_within_unit_interval() -> None:
    mask_a = np.zeros((16, 16), dtype=np.uint8)
    mask_a[2:8, 2:8] = 1
    mask_b = np.zeros((16, 16), dtype=np.uint8)
    mask_b[8:14, 8:14] = 1

    results = masks_to_results(
        image_id=7,
        masks=[mask_a, mask_b],
        fg_scores=[0.95, 0.60],
        boundary_scores=[0.05, 0.45],
        merge_scores=[0.90, 0.55],
    )

    assert len(results) == 2
    scores = [float(item["score"]) for item in results]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert len(set(scores)) == 2
    assert scores[0] > scores[1]


def test_prepare_overlay_dir_removes_stale_pngs(tmp_path) -> None:
    overlay_dir = tmp_path / "visualizations" / "overlay"
    overlay_dir.mkdir(parents=True)
    (overlay_dir / "stale_a.png").write_bytes(b"png")
    (overlay_dir / "stale_b.png").write_bytes(b"png")

    _prepare_overlay_dir(overlay_dir)

    assert list(overlay_dir.glob("*.png")) == []


def test_classify_mask_failure_detects_empty_tiny_full_and_normal() -> None:
    empty = _classify_mask_failure([], image_shape=(16, 16), min_area=4)

    tiny_mask = np.zeros((16, 16), dtype=np.uint8)
    tiny_mask[0:2, 0:2] = 1
    tiny = _classify_mask_failure([tiny_mask], image_shape=(16, 16), min_area=16)

    full_mask = np.ones((16, 16), dtype=np.uint8)
    full = _classify_mask_failure([full_mask], image_shape=(16, 16), min_area=4)

    normal_mask = np.zeros((16, 16), dtype=np.uint8)
    normal_mask[4:12, 4:12] = 1
    normal = _classify_mask_failure([normal_mask], image_shape=(16, 16), min_area=4)

    assert empty == "empty"
    assert tiny == "tiny"
    assert full == "full"
    assert normal == "normal"


def test_summarize_reference_routing_counts_selected_views() -> None:
    summary = _summarize_reference_routing(
        [
            {
                "prototype_slot_count": 6,
                "prototype_topk": 2,
                "selected_view_ids": ["view_001", "view_003"],
            },
            {
                "prototype_slot_count": 6,
                "prototype_topk": 2,
                "selected_view_ids": ["view_003", "view_005"],
            },
        ]
    )

    assert summary["total_images"] == 2
    assert summary["prototype_slot_count"] == 6
    assert summary["prototype_topk"] == 2
    assert summary["selected_view_histogram"]["view_003"] == 2
