from __future__ import annotations

import numpy as np

from gisec.engine.runtime import (
    _component_merge_score,
    _classify_mask_failure,
    _summarize_instance_matching,
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


def test_classify_mask_failure_detects_empty_tiny_border_strip_full_frame_and_normal() -> None:
    empty = _classify_mask_failure([], image_shape=(16, 16), min_area=4)

    tiny_mask = np.zeros((16, 16), dtype=np.uint8)
    tiny_mask[0:2, 0:2] = 1
    tiny = _classify_mask_failure([tiny_mask], image_shape=(16, 16), min_area=16)

    border_strip_mask = np.zeros((16, 16), dtype=np.uint8)
    border_strip_mask[:, 0:2] = 1
    border_strip = _classify_mask_failure([border_strip_mask], image_shape=(16, 16), min_area=4)

    full_mask = np.ones((16, 16), dtype=np.uint8)
    full = _classify_mask_failure([full_mask], image_shape=(16, 16), min_area=4)

    normal_mask = np.zeros((16, 16), dtype=np.uint8)
    normal_mask[4:12, 4:12] = 1
    normal = _classify_mask_failure([normal_mask], image_shape=(16, 16), min_area=4)

    assert empty == "empty"
    assert tiny == "tiny_island"
    assert border_strip == "border_strip"
    assert full == "full_frame"
    assert normal == "normal"


def test_classify_mask_failure_does_not_mark_valid_1024_scale_component_as_tiny_island() -> None:
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[256:296, 256:296] = 1

    label = _classify_mask_failure([mask], image_shape=(1024, 1024), min_area=256)

    assert label == "normal"


def test_component_merge_score_returns_zero_for_components_without_accepted_edges() -> None:
    merged_mask = np.zeros((8, 8), dtype=bool)
    merged_mask[2:6, 2:6] = True
    fragments = np.zeros((8, 8), dtype=np.int32)
    fragments[2:6, 2:6] = 1

    score = _component_merge_score(
        merged_mask=merged_mask,
        fragments=fragments,
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_scores=np.zeros((0,), dtype=np.float32),
        threshold=0.5,
    )

    assert score == 0.0


def test_summarize_reference_routing_counts_selected_views() -> None:
    summary = _summarize_reference_routing(
        [
            {
                "reference_conditioning_mode": "full",
                "reference_routing_mode": "hard_top1",
                "prototype_slot_count": 6,
                "prototype_topk": 2,
                "top1_weight": [0.9],
                "top2_weight": [0.1],
                "top1_top2_margin": [0.8],
                "routing_entropy": [0.2],
                "skip_conditioning": [False],
                "selected_view_ids": ["view_001", "view_003"],
            },
            {
                "reference_conditioning_mode": "full",
                "reference_routing_mode": "hard_top1",
                "prototype_slot_count": 6,
                "prototype_topk": 2,
                "top1_weight": [0.8],
                "top2_weight": [0.2],
                "top1_top2_margin": [0.6],
                "routing_entropy": [0.3],
                "skip_conditioning": [True],
                "selected_view_ids": ["view_003", "view_005"],
            },
        ]
    )

    assert summary["total_images"] == 2
    assert summary["reference_conditioning_mode"] == "full"
    assert summary["reference_routing_mode"] == "hard_top1"
    assert summary["prototype_slot_count"] == 6
    assert summary["prototype_topk"] == 2
    assert summary["top1_weight_mean"] == 0.85
    assert summary["skip_conditioning_ratio"] == 0.5
    assert summary["selected_view_histogram"]["view_003"] == 2


def test_summarize_instance_matching_reports_gt_pred_count_and_iou() -> None:
    gt_a = np.zeros((16, 16), dtype=np.uint8)
    gt_a[2:8, 2:8] = 1
    gt_b = np.zeros((16, 16), dtype=np.uint8)
    gt_b[8:14, 8:14] = 1
    pred = np.zeros((16, 16), dtype=np.uint8)
    pred[2:8, 2:8] = 1

    row = _summarize_instance_matching(
        image_id=3,
        file_name="toy.png",
        gt_masks=[gt_a, gt_b],
        pred_masks=[pred],
    )

    assert row["image_id"] == 3
    assert row["gt_count"] == 2
    assert row["pred_count"] == 1
    assert row["best_bbox_iou_mean"] == 1.0
    assert row["best_mask_iou_mean"] == 1.0
