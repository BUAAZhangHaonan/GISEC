from __future__ import annotations

import numpy as np

from gisec.engine.runtime import _prepare_overlay_dir, masks_to_results


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
