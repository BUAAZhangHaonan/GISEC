from __future__ import annotations

import numpy as np

from gisec.eval.coco_export import masks_to_coco_results


def test_masks_to_coco_results_encodes_basic_instance_records() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 5:13] = 1

    results = masks_to_coco_results(
        image_id=7,
        masks=[mask],
        scores=[0.85],
        category_id=1,
    )

    assert len(results) == 1
    assert results[0]["image_id"] == 7
    assert results[0]["category_id"] == 1
    assert results[0]["score"] == 0.85
    assert results[0]["bbox"] == [5, 4, 8, 8]

    zero_id_results = masks_to_coco_results(
        image_id=7,
        masks=[mask],
        scores=[0.85],
        category_id=0,
    )

    assert zero_id_results[0]["category_id"] == 0
