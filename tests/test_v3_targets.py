from __future__ import annotations

import numpy as np

from gisec_v3.train.targets import (
    build_boundary_target,
    build_core_heatmap_target,
    build_fg_target,
    build_ownership_target,
)


def test_v3_target_builders_emit_fg_boundary_core_and_ownership() -> None:
    instance_map = np.zeros((11, 11), dtype=np.int32)
    instance_map[2:9, 3:8] = 1

    fg = build_fg_target(instance_map)
    boundary = build_boundary_target(instance_map == 1)
    core = build_core_heatmap_target(instance_map)
    ownership = build_ownership_target(instance_map)

    assert fg.shape == (11, 11)
    assert boundary.shape == (11, 11)
    assert core.shape == (11, 11)
    assert ownership.shape == (2, 11, 11)
    assert fg[5, 5] == 1.0
    assert boundary.sum() > 0
    assert float(core.max()) == 1.0
    peak_y, peak_x = np.unravel_index(int(core.argmax()), core.shape)
    assert ownership[:, peak_y, peak_x].tolist() == [0.0, 0.0]


def test_v3_core_heatmap_target_stays_inside_elongated_object() -> None:
    instance_map = np.zeros((15, 15), dtype=np.int32)
    instance_map[6:9, 2:13] = 1

    core = build_core_heatmap_target(instance_map)
    peak_y, peak_x = np.unravel_index(int(core.argmax()), core.shape)

    assert instance_map[peak_y, peak_x] == 1
    assert core[peak_y, peak_x] == 1.0
