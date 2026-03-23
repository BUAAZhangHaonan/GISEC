from __future__ import annotations

import numpy as np

from gisec.train.v3_targets import (
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


def test_v3_core_heatmap_target_prefers_center_of_elongated_plateau_not_plateau_edge() -> None:
    instance_map = np.zeros((15, 15), dtype=np.int32)
    instance_map[6:9, 2:13] = 1

    core = build_core_heatmap_target(instance_map)
    peak_y, peak_x = np.unravel_index(int(core.argmax()), core.shape)

    assert peak_y == 7
    assert 6 <= peak_x <= 8


def test_v3_core_heatmap_target_scales_support_with_image_resolution() -> None:
    base = np.zeros((256, 256), dtype=np.int32)
    base[80:176, 96:160] = 1
    base[96:152, 176:224] = 2

    scaled = np.zeros((1024, 1024), dtype=np.int32)
    scaled[320:704, 384:640] = 1
    scaled[384:608, 704:896] = 2

    base_core = build_core_heatmap_target(base)
    scaled_core = build_core_heatmap_target(scaled)
    base_ratio = float((base_core >= 0.5).mean())
    scaled_ratio = float((scaled_core >= 0.5).mean())

    assert base_ratio > 0.0
    assert scaled_ratio > 0.0
    assert 0.5 <= scaled_ratio / base_ratio <= 2.0
