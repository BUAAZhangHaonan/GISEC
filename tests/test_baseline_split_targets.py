from __future__ import annotations

import numpy as np
import pytest

from baseline.common.instance_targets import build_instance_target_pack


def test_instance_target_pack_builds_non_empty_core_for_regular_instance() -> None:
    instance_map = np.zeros((32, 32), dtype=np.int64)
    instance_map[8:24, 8:24] = 1

    targets = build_instance_target_pack(instance_map)

    assert targets["fg"].shape == (1, 32, 32)
    assert float(targets["fg"].sum()) > 0.0
    assert float(targets["fg"].sum()) < float((instance_map > 0).sum())


def test_instance_target_pack_keeps_non_empty_core_for_small_instance() -> None:
    instance_map = np.zeros((16, 16), dtype=np.int64)
    instance_map[6:8, 6:8] = 1

    targets = build_instance_target_pack(instance_map)

    assert float(targets["fg"].sum()) >= 1.0
    assert float(targets["center"].max()) > 0.0


def test_instance_target_pack_builds_thick_boundary_band() -> None:
    instance_map = np.zeros((48, 48), dtype=np.int64)
    instance_map[10:22, 8:20] = 1
    instance_map[10:22, 23:35] = 2

    targets = build_instance_target_pack(instance_map)

    seam_band = targets["boundary"][0, 9:23, 18:25]
    assert float(seam_band.mean()) > 0.5
    assert int(targets["boundary"].sum()) > 100


def test_instance_target_pack_supports_configured_target_params() -> None:
    instance_map = np.zeros((32, 32), dtype=np.int64)
    instance_map[8:24, 8:24] = 1

    default_targets = build_instance_target_pack(instance_map)
    thin_targets = build_instance_target_pack(instance_map, core_erosion_px=1, boundary_band_px=3)

    assert float(thin_targets["fg"].sum()) > float(default_targets["fg"].sum())
    assert float(thin_targets["boundary"].sum()) < float(default_targets["boundary"].sum())
