from __future__ import annotations

import json

from gisec.datasets.reference_bank import (
    _pose_farthest_sample_view_ids,
    _sample_view_ids,
    _uniform_sample_view_ids,
)


def _write_pose(path, position: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"position": position}), encoding="utf-8")


def _write_poses(root, view_ids: list[str], positions: dict[str, list[float]]) -> None:
    for view_id in view_ids:
        _write_pose(root / "camera" / f"{view_id}.json", positions[view_id])


def test_pose_farthest_picks_the_farthest_views(tmp_path) -> None:
    view_ids = [f"v{index}" for index in range(8)]
    _write_poses(
        tmp_path,
        view_ids,
        {view_id: [float(index), 0.0, 0.0] for index, view_id in enumerate(view_ids)},
    )

    sampled = _sample_view_ids(
        root=tmp_path, view_ids=view_ids, max_views=3, view_sampler="pose_farthest"
    )

    assert sampled == ["v0", "v3", "v7"]


def test_pose_farthest_falls_back_to_uniform_on_nan_pose(tmp_path) -> None:
    view_ids = [f"v{index}" for index in range(8)]
    positions = {
        view_id: [float(index), 0.0, 0.0] for index, view_id in enumerate(view_ids)
    }
    positions["v3"] = [float("nan"), 0.0, 0.0]
    _write_poses(tmp_path, view_ids, positions)

    # The NaN used to make the sampler return the single seed view
    # silently; it must degrade to the uniform fallback instead.
    assert _pose_farthest_sample_view_ids(tmp_path, view_ids, 3) == []
    assert _sample_view_ids(
        root=tmp_path,
        view_ids=view_ids,
        max_views=3,
        view_sampler="pose_farthest",
    ) == _uniform_sample_view_ids(view_ids, 3)
