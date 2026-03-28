from __future__ import annotations

import pytest

from gisec.train.train_active import parse_train_args


def test_active_train_cli_accepts_valid_mask_depth_override_for_base_rgbd(tmp_path) -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--variant",
            "base_rgbd_1024",
            "--depth-mode",
            "rgbd_concat_valid_mask",
        ]
    )

    assert args.depth_mode == "rgbd_concat_valid_mask"


def test_active_train_cli_reads_valid_mask_depth_mode_from_config_stack(tmp_path) -> None:
    config_path = tmp_path / "active_depth.yaml"
    config_path.write_text(
        """
model:
  variant: base_rgbd_1024
  depth_mode: rgbd_concat_valid_mask
""".strip(),
        encoding="utf-8",
    )

    args = parse_train_args(
        [
            "--config",
            str(config_path),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.variant == "base_rgbd_1024"
    assert args.depth_mode == "rgbd_concat_valid_mask"


def test_active_train_cli_rejects_depth_override_for_base_rgb(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "out"),
                "--variant",
                "base_rgb_1024",
                "--depth-mode",
                "rgbd_concat_valid_mask",
            ]
        )
