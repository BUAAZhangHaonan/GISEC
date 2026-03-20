from __future__ import annotations

from pathlib import Path

import yaml

from gisec.train.train_gisec import parse_eval_args, parse_infer_args, parse_train_args


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_parse_train_args_reads_yaml_defaults(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {
            "common": {
                "dataset_root": "/tmp/dataset",
                "prototype_root": "/tmp/prototypes",
                "output_dir": "/tmp/out",
                "variant": "A1",
                "image_size": 512,
                "batch": 2,
            },
            "train": {
                "epochs": 3,
                "lr": 2.0e-4,
                "max_train_steps": 5,
            },
        },
    )

    args = parse_train_args(["--config", str(config_path)])

    assert args.dataset_root == "/tmp/dataset"
    assert args.prototype_root == "/tmp/prototypes"
    assert args.output_dir == "/tmp/out"
    assert args.variant == "A1"
    assert args.image_size == 512
    assert args.batch == 2
    assert args.epochs == 3
    assert args.lr == 2.0e-4
    assert args.max_train_steps == 5


def test_cli_overrides_yaml_defaults(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {
            "common": {
                "dataset_root": "/tmp/dataset",
                "prototype_root": "/tmp/prototypes",
                "output_dir": "/tmp/out",
                "variant": "A0",
                "batch": 4,
            }
        },
    )

    args = parse_train_args(
        [
            "--config",
            str(config_path),
            "--variant",
            "A1",
            "--batch",
            "1",
        ]
    )

    assert args.variant == "A1"
    assert args.batch == 1


def test_parse_eval_and_infer_args_read_mode_specific_yaml_sections(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "eval.yaml",
        {
            "common": {
                "dataset_root": "/tmp/dataset",
                "prototype_root": "/tmp/prototypes",
                "output_dir": "/tmp/out",
                "variant": "A1",
            },
            "eval": {
                "checkpoint": "/tmp/out/model_best.pth",
                "split": "val",
                "max_images": 11,
            },
            "infer": {
                "checkpoint": "/tmp/out/model_final.pth",
                "split": "test",
                "max_images": 7,
            },
        },
    )

    eval_args = parse_eval_args(["--config", str(config_path)])
    infer_args = parse_infer_args(["--config", str(config_path)])

    assert eval_args.checkpoint == "/tmp/out/model_best.pth"
    assert eval_args.split == "val"
    assert eval_args.max_images == 11
    assert infer_args.checkpoint == "/tmp/out/model_final.pth"
    assert infer_args.split == "test"
    assert infer_args.max_images == 7


def test_multiple_configs_merge_with_later_override(tmp_path: Path) -> None:
    base_path = _write_yaml(
        tmp_path / "base.yaml",
        {
            "common": {
                "dataset_root": "/tmp/dataset",
                "prototype_root": "/tmp/prototypes",
                "output_dir": "/tmp/out",
                "variant": "A0",
                "batch": 4,
            },
            "train": {"epochs": 20},
        },
    )
    smoke_path = _write_yaml(
        tmp_path / "smoke.yaml",
        {
            "common": {"batch": 1, "variant": "A1"},
            "train": {"epochs": 1, "max_train_steps": 8},
        },
    )

    args = parse_train_args(
        [
            "--config",
            str(base_path),
            "--config",
            str(smoke_path),
        ]
    )

    assert args.batch == 1
    assert args.variant == "A1"
    assert args.epochs == 1
    assert args.max_train_steps == 8
