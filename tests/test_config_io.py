from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from argparse import Namespace

from gisec.train.train_gisec import (
    parse_eval_args,
    parse_infer_args,
    parse_train_args,
    resolve_model_config,
)
from gisec.config.query_models import get_query_model_spec


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


def test_parse_train_args_reads_model_defaults(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "model.yaml",
        {
            "common": {
                "dataset_root": "/tmp/dataset",
                "prototype_root": "/tmp/prototypes",
                "output_dir": "/tmp/out",
                "base_channels": 12,
                "graph_hidden_dim": 48,
                "norm_layer": "group",
                "prototype_slot_count": 5,
                "prototype_topk": 1,
                "fg_prior": 0.1,
                "boundary_prior": 0.02,
                "reference_conditioning_mode": "bottleneck_only",
                "reference_routing_mode": "hard_top1",
                "reference_skip_margin": 0.2,
            }
        },
    )

    args = parse_train_args(["--config", str(config_path)])

    assert args.base_channels == 12
    assert args.graph_hidden_dim == 48
    assert args.norm_layer == "group"
    assert args.prototype_slot_count == 5
    assert args.prototype_topk == 1
    assert args.fg_prior == 0.1
    assert args.boundary_prior == 0.02
    assert args.reference_conditioning_mode == "bottleneck_only"
    assert args.reference_routing_mode == "hard_top1"
    assert args.reference_skip_margin == 0.2


def test_parse_train_args_normalizes_unquoted_off_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "q0.yaml"
    config_path.write_text(
        "\n".join(
            [
                "common:",
                "  dataset_root: /tmp/dataset",
                "  prototype_root: /tmp/prototypes",
                "  output_dir: /tmp/out",
                "  reference_conditioning_mode: off",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    args = parse_train_args(["--config", str(config_path)])

    assert args.reference_conditioning_mode == "off"


def test_resolve_model_config_normalizes_legacy_false_reference_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "model_config.json").write_text(
        json.dumps(
            {
                "reference_conditioning_mode": "False",
                "reference_routing_mode": "hard_top1",
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(**vars(parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--prototype-root",
            "/tmp/prototypes",
            "--output-dir",
            "/tmp/out",
        ]
    )))

    checkpoint_path = output_dir / "model_best.pth"
    checkpoint_path.write_text("", encoding="utf-8")
    config = resolve_model_config(args, checkpoint_path=checkpoint_path, output_dir=output_dir)

    assert config["reference_conditioning_mode"] == "off"


def test_parse_train_args_reads_fragment_threshold_defaults(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "thresholds.yaml",
        {
            "common": {
                "dataset_root": "/tmp/dataset",
                "prototype_root": "/tmp/prototypes",
                "output_dir": "/tmp/out",
                "fragment_fg_threshold": 0.6,
                "fragment_boundary_threshold": 0.75,
            }
        },
    )

    args = parse_train_args(["--config", str(config_path)])

    assert args.fragment_fg_threshold == 0.6
    assert args.fragment_boundary_threshold == 0.75


def test_parse_train_args_reads_seed_default(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "seed.yaml",
        {
            "common": {
                "dataset_root": "/tmp/dataset",
                "prototype_root": "/tmp/prototypes",
                "output_dir": "/tmp/out",
            },
            "train": {
                "seed": 123,
            },
        },
    )

    args = parse_train_args(["--config", str(config_path)])

    assert args.seed == 123


def test_parse_train_args_allows_smoke_config_to_override_reference_policy(tmp_path: Path) -> None:
    reference_path = _write_yaml(
        tmp_path / "reference.yaml",
        {
            "common": {
                "dataset_root": "/tmp/dataset",
                "prototype_root": "/tmp/prototypes",
                "output_dir": "/tmp/out",
                "reference_max_views": 16,
                "reference_view_sampler": "pose_farthest",
                "prototype_slot_count": 6,
                "prototype_topk": 2,
                "reference_conditioning_mode": "full",
                "reference_routing_mode": "soft_topk",
                "reference_skip_margin": 0.05,
            }
        },
    )
    smoke_path = _write_yaml(
        tmp_path / "smoke.yaml",
        {
            "common": {
                "reference_max_views": 6,
                "prototype_slot_count": 4,
                "prototype_topk": 1,
                "reference_conditioning_mode": "bottleneck_only",
                "reference_routing_mode": "hard_top1",
                "reference_skip_margin": 0.15,
            }
        },
    )

    args = parse_train_args(
        [
            "--config",
            str(reference_path),
            "--config",
            str(smoke_path),
        ]
    )

    assert args.reference_max_views == 6
    assert args.reference_view_sampler == "pose_farthest"
    assert args.prototype_slot_count == 4
    assert args.prototype_topk == 1
    assert args.reference_conditioning_mode == "bottleneck_only"
    assert args.reference_routing_mode == "hard_top1"
    assert args.reference_skip_margin == 0.15


def test_parse_train_args_reads_graph_warmup_and_reweighted_boundary_defaults(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {
            "common": {
                "dataset_root": "/tmp/dataset",
                "prototype_root": "/tmp/prototypes",
                "output_dir": "/tmp/out",
            },
            "train": {
                "graph_warmup_steps": 24,
                "fg_pos_weight": 11.0,
                "boundary_pos_weight": 12.0,
            },
        },
    )

    args = parse_train_args(["--config", str(config_path)])

    assert args.graph_warmup_steps == 24
    assert args.fg_pos_weight == 11.0
    assert args.boundary_pos_weight == 12.0


def test_parse_train_args_accepts_new_recovery_cli_flags() -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--prototype-root",
            "/tmp/prototypes",
            "--output-dir",
            "/tmp/out",
            "--fg-prior",
            "0.11",
            "--boundary-prior",
            "0.03",
            "--graph-warmup-steps",
            "12",
            "--reference-conditioning-mode",
            "bottleneck_only",
            "--reference-routing-mode",
            "hard_top1",
            "--reference-skip-margin",
            "0.2",
        ]
    )

    assert args.fg_prior == 0.11
    assert args.boundary_prior == 0.03
    assert args.graph_warmup_steps == 12
    assert args.reference_conditioning_mode == "bottleneck_only"
    assert args.reference_routing_mode == "hard_top1"
    assert args.reference_skip_margin == 0.2


def test_query_model_registry_reserves_uq_scales_and_keeps_legacy_names_out() -> None:
    uq_s = get_query_model_spec("UQ-s")
    uq_m = get_query_model_spec("UQ-m")

    assert uq_s.model_family == "UQ"
    assert uq_m.model_family == "UQ"
    assert uq_s.encoder_family == "resnet"
    assert uq_m.encoder_family == "resnet"
    assert uq_s.depth_fusion_mode == "early6"
    assert uq_m.depth_fusion_mode == "early6"
    assert uq_s.model_scale == "s"
    assert uq_m.model_scale == "m"

    with pytest.raises(ValueError):
        get_query_model_spec("A1")
