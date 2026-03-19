from __future__ import annotations

from gisec.train.train_gisec import parse_train_args


def test_parse_train_args_accepts_torchrun_launch_flags() -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--prototype-root",
            "/tmp/prototypes",
            "--output-dir",
            "/tmp/out",
            "--launcher",
            "torchrun",
            "--local-rank",
            "2",
            "--dist-backend",
            "nccl",
            "--dist-url",
            "env://",
        ]
    )

    assert args.launcher == "torchrun"
    assert args.local_rank == 2
    assert args.dist_backend == "nccl"
    assert args.dist_url == "env://"
