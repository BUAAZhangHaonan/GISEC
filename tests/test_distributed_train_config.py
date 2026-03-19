from __future__ import annotations

from gisec.train.train_gisec import parse_train_args, resolve_distributed_context


def test_parse_train_args_accepts_torchrun_launcher() -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--prototype-root",
            "/tmp/prototype",
            "--output-dir",
            "/tmp/out",
            "--launcher",
            "torchrun",
            "--nproc-per-node",
            "6",
            "--master-port",
            "29610",
        ]
    )

    assert args.launcher == "torchrun"
    assert args.nproc_per_node == 6
    assert args.master_port == 29610


def test_resolve_distributed_context_reads_torchrun_env(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "6")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    context = resolve_distributed_context(
        launcher="torchrun",
        device_name="cuda",
        dist_backend="nccl",
    )

    assert context.enabled
    assert context.world_size == 6
    assert context.rank == 2
    assert context.local_rank == 2
    assert context.backend == "nccl"
