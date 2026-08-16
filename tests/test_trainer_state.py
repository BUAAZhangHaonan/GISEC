from __future__ import annotations

import json
import os
import subprocess

import pytest

from gisec.train.trainer import (
    _acquire_run_lock,
    _drop_stale_metrics_rows,
    _release_run_lock,
)


def _write_rows(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_rows(path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_drop_stale_metrics_rows_removes_partial_epoch_rows(tmp_path) -> None:
    log = tmp_path / "metrics_log.jsonl"
    _write_rows(log, [
        {"mode": "train_step", "epoch": 1, "global_step": 1},
        {"mode": "epoch_train", "epoch": 1, "global_step": 2},
        {"mode": "epoch_eval", "epoch": 1, "metric": 0.5},
        {"mode": "run_resume", "epoch": 1},
        {"mode": "train_step", "epoch": 2, "global_step": 3},
        {"mode": "train_step", "epoch": 2, "global_step": 4},
    ])

    dropped = _drop_stale_metrics_rows(log, completed_epoch=1)

    assert dropped == 2
    assert [row["mode"] for row in _read_rows(log)] == [
        "train_step", "epoch_train", "epoch_eval", "run_resume",
    ]


def test_drop_stale_metrics_rows_keeps_clean_history(tmp_path) -> None:
    log = tmp_path / "metrics_log.jsonl"
    _write_rows(log, [
        {"mode": "epoch_eval", "epoch": 1, "metric": 0.5},
        {"mode": "run_final", "best_metric": 0.5},
    ])

    dropped = _drop_stale_metrics_rows(log, completed_epoch=1)

    assert dropped == 0
    assert [row["mode"] for row in _read_rows(log)] == [
        "epoch_eval", "run_final",
    ]


def test_drop_stale_metrics_rows_removes_half_written_line(tmp_path) -> None:
    log = tmp_path / "metrics_log.jsonl"
    log.write_text(
        '{"mode": "epoch_train", "epoch": 1}\n{"mode": "train_step", "ep',
        encoding="utf-8",
    )

    dropped = _drop_stale_metrics_rows(log, completed_epoch=1)

    assert dropped == 1
    assert [row["mode"] for row in _read_rows(log)] == ["epoch_train"]


def test_drop_stale_metrics_rows_tolerates_missing_log(tmp_path) -> None:
    assert _drop_stale_metrics_rows(tmp_path / "absent.jsonl", 3) == 0


def test_run_lock_blocks_a_second_live_launch(tmp_path) -> None:
    output_dir = tmp_path / "run"
    _acquire_run_lock(output_dir)

    with pytest.raises(RuntimeError, match="already in use"):
        _acquire_run_lock(output_dir)

    _release_run_lock(output_dir)
    _acquire_run_lock(output_dir)

    assert (output_dir / ".run_lock").read_text(
        encoding="utf-8").strip() == str(os.getpid())


def test_run_lock_overrides_a_stale_lock(tmp_path) -> None:
    stale = subprocess.Popen(["sleep", "0"])
    stale.wait()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / ".run_lock").write_text(
        f"{stale.pid}\n", encoding="utf-8")

    _acquire_run_lock(output_dir)

    assert (output_dir / ".run_lock").read_text(
        encoding="utf-8").strip() == str(os.getpid())
