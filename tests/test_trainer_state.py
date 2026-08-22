from __future__ import annotations

import json
import os
import subprocess
import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.amp import GradScaler

from gisec.config.variants import get_gisec_variant_spec
from gisec.train import trainer as trainer_module
from gisec.train.args import parse_train_args
from gisec.train.model_builder import resume_payload, save_torch_payload
from gisec.train.trainer import (
    _acquire_run_lock,
    _drop_stale_metrics_rows,
    _release_run_lock,
    _run_training_loop,
    _TrainingRun,
)


def _write_rows(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_drop_stale_metrics_rows_removes_partial_epoch_rows(tmp_path) -> None:
    log = tmp_path / "metrics_log.jsonl"
    _write_rows(
        log,
        [
            {"mode": "train_step", "epoch": 1, "global_step": 1},
            {"mode": "epoch_train", "epoch": 1, "global_step": 2},
            {"mode": "epoch_eval", "epoch": 1, "metric": 0.5},
            {"mode": "run_resume", "epoch": 1},
            {"mode": "train_step", "epoch": 2, "global_step": 3},
            {"mode": "train_step", "epoch": 2, "global_step": 4},
        ],
    )

    dropped = _drop_stale_metrics_rows(log, completed_epoch=1)

    assert dropped == 2
    assert [row["mode"] for row in _read_rows(log)] == [
        "train_step",
        "epoch_train",
        "epoch_eval",
        "run_resume",
    ]


def test_drop_stale_metrics_rows_keeps_clean_history(tmp_path) -> None:
    log = tmp_path / "metrics_log.jsonl"
    _write_rows(
        log,
        [
            {"mode": "epoch_eval", "epoch": 1, "metric": 0.5},
            {"mode": "run_final", "best_metric": 0.5},
        ],
    )

    dropped = _drop_stale_metrics_rows(log, completed_epoch=1)

    assert dropped == 0
    assert [row["mode"] for row in _read_rows(log)] == [
        "epoch_eval",
        "run_final",
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

    assert (output_dir / ".run_lock").read_text(encoding="utf-8").strip() == str(
        os.getpid()
    )


def test_run_lock_overrides_a_stale_lock(tmp_path) -> None:
    stale = subprocess.Popen(["sleep", "0"])
    stale.wait()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / ".run_lock").write_text(f"{stale.pid}\n", encoding="utf-8")

    _acquire_run_lock(output_dir)

    assert (output_dir / ".run_lock").read_text(encoding="utf-8").strip() == str(
        os.getpid()
    )


class _StubBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 4, kernel_size=1)

    def forward(
        self,
        pixel_values,
        pixel_mask,
        output_hidden_states,
        mask_labels=None,
        class_labels=None,
    ):
        return SimpleNamespace(loss=self.proj(pixel_values).sum(), loss_dict=None)


class _LoopModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _StubBackbone()


def _stub_loop_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        trainer_module,
        "train_local_modules_with_metrics",
        lambda **kwargs: (torch.tensor(0.0), {"loss_local_total": 0.0}),
    )
    monkeypatch.setattr(
        trainer_module,
        "evaluate_gisec",
        lambda **kwargs: ({"segm/AP": 0.25}, {}),
    )


def _loop_run(args, model: _LoopModel, tmp_path) -> _TrainingRun:
    batches = [[{"image": torch.randn(3, 8, 8)}] for _ in range(2)]
    return _TrainingRun(
        args=args,
        variant_spec=get_gisec_variant_spec(args.variant),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        include_depth=False,
        train_loader=batches,
        val_loader=[],
        component_class_index=1,
        model=model,
        reference_source=None,
        optimizer=torch.optim.AdamW(model.parameters()),
        scaler=GradScaler(enabled=False),
        ann_file=tmp_path / "instances_val.json",
        metrics_log_path=tmp_path / "metrics_log.jsonl",
        resume_last_checkpoint=tmp_path / "resume_last.pth",
        best_checkpoint=tmp_path / "model_best.pth",
        params_trainable=sum(param.numel() for param in model.parameters()),
        start=time.perf_counter(),
    )


def _train_step_rows(path):
    return [row for row in _read_rows(path) if row["mode"] == "train_step"]


def test_max_train_steps_cap_runs_exactly_n_steps(tmp_path, monkeypatch) -> None:
    _stub_loop_dependencies(monkeypatch)
    args = parse_train_args(
        [
            "--dataset-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--variant",
            "base_rgb_1024",
            "--epochs",
            "3",
            "--max-train-steps",
            "3",
            "--log-every-steps",
            "1",
        ]
    )
    run = _loop_run(args, _LoopModel(), tmp_path)

    last_epoch, best_ap, _, _, _ = _run_training_loop(run)

    assert [row["global_step"] for row in _train_step_rows(run.metrics_log_path)] == [
        1,
        2,
        3,
    ]
    assert last_epoch == 2
    assert best_ap == 0.25


def test_resumed_run_at_cap_trains_no_extra_step(tmp_path, monkeypatch) -> None:
    _stub_loop_dependencies(monkeypatch)
    resume_checkpoint = tmp_path / "resume_last.pth"
    args = parse_train_args(
        [
            "--dataset-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--variant",
            "base_rgb_1024",
            "--epochs",
            "3",
            "--max-train-steps",
            "3",
            "--log-every-steps",
            "1",
            "--resume-checkpoint",
            str(resume_checkpoint),
        ]
    )
    model = _LoopModel()
    save_torch_payload(
        resume_checkpoint,
        resume_payload(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=GradScaler(enabled=False),
            args=args,
            completed_epoch=2,
            global_step=3,
            best_metric=0.25,
            running_step_time_total=0.5,
            elapsed_sec=10.0,
            peak_memory_mb=0.0,
        ),
    )
    run = _loop_run(args, model, tmp_path)

    last_epoch, best_ap, _, _, _ = _run_training_loop(run)

    assert _train_step_rows(run.metrics_log_path) == []
    assert last_epoch == 3
    assert best_ap == 0.25


def test_epoch_eval_row_stamps_the_decode_protocol(tmp_path, monkeypatch) -> None:
    _stub_loop_dependencies(monkeypatch)
    args = parse_train_args(
        [
            "--dataset-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--variant",
            "base_rgb_1024",
            "--eval-score-threshold",
            "0.1",
        ]
    )
    run = _loop_run(args, _LoopModel(), tmp_path)

    trainer_module._run_epoch_eval(run, epoch=1, best_ap_in=float("-inf"))

    row = next(r for r in _read_rows(run.metrics_log_path) if r["mode"] == "epoch_eval")
    assert row["eval_score_threshold"] == 0.1
    assert row["mask_threshold"] == 0.5


def test_run_summary_records_the_actual_eval_threshold(tmp_path, monkeypatch) -> None:
    _stub_loop_dependencies(monkeypatch)
    args = parse_train_args(
        [
            "--dataset-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--variant",
            "base_rgb_1024",
            "--eval-score-threshold",
            "0.2",
        ]
    )
    run = _loop_run(args, _LoopModel(), tmp_path)

    trainer_module._finalize_run(
        run,
        last_epoch=1,
        best_ap=0.25,
        final_metrics={"segm/AP": 0.25},
        final_speed={},
        final_eval_pending=False,
    )

    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["decode_config"] == {
        "eval_score_threshold": 0.2,
        "mask_threshold": 0.5,
        "graph_merge_threshold": 0.5,
    }
