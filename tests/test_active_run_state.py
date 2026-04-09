from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import gisec.train.train_active as train_active_module
from gisec.train.train_active import parse_train_args, train_active


class _SingleBatchLoader:
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self._samples = samples

    def __iter__(self):
        yield list(self._samples)

    def __len__(self) -> int:
        return 1


class _TinyActiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(1.0))
        self.backbone = nn.Linear(1, 1)
        self.refiner = nn.Identity()
        self.feature_proj = nn.Identity()
        self.graph_head = None


def _sample() -> dict[str, torch.Tensor]:
    return {
        "image": torch.zeros((3, 8, 8), dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
        "masks": torch.zeros((1, 8, 8), dtype=torch.float32),
        "file_name": "sample.png",
    }


def _args(tmp_path: Path, *, variant: str = "base_rgb_1024") -> SimpleNamespace:
    argv = [
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--output-dir",
        str(tmp_path / "out"),
        "--variant",
        variant,
        "--device",
        "cpu",
        "--image-size",
        "8",
        "--epochs",
        "1",
        "--batch",
        "1",
        "--num-workers",
        "0",
        "--max-train-steps",
        "1",
        "--max-val-images",
        "1",
        "--eval-every-epochs",
        "0",
        "--log-every-steps",
        "1",
    ]
    if "refine" in variant:
        init_checkpoint = tmp_path / "init_model.pth"
        init_checkpoint.write_text("stub\n", encoding="utf-8")
        argv.extend(["--init-checkpoint", str(init_checkpoint)])
    return parse_train_args(argv)


def _patch_minimal_training(monkeypatch: pytest.MonkeyPatch, *, model: _TinyActiveModel) -> None:
    monkeypatch.setattr(
        train_active_module,
        "_build_loader",
        lambda **kwargs: _SingleBatchLoader([_sample()]),
    )
    monkeypatch.setattr(train_active_module, "_build_active_model", lambda args: model)
    monkeypatch.setattr(train_active_module, "_configure_model_for_stage", lambda model, args: None)
    monkeypatch.setattr(
        train_active_module,
        "_run_backbone",
        lambda **kwargs: SimpleNamespace(
            loss=model.scalar * 0.0 + 1.0,
            pixel_decoder_last_hidden_state=torch.zeros((1, 4, 4, 4), dtype=torch.float32),
            class_queries_logits=torch.zeros((1, 1, 2), dtype=torch.float32),
            masks_queries_logits=torch.zeros((1, 1, 8, 8), dtype=torch.float32),
            loss_dict={},
        ),
    )
    monkeypatch.setattr(
        train_active_module,
        "_evaluate_active",
        lambda **kwargs: ({"segm/AP": 0.0}, {"status": "ok", "timed_images": 1}),
    )


def test_train_active_fails_fast_on_non_finite_local_loss_and_marks_run_state_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyActiveModel()
    args = _args(tmp_path, variant="base_rgb_1024_refine")
    _patch_minimal_training(monkeypatch, model=model)
    monkeypatch.setattr(
        train_active_module,
        "_train_local_modules_with_metrics",
        lambda **kwargs: (
            model.scalar * torch.tensor(float("nan")),
            {
                "loss_local_total": float("nan"),
                "loss_local_mask": 0.0,
                "loss_local_boundary": 0.0,
                "loss_local_reference_positive": 0.0,
                "loss_local_reference_negative": 0.0,
                "loss_local_graph": 0.0,
                "local_refine_sec": 0.0,
                "local_reference_sec": 0.0,
                "local_graph_sec": 0.0,
            },
        ),
    )

    with pytest.raises(RuntimeError, match="Non-finite"):
        train_active(args)

    output_dir = Path(args.output_dir)
    run_state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))

    assert run_state["status"] == "failed"
    assert run_state["allow_resume"] is False
    assert "Non-finite" in run_state["failure_reason"]
    assert not (output_dir / "model_final.pth").exists()
    assert not (output_dir / "run_summary.json").exists()
    assert not (output_dir / "resume_last.pth").exists()


def test_train_active_refuses_to_start_when_stage_lock_is_owned_by_live_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyActiveModel()
    args = _args(tmp_path, variant="base_rgb_1024")
    _patch_minimal_training(monkeypatch, model=model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.lock").write_text(
        json.dumps({"pid": os.getpid(), "created_at": "2026-04-09T00:00:00"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="lock"):
        train_active(args)


def test_train_active_rejects_resume_checkpoint_without_running_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyActiveModel()
    args = _args(tmp_path, variant="base_rgb_1024")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = output_dir / "resume_last.pth"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": str(args.variant),
            "depth_mode": str(args.depth_mode),
            "model": {"variant": str(args.variant)},
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "completed_epoch": 1,
            "global_step": 1,
            "best_metric": 0.0,
            "running_step_time_total": 1.0,
        },
        resume_checkpoint,
    )
    args.resume_checkpoint = str(resume_checkpoint)
    _patch_minimal_training(monkeypatch, model=model)

    with pytest.raises(RuntimeError, match="run_state"):
        train_active(args)
