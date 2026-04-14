from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.cuda.amp import GradScaler

from gisec.train import train_active as active_module

from scripts.audit.common import AUDIT_ROOT, mean_of, write_json


class _TimedLoader:
    def __init__(self, loader: Any, state: dict[str, Any]) -> None:
        self._loader = loader
        self._state = state

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Any:
        iterator = iter(self._loader)
        state = self._state

        class _Iterator:
            def __iter__(self) -> "_Iterator":
                return self

            def __next__(self) -> Any:
                start = time.perf_counter()
                batch = next(iterator)
                record = {
                    "data_load_sec": float(time.perf_counter() - start),
                    "forward_pass_sec": 0.0,
                    "loss_computation_sec": 0.0,
                    "backward_pass_sec": 0.0,
                    "optimizer_step_sec": 0.0,
                    "postprocess_metric_logging_sec": 0.0,
                    "local_refine_sec": 0.0,
                    "local_reference_sec": 0.0,
                    "local_graph_sec": 0.0,
                }
                state["current"] = record
                return batch

        return _Iterator()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._loader, item)


def _instrument_state() -> dict[str, Any]:
    return {
        "records": [],
        "current": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one-step breakdowns for active training.")
    parser.add_argument("--json-output", default=str(AUDIT_ROOT / "active_stage1_step_breakdown.json"))
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    if "--log-every-steps" not in train_args:
        train_args.extend(["--log-every-steps", "1"])

    state = _instrument_state()

    original_build_loader = active_module._build_loader
    original_run_backbone = active_module._run_backbone
    original_local = active_module._train_local_modules_with_metrics
    original_backward = active_module._backward_active_loss
    original_emit = active_module._emit_active_log

    def _build_loader(*args_build: Any, **kwargs_build: Any) -> Any:
        loader = original_build_loader(*args_build, **kwargs_build)
        if bool(kwargs_build.get("train", False)):
            return _TimedLoader(loader, state)
        return loader

    def _run_backbone(*args_backbone: Any, **kwargs_backbone: Any) -> Any:
        start = time.perf_counter()
        out = original_run_backbone(*args_backbone, **kwargs_backbone)
        record = state.get("current")
        if record is not None:
            record["forward_pass_sec"] += float(time.perf_counter() - start)
        return out

    def _train_local_modules_with_metrics(*args_local: Any, **kwargs_local: Any) -> Any:
        start = time.perf_counter()
        loss, metrics = original_local(*args_local, **kwargs_local)
        record = state.get("current")
        if record is not None:
            record["loss_computation_sec"] += float(time.perf_counter() - start)
            record["local_refine_sec"] = float(metrics.get("local_refine_sec", 0.0))
            record["local_reference_sec"] = float(metrics.get("local_reference_sec", 0.0))
            record["local_graph_sec"] = float(metrics.get("local_graph_sec", 0.0))
        return loss, metrics

    def _backward_active_loss(*, model: Any = None, optimizer: Any, scaler: GradScaler, loss: Any) -> bool:
        record = state.get("current")
        if record is None:
            return original_backward(model=model, optimizer=optimizer, scaler=scaler, loss=loss)
        optimizer.zero_grad(set_to_none=True)
        if not bool(loss.requires_grad):
            return False
        backward_start = time.perf_counter()
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if model is not None:
            grad_failures = []
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if not bool(active_module.torch.isfinite(param.grad.detach()).all().item()):
                    grad_failures.append(name)
            if grad_failures and not scaler.is_enabled():
                preview = ", ".join(grad_failures[:8])
                raise active_module.NonFiniteActiveTrainingError(f"Non-finite gradients detected after backward: {preview}")
        record["backward_pass_sec"] += float(time.perf_counter() - backward_start)
        step_start = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        record["optimizer_step_sec"] += float(time.perf_counter() - step_start)
        optimizer_failures = active_module._collect_non_finite_paths(optimizer.state_dict(), prefix="optimizer_state_dict")
        scaler_failures = active_module._collect_non_finite_paths(scaler.state_dict(), prefix="scaler_state_dict")
        if optimizer_failures or scaler_failures:
            preview = ", ".join((optimizer_failures + scaler_failures)[:8])
            raise active_module.NonFiniteActiveTrainingError(f"Non-finite optimizer or scaler state detected after step: {preview}")
        return True

    def _emit_active_log(metrics_log_path: Path, payload: dict[str, Any]) -> None:
        record = state.get("current")
        start = time.perf_counter()
        original_emit(metrics_log_path, payload)
        if record is not None and str(payload.get("mode")) == "train_step":
            record["postprocess_metric_logging_sec"] += float(time.perf_counter() - start)
            record["epoch"] = int(payload.get("epoch", 0))
            record["global_step"] = int(payload.get("global_step", 0))
            record["step_time_sec_logged"] = float(payload.get("step_time_sec", 0.0))
            state["records"].append(dict(record))
            state["current"] = None

    active_module._build_loader = _build_loader
    active_module._run_backbone = _run_backbone
    active_module._train_local_modules_with_metrics = _train_local_modules_with_metrics
    active_module._backward_active_loss = _backward_active_loss
    active_module._emit_active_log = _emit_active_log
    try:
        parsed = active_module.parse_train_args(train_args)
        active_module.train_active(parsed)
    finally:
        active_module._build_loader = original_build_loader
        active_module._run_backbone = original_run_backbone
        active_module._train_local_modules_with_metrics = original_local
        active_module._backward_active_loss = original_backward
        active_module._emit_active_log = original_emit

    records = state["records"]
    payload = {
        "step_count": len(records),
        "steps": records,
        "aggregate": {
            "data_load_sec_mean": mean_of(records, "data_load_sec"),
            "forward_pass_sec_mean": mean_of(records, "forward_pass_sec"),
            "loss_computation_sec_mean": mean_of(records, "loss_computation_sec"),
            "backward_pass_sec_mean": mean_of(records, "backward_pass_sec"),
            "optimizer_step_sec_mean": mean_of(records, "optimizer_step_sec"),
            "postprocess_metric_logging_sec_mean": mean_of(records, "postprocess_metric_logging_sec"),
            "local_refine_sec_mean": mean_of(records, "local_refine_sec"),
            "local_reference_sec_mean": mean_of(records, "local_reference_sec"),
            "local_graph_sec_mean": mean_of(records, "local_graph_sec"),
        },
    }
    write_json(args.json_output, payload)


if __name__ == "__main__":
    main()
