"""Shared run machinery for the trainer and evaluator.

Device selection, JSON artifact writing, and the latency statistics payload
used by the training and evaluation entrypoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def build_latency_payload(
    latencies_ms: list[float],
    device: torch.device,
    *,
    scope: str = "backbone_forward",
) -> dict[str, Any]:
    values = np.asarray(latencies_ms, dtype=np.float32)
    if values.size == 0:
        return {
            "device": device.type,
            "scope": str(scope),
            "images": 0,
            "latency_ms_mean": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p90": 0.0,
        }
    return {
        "device": device.type,
        "scope": str(scope),
        "images": int(values.size),
        "latency_ms_mean": float(values.mean()),
        "latency_ms_p50": float(np.quantile(values, 0.50)),
        "latency_ms_p90": float(np.quantile(values, 0.90)),
    }


def build_device(device_name: str) -> torch.device:
    requested = str(device_name or "cpu").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2,
                    ensure_ascii=False) + "\n", encoding="utf-8")
