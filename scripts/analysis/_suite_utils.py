from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRow:
    variant: str
    path: Path
    metrics: dict[str, Any]
    inference_speed: dict[str, Any]
    params_trainable: int | None
    wall_time_sec: int | None

    @property
    def segm_ap(self) -> float:
        return float(self.metrics.get("segm/AP", 0.0))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_suite_rows(suite_root: Path) -> list[RunRow]:
    rows: list[RunRow] = []
    for run_summary in sorted(suite_root.rglob("run_summary.json")):
        payload = _read_json(run_summary)
        rows.append(
            RunRow(
                variant=str(payload.get("variant", run_summary.parent.name)),
                path=run_summary.parent,
                metrics=dict(payload.get("metrics", {})),
                inference_speed=dict(payload.get("inference_speed", {})),
                params_trainable=payload.get("params_trainable"),
                wall_time_sec=payload.get("wall_time_sec"),
            )
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body]) + "\n"
