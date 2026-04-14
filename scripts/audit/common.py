from __future__ import annotations

import json
import math
import statistics
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from gisec.config.io import extract_argparse_defaults, load_yaml_config, merge_config_dicts


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / "output" / "audit" / "2026-04-14-performance"
DATA_SMALL_CONFIG = REPO_ROOT / "configs" / "data" / "ecc_20260318_1k_1566.yaml"
DATA_FULL_CONFIG = REPO_ROOT / "configs" / "data" / "ecc_20260318_1k_32254.yaml"
REFERENCE_CONFIG = REPO_ROOT / "configs" / "reference" / "reference_20260318_1k_13440.yaml"
ACTIVE_STAGE1_CONFIG = REPO_ROOT / "configs" / "active" / "base_rgb_1024.yaml"
ACTIVE_RESCUE_CONFIG = REPO_ROOT / "configs" / "active" / "base_rgb_1024_refine_ref_graph.yaml"
QUERY_BASELINE_TRAIN_CONFIG = REPO_ROOT / "configs" / "query" / "train" / "query_small_resnet18_full_train.yaml"
QUERY_GRAPH_TRAIN_CONFIG = REPO_ROOT / "configs" / "query" / "train" / "query_graph_resnet18_full_train.yaml"
QUERY_REFGRAPH_TRAIN_CONFIG = REPO_ROOT / "configs" / "query" / "train" / "query_refgraph_resnet18_full_train.yaml"
QUERY_BASELINE_EVAL_CONFIG = REPO_ROOT / "configs" / "query" / "eval" / "query_small_resnet18_full_eval.yaml"
LEGACY_VARIANT_CONFIG = REPO_ROOT / "configs" / "variant" / "legacy_rgbd_prototype_ownership_graph_cues.yaml"
LEGACY_TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "recovery_smoke_1024.yaml"
ACTIVE_INIT_CHECKPOINT = REPO_ROOT / "output" / "experiments" / "2026-04-13-rgb-full-rerun" / "phase_c" / "active_rgb_official" / "train" / "base_rgb_1024" / "model_best.pth"


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = ensure_parent(path)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved


def load_defaults(config_paths: list[str | Path], *, mode: str) -> dict[str, Any]:
    merged = merge_config_dicts(load_yaml_config(Path(path)) for path in config_paths)
    return extract_argparse_defaults(merged, mode=mode)


def resolve_dataset_root(config_paths: list[str | Path], *, mode: str = "train") -> str:
    defaults = load_defaults(config_paths, mode=mode)
    value = defaults.get("dataset_root", "")
    if not value:
        raise ValueError(f"dataset_root missing from config set: {config_paths}")
    return str(Path(str(value)).resolve())


def resolve_prototype_root(config_paths: list[str | Path], *, mode: str = "train") -> str:
    defaults = load_defaults(config_paths, mode=mode)
    value = defaults.get("prototype_root", "")
    if not value:
        raise ValueError(f"prototype_root missing from config set: {config_paths}")
    return str(Path(str(value)).resolve())


def resolve_output_dir(config_paths: list[str | Path], *, mode: str = "train") -> str:
    defaults = load_defaults(config_paths, mode=mode)
    value = defaults.get("output_dir", "")
    if not value:
        raise ValueError(f"output_dir missing from config set: {config_paths}")
    return str(Path(str(value)).resolve())


def percentile_ms(values_sec: list[float], q: float) -> float:
    if not values_sec:
        return 0.0
    return float(np.percentile(np.asarray(values_sec, dtype=np.float64), q)) * 1000.0


def summarize_latencies(values_sec: list[float]) -> dict[str, float | int]:
    if not values_sec:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p95_ms": 0.0,
            "stdev_ms": 0.0,
        }
    values_ms = [float(value) * 1000.0 for value in values_sec]
    return {
        "count": len(values_sec),
        "mean_ms": float(statistics.mean(values_ms)),
        "min_ms": float(min(values_ms)),
        "max_ms": float(max(values_ms)),
        "p95_ms": percentile_ms(values_sec, 95.0),
        "stdev_ms": 0.0 if len(values_ms) < 2 else float(statistics.pstdev(values_ms)),
    }


def time_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn()
    return out, float(time.perf_counter() - start)


def time_repeated(fn: Callable[[int], Any], count: int) -> tuple[list[Any], list[float]]:
    outputs: list[Any] = []
    latencies: list[float] = []
    for index in range(int(count)):
        start = time.perf_counter()
        outputs.append(fn(index))
        latencies.append(float(time.perf_counter() - start))
    return outputs, latencies


class CallTimer:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.seconds += float(time.perf_counter() - start)
                self.calls += 1

        return _wrapped

    def summary(self) -> dict[str, float | int]:
        return {
            "calls": int(self.calls),
            "total_ms": float(self.seconds * 1000.0),
        }


@contextmanager
def patch_many(patches: list[Any]) -> Iterator[None]:
    with ExitStack() as stack:
        for patch_obj in patches:
            stack.enter_context(patch_obj)
        yield


def make_scalar(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def mean_of(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0)) for record in records]
    return 0.0 if not values else float(statistics.mean(values))


def sum_of(records: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(record.get(key, 0.0)) for record in records))


def has_nonzero(records: list[dict[str, Any]], key: str) -> bool:
    return any(abs(float(record.get(key, 0.0))) > 0.0 for record in records)


def normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def safe_ratio(numerator: float, denominator: float) -> float:
    if math.isclose(float(denominator), 0.0):
        return 0.0
    return float(numerator) / float(denominator)
