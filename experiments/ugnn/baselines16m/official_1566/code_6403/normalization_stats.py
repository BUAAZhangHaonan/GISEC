from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.ensure_dataset_stats import ensure_dataset_stats


@dataclass(frozen=True)
class DatasetNormalizationStats:
    dataset_root: str
    rgb_mean_rgb_255: tuple[float, float, float]
    rgb_std_rgb_255: tuple[float, float, float]
    rgb_mean_bgr_255: tuple[float, float, float]
    rgb_std_bgr_255: tuple[float, float, float]
    depth_clip_min: float
    depth_clip_max: float


def _safe_std_triplet(values: list[float]) -> tuple[float, float, float]:
    return tuple(max(float(v), 1.0) for v in values[:3])  # type: ignore[return-value]


def load_dataset_normalization_stats(dataset_root: str) -> DatasetNormalizationStats:
    payload = ensure_dataset_stats(
        dataset_root=Path(dataset_root),
        cache_root=REPO_ROOT / "output" / "cache" / "dataset_stats",
    )
    rgb_stats = payload["rgb_stats"]
    depth_stats = payload["depth_stats"]
    return DatasetNormalizationStats(
        dataset_root=str(Path(dataset_root).resolve()),
        rgb_mean_rgb_255=tuple(float(v) for v in rgb_stats["mean_rgb"][:3]),
        rgb_std_rgb_255=_safe_std_triplet(rgb_stats["std_rgb"]),
        rgb_mean_bgr_255=tuple(float(v) for v in rgb_stats["mean_bgr"][:3]),
        rgb_std_bgr_255=_safe_std_triplet(rgb_stats["std_bgr"]),
        depth_clip_min=float(depth_stats["p1"]),
        depth_clip_max=float(depth_stats["p99"]),
    )
