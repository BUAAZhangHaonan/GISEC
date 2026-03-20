from __future__ import annotations


def benchmark_config_defaults() -> dict[str, int | str]:
    return {
        "image_size": 1024,
        "batch": 1,
        "num_workers": 2,
        "device": "cuda",
    }
