from __future__ import annotations


def benchmark_config_defaults() -> dict[str, int | str | bool]:
    return {
        "image_size": 1024,
        "batch": 1,
        "num_workers": 2,
        "device": "cuda",
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "eval_every_epochs": 1,
    }
