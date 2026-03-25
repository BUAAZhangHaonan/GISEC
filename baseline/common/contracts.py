"""Shared benchmark contracts for baseline integrations."""

REQUIRED_ARTIFACTS = (
    "run_summary.json",
    "metrics.cocoeval.json",
    "inference_speed.json",
    "params_trainable.txt",
    "wall_time_sec.txt",
    "peak_memory_mb.txt",
    "visualizations/overlay",
)

REQUIRED_RUN_SUMMARY_KEYS = (
    "model",
    "variant",
    "modality",
    "artifact_root",
    "checkpoint",
    "results_json",
    "params_trainable",
    "wall_time_sec",
    "training_peak_memory_mb",
    "timing",
    "metrics",
    "inference_speed",
)
