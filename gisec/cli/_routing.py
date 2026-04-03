from __future__ import annotations

import json
from pathlib import Path

from gisec.active.config import active_variant_names
from gisec.config.io import load_yaml_config, merge_config_dicts


def _existing(path_str: str | None) -> Path | None:
    if path_str in (None, ""):
        return None
    path = Path(str(path_str)).resolve()
    return path if path.exists() else None


def _summary_variant(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    variant = payload.get("variant")
    return None if variant in (None, "") else str(variant)


def explicit_cli_variant(argv: list[str]) -> str | None:
    variant = None
    for index, token in enumerate(argv):
        if token == "--variant" and index + 1 < len(argv):
            variant = argv[index + 1]
    return None if variant in {"", None} else str(variant)


def _config_variant(argv: list[str]) -> str | None:
    config_paths: list[str] = []
    for index, token in enumerate(argv):
        if token == "--config" and index + 1 < len(argv):
            config_paths.append(argv[index + 1])
    if not config_paths:
        return None
    merged = merge_config_dicts(load_yaml_config(Path(path)) for path in config_paths)
    config_variant = merged.get("model", {}).get("variant", "")
    return None if config_variant in {"", None} else str(config_variant)


def resolve_run_directory_variant(argv: list[str]) -> str | None:
    checkpoint_path = None
    output_dir = None
    for index, token in enumerate(argv):
        if token == "--checkpoint" and index + 1 < len(argv):
            checkpoint_path = argv[index + 1]
        if token == "--output-dir" and index + 1 < len(argv):
            output_dir = argv[index + 1]
    active_variants = set(active_variant_names())
    checkpoint = _existing(checkpoint_path)
    output_root = _existing(output_dir)
    candidate_roots = []
    if checkpoint is not None:
        candidate_roots.append(checkpoint.parent)
    if output_root is not None:
        candidate_roots.append(output_root)
    for root in candidate_roots:
        if (root / "model_config.json").exists():
            return "__legacy__"
        summary_variant = _summary_variant(root / "run_summary.json")
        if summary_variant in active_variants:
            return summary_variant
        if summary_variant not in {None, ""}:
            return "__legacy__"
    return None


def resolve_cli_variant(argv: list[str]) -> str | None:
    variant = explicit_cli_variant(argv)
    if variant not in {"", None}:
        return str(variant)
    config_variant = _config_variant(argv)
    if config_variant not in {"", None}:
        return str(config_variant)
    return resolve_run_directory_variant(argv)


def should_route_legacy(argv: list[str]) -> bool:
    variant = resolve_cli_variant(argv)
    return variant == "__legacy__" or (variant not in {None, ""} and variant not in set(active_variant_names()))
