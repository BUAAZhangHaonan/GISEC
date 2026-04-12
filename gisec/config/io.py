from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


_MODE_KEYS = {"common", "train", "eval", "infer"}
_SECTION_ALIASES = {
    "data": {"root": "dataset_root"},
    "dataset": {"root": "dataset_root"},
    "reference": {
        "root": "prototype_root",
        "max_views": "reference_max_views",
        "view_sampler": "reference_view_sampler",
    },
    "output": {"dir": "output_dir"},
    "run": {"output_dir": "output_dir"},
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[str, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(payload).__name__}")
    return payload


def merge_config_dicts(configs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in configs:
        merged = _deep_merge(merged, config)
    return merged


def extract_argparse_defaults(config: dict[str, Any], mode: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key, value in config.items():
        if key in _MODE_KEYS:
            continue
        if isinstance(value, dict):
            defaults.update(_flatten_alias_section(key, value))
        else:
            defaults[key] = value
    defaults.update(_as_mapping(config.get("common")))
    defaults.update(_as_mapping(config.get(mode)))
    return defaults


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected config section to be a mapping, got {type(value).__name__}")
    return dict(value)


def _flatten_alias_section(section_key: str, section_value: dict[str, Any]) -> dict[str, Any]:
    alias_map = _SECTION_ALIASES.get(section_key, {})
    flattened: dict[str, Any] = {}
    for key, value in section_value.items():
        flattened[alias_map.get(key, f"{section_key}_{key}")] = value
    return flattened


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged
