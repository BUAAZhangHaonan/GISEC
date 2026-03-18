from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from gisec.engine.runtime import evaluate_json


def test_evaluate_json_requires_pycocotools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ann_file = tmp_path / "instances_val.json"
    ann_file.write_text(json.dumps({"images": [], "annotations": [], "categories": []}), encoding="utf-8")
    results_json = tmp_path / "results.json"
    results_json.write_text("[]\n", encoding="utf-8")

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("pycocotools"):
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        evaluate_json(ann_file, results_json)

    assert "pycocotools" in str(exc_info.value)
