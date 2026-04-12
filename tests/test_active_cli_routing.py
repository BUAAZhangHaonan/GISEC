from __future__ import annotations

import json
from pathlib import Path

from gisec.cli._routing import (
    explicit_cli_variant,
    resolve_cli_variant,
    resolve_run_directory_variant,
)


def test_default_cli_modules_route_to_active_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train_text = (repo_root / "gisec" / "cli" / "train.py").read_text(encoding="utf-8")
    eval_text = (repo_root / "gisec" / "cli" / "eval.py").read_text(encoding="utf-8")
    infer_text = (repo_root / "gisec" / "cli" / "infer.py").read_text(encoding="utf-8")

    assert "train_active" in train_text
    assert "eval_active" in eval_text
    assert "infer_active" in infer_text
    assert "train_gisec" not in train_text
    assert "train_gisec" not in eval_text
    assert "train_gisec" not in infer_text


def test_explicit_legacy_cli_modules_preserve_fragment_first_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for module_name in ["train_legacy.py", "eval_legacy.py", "infer_legacy.py"]:
        path = repo_root / "gisec" / "cli" / module_name
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "train_gisec" in text


def test_routing_helpers_accept_equals_form_flags(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run_summary.json").write_text(
        json.dumps({"variant": "base_rgb_1024"}),
        encoding="utf-8",
    )
    checkpoint_path = run_root / "model_final.pth"
    checkpoint_path.write_text("stub\n", encoding="utf-8")
    config_path = tmp_path / "active.yaml"
    config_path.write_text("model:\n  variant: base_rgb_1024\n", encoding="utf-8")

    assert explicit_cli_variant(["--variant=base_rgb_1024"]) == "base_rgb_1024"
    assert resolve_cli_variant(["--config=" + str(config_path)]) == "base_rgb_1024"
    assert resolve_run_directory_variant(["--checkpoint=" + str(checkpoint_path)]) == "base_rgb_1024"
    assert resolve_run_directory_variant(["--output-dir=" + str(run_root)]) == "base_rgb_1024"
