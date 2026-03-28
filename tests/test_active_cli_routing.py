from __future__ import annotations

from pathlib import Path


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
