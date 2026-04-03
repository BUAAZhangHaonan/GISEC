from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Codex"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "codex@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True, text=True)


def test_repo_hygiene_reports_forbidden_top_level_dirs(tmp_path: Path) -> None:
    repo_script = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "check_repo_hygiene.py"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    (repo_root / "gisec_v3").mkdir()
    (repo_root / "gisec_v3" / "tmp.txt").write_text("bad\n", encoding="utf-8")
    out_json = tmp_path / "repo_hygiene.json"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_script),
            "--repo-root",
            str(repo_root),
            "--output-json",
            str(out_json),
            "--fail-on-issues",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["current_head"]
    assert payload["forbidden_paths"] == ["gisec_v3"]
    assert payload["issues"]
    assert payload["worktrees"]


def test_repo_hygiene_passes_clean_repo(tmp_path: Path) -> None:
    repo_script = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "check_repo_hygiene.py"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    out_json = tmp_path / "repo_hygiene.json"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_script),
            "--repo-root",
            str(repo_root),
            "--output-json",
            str(out_json),
            "--fail-on-issues",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["forbidden_paths"] == []
    assert payload["issues"] == []
