#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_TOP_LEVEL_DIRS = (
    "gisec_v3",
    "object_first",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--fail-on-issues", action="store_true")
    return parser.parse_args()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


def _collect_worktrees(repo_root: Path) -> list[dict[str, Any]]:
    raw = _git(repo_root, "worktree", "list", "--porcelain")
    if not raw:
        return []
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in raw.splitlines():
        if line.startswith("worktree "):
            if current is not None:
                rows.append(current)
            current = {"path": line.split(" ", 1)[1]}
            continue
        if not line.strip():
            continue
        if current is None:
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    if current is not None:
        rows.append(current)
    return rows


def collect_repo_hygiene(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    head = _git(repo_root, "rev-parse", "HEAD")
    origin_master = _git(repo_root, "rev-parse", "origin/master")
    top_level_entries = [path.name for path in repo_root.iterdir()]
    forbidden_paths = sorted(name for name in top_level_entries if name in FORBIDDEN_TOP_LEVEL_DIRS and (repo_root / name).is_dir())
    worktrees = _collect_worktrees(repo_root)
    extra_visible_worktrees = sorted(
        row["path"]
        for row in worktrees
        if Path(row["path"]).resolve() != repo_root and "/.worktree/" not in Path(row["path"]).as_posix()
    )
    issues: list[str] = []
    if forbidden_paths:
        issues.append(f"forbidden top-level dirs present: {', '.join(forbidden_paths)}")
    if extra_visible_worktrees:
        issues.append(f"non-hidden extra worktrees present: {', '.join(extra_visible_worktrees)}")
    return {
        "repo_root": str(repo_root),
        "current_head": head,
        "origin_master_head": origin_master,
        "worktrees": worktrees,
        "forbidden_paths": forbidden_paths,
        "extra_visible_worktrees": extra_visible_worktrees,
        "issues": issues,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Repo Hygiene Report",
        "",
        f"- repo_root: `{payload['repo_root']}`",
        f"- current_head: `{payload['current_head']}`",
        f"- origin_master_head: `{payload['origin_master_head']}`",
        f"- worktree_count: `{len(payload['worktrees'])}`",
        f"- forbidden_paths: `{payload['forbidden_paths']}`",
        f"- extra_visible_worktrees: `{payload['extra_visible_worktrees']}`",
        f"- issues: `{payload['issues']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = collect_repo_hygiene(Path(args.repo_root))
    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown_report(payload), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.fail_on_issues and payload["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
