#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


FORBIDDEN_TOP_LEVEL_DIRS = (
    "gisec_v3",
    "object_first",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--fail-on-issues", action="store_true")
    return parser.parse_args()


def _git(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _collect_worktrees(repo_root: Path) -> list[str]:
    raw = _git(repo_root, "worktree", "list", "--porcelain")
    if not raw:
        return []
    worktrees: list[str] = []
    for line in raw.splitlines():
        if line.startswith("worktree "):
            worktrees.append(line.removeprefix("worktree ").strip())
    return worktrees


def collect_repo_hygiene(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    current_head = _git(repo_root, "rev-parse", "HEAD")
    current_branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    origin_master = _git(repo_root, "rev-parse", "--verify", "origin/master")
    ahead_count_raw = _git(repo_root, "rev-list", "--count", "origin/master..HEAD") if origin_master else None
    ahead_count = int(ahead_count_raw) if ahead_count_raw else 0
    forbidden_paths = sorted(
        path.name
        for path in repo_root.iterdir()
        if path.is_dir() and path.name in FORBIDDEN_TOP_LEVEL_DIRS
    )
    worktrees = _collect_worktrees(repo_root)
    non_hidden_extra_worktrees = sorted(
        path
        for path in worktrees
        if Path(path).resolve() != repo_root and "/.worktree/" not in Path(path).as_posix()
    )
    untracked_raw = _git(repo_root, "status", "--short", "--untracked-files=all") or ""
    untracked_paths = sorted(
        line[3:]
        for line in untracked_raw.splitlines()
        if line.startswith("?? ")
    )
    issues: list[str] = []
    if forbidden_paths:
        issues.append(f"forbidden_top_level_dirs={','.join(forbidden_paths)}")
    if non_hidden_extra_worktrees:
        issues.append(f"non_hidden_extra_worktrees={','.join(non_hidden_extra_worktrees)}")
    return {
        "repo_root": str(repo_root),
        "current_head": current_head,
        "current_branch": current_branch,
        "origin_master_head": origin_master,
        "ahead_of_origin_master": ahead_count,
        "worktrees": worktrees,
        "non_hidden_extra_worktrees": non_hidden_extra_worktrees,
        "forbidden_paths": forbidden_paths,
        "untracked_paths": untracked_paths,
        "issues": issues,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Repo Hygiene Report",
        "",
        f"- repo_root: `{payload['repo_root']}`",
        f"- current_branch: `{payload.get('current_branch')}`",
        f"- current_head: `{payload.get('current_head')}`",
        f"- origin_master_head: `{payload.get('origin_master_head')}`",
        f"- ahead_of_origin_master: `{payload.get('ahead_of_origin_master')}`",
        f"- worktree_count: `{len(payload.get('worktrees', []))}`",
        f"- forbidden_paths: `{payload.get('forbidden_paths', [])}`",
        f"- issues: `{payload.get('issues', [])}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    payload = collect_repo_hygiene(repo_root)
    if args.output_json:
        out_json = Path(args.output_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        out_md = Path(args.output_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_markdown(payload), encoding="utf-8")
    if not args.output_json and not args.output_md:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.fail_on_issues and payload["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
