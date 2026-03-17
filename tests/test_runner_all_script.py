from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_reference_unet_gnn_all_runner_lists_full_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_affinigraph_all.sh"
    text = script.read_text(encoding="utf-8")
    for variant in ["B0", "G1", "G2", "G3", "G4", "G5"]:
        assert variant in text


def test_new_cli_modules_expose_train_eval_infer_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for module_name in ["affinigraph.cli.train", "affinigraph.cli.eval", "affinigraph.cli.infer"]:
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
