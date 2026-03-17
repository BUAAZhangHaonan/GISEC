from __future__ import annotations

from pathlib import Path


def test_reference_unet_gnn_all_runner_lists_full_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_reference_unet_gnn_all.sh"
    text = script.read_text(encoding="utf-8")
    for variant in ["B0", "G1", "G2", "G3", "G4", "G5"]:
        assert variant in text
