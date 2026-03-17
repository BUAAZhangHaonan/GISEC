from __future__ import annotations

import subprocess
from pathlib import Path


def test_reference_unet_gnn_runner_dry_run_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_affinigraph.sh"
    ref_root = tmp_path / "reference"
    for name in ["rgb", "depth", "mask", "meta"]:
        (ref_root / name).mkdir(parents=True)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--reference-root",
            str(ref_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "G5",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode=dry-run" in res.stdout
    assert "variant=G5" in res.stdout
    assert "reference_root=" in res.stdout
    assert "python -m affinigraph.cli.train" in res.stdout
    assert "conda run -n magformer" not in res.stdout
