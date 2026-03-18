from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_overlay_diagnostics_writes_preview(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = (60, 90, 140)
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    fragments = np.zeros((32, 32), dtype=np.int32)
    fragments[8:24, 8:16] = 1
    fragments[8:24, 16:24] = 2
    fragments_path = tmp_path / "fragments.npy"
    np.save(fragments_path, fragments)

    merged = np.zeros((32, 32), dtype=np.int32)
    merged[8:24, 8:24] = 1
    merged_path = tmp_path / "merged.npy"
    np.save(merged_path, merged)

    output_path = tmp_path / "overlay.png"

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/overlay_diagnostics.py",
            "--image",
            str(image_path),
            "--fragments",
            str(fragments_path),
            "--merged",
            str(merged_path),
            "--output",
            str(output_path),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
