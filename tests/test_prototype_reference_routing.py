from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gisec.datasets.prototype_bank import (
    PrototypeBankSource,
    extract_query_part_key,
    load_prototype_bank,
)


def _write_view(root: Path, stem: str, *, pos_x: float) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = (20, 40, 60)
    cv2.imwrite(str(root / "rgb" / f"{stem}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / f"{stem}.npy", np.full((32, 32), 0.95, dtype=np.float32))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(root / "mask" / f"{stem}.png"), mask)
    (root / "camera" / f"{stem}.json").write_text(
        (
            "{"
            f"\"position\":[{pos_x},0.0,1.0],"
            "\"quat_xyzw\":[0.0,0.0,0.0,1.0]"
            "}"
        ),
        encoding="utf-8",
    )


def _write_part_bank(root: Path, stems: list[str]) -> Path:
    for name in ["rgb", "depth", "mask", "meta", "camera"]:
        (root / name).mkdir(parents=True)
    for index, stem in enumerate(stems):
        _write_view(root, stem, pos_x=float(index))
    (root / "meta" / "manifest.json").write_text(
        '{"part_name":"demo","views":4,"qa_passed":true}',
        encoding="utf-8",
    )
    (root / "meta" / "qa_report.json").write_text(
        '{"qa_passed":true,"errors":[]}',
        encoding="utf-8",
    )
    return root


def test_extract_query_part_key_uses_longest_matching_prefix() -> None:
    part_key = extract_query_part_key(
        "A-DF15A_KG-T2S_1_100_scene_000003_000968_v0.png",
        [
            "A-DF15A",
            "A-DF15A_KG-T2S_1",
            "A-DF15A_KG",
        ],
    )

    assert part_key == "A-DF15A_KG-T2S_1"


def test_prototype_bank_source_resolves_part_subdirectory_from_query_name(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write_part_bank(root / "150044M155220", ["view_000", "view_001"])
    _write_part_bank(root / "A-DF15A_KG-T2S_1", ["view_000", "view_001"])

    source = PrototypeBankSource(root, image_size=64, contract_mode="compat")

    resolved = source.resolve_root_for_query("A-DF15A_KG-T2S_1_100_scene_000003_000968_v0.png")

    assert resolved == (root / "A-DF15A_KG-T2S_1").resolve()


def test_prototype_bank_source_loads_matching_part_bank(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write_part_bank(root / "150044M155220", ["view_000", "view_001"])
    _write_part_bank(root / "A-DF15A_KG-T2S_1", ["view_000", "view_001"])

    source = PrototypeBankSource(root, image_size=64, contract_mode="compat")
    bank = source.load_for_query("150044M155220_100_scene_000001_001202_v0.png")

    assert bank.root == (root / "150044M155220").resolve()
    assert len(bank.view_ids) == 2


def test_load_prototype_bank_can_limit_view_count_with_sampling(tmp_path: Path) -> None:
    root = _write_part_bank(tmp_path / "refs", ["view_000", "view_001", "view_002", "view_003"])

    bank = load_prototype_bank(
        root,
        image_size=64,
        contract_mode="compat",
        max_views=2,
        view_sampler="pose_farthest",
    )

    assert len(bank.view_ids) == 2
    assert tuple(bank.images.shape) == (2, 3, 64, 64)
