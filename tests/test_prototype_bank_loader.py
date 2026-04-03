from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gisec.datasets.prototype_bank import PrototypeBankContractError, load_prototype_bank


def test_load_prototype_bank_requires_rgb_depth_mask_dirs(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    (root / "rgb").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_prototype_bank(root, image_size=64)


def test_load_prototype_bank_sorts_views_and_builds_shape_stats(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    for name in ["rgb", "depth", "mask", "meta"]:
        (root / name).mkdir(parents=True)
    for stem in ["b", "a"]:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[8:24, 8:24] = (20, 40, 60)
        cv2.imwrite(str(root / "rgb" / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / f"{stem}.npy", np.full((32, 32), 0.95, dtype=np.float32))
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        cv2.imwrite(str(root / "mask" / f"{stem}.png"), mask)

    bank = load_prototype_bank(root, image_size=64)
    assert bank.view_ids == ["a", "b"]
    assert tuple(bank.images.shape) == (2, 3, 64, 64)
    assert tuple(bank.depths.shape) == (2, 1, 64, 64)
    assert tuple(bank.masks.shape) == (2, 1, 64, 64)
    assert bank.shape_stats["mean_area_ratio"] > 0.0


def test_load_prototype_bank_compat_mode_backfills_missing_shape_stats(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    for name in ["rgb", "depth", "mask", "meta", "camera"]:
        (root / name).mkdir(parents=True)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:24, 8:24] = (20, 40, 60)
    cv2.imwrite(str(root / "rgb" / "view0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / "view0.npy", np.full((32, 32), 0.95, dtype=np.float32))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(root / "mask" / "view0.png"), mask)
    (root / "meta" / "manifest.json").write_text('{"part_name":"demo","num_views":1,"qa_passed":true}', encoding="utf-8")
    (root / "meta" / "qa_report.json").write_text('{"qa_passed":true,"errors":[]}', encoding="utf-8")
    (root / "camera" / "view0.json").write_text('{"view_id":"view0"}', encoding="utf-8")

    bank = load_prototype_bank(root, image_size=64, contract_mode="compat")

    assert bank.manifest.contract_mode == "compat"
    assert not bank.manifest.has_shape_stats
    assert not bank.manifest.has_preview_contact_sheet
    assert bank.shape_stats["mean_area_ratio"] > 0.0
    assert bank.shape_stats["mean_bbox_aspect_ratio"] > 0.0
    for key in ["area_q10", "area_q50", "area_q90", "aspect_q10", "aspect_q50", "aspect_q90"]:
        assert key in bank.shape_stats
        assert isinstance(bank.shape_stats[key], float)


def test_load_prototype_bank_strict_mode_reports_missing_contract_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    for name in ["rgb", "depth", "mask", "meta", "camera"]:
        (root / name).mkdir(parents=True)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "rgb" / "view0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / "view0.npy", np.full((32, 32), 0.95, dtype=np.float32))
    cv2.imwrite(str(root / "mask" / "view0.png"), np.ones((32, 32), dtype=np.uint8) * 255)
    (root / "meta" / "manifest.json").write_text('{"part_name":"demo","num_views":1,"qa_passed":true}', encoding="utf-8")
    (root / "meta" / "qa_report.json").write_text('{"qa_passed":true,"errors":[]}', encoding="utf-8")
    (root / "camera" / "view0.json").write_text('{"view_id":"view0"}', encoding="utf-8")

    with pytest.raises(PrototypeBankContractError) as exc_info:
        load_prototype_bank(root, image_size=64, contract_mode="strict")

    assert "meta/shape_stats.json" in str(exc_info.value)
    assert "meta/preview_contact_sheet.png" in str(exc_info.value)


def test_load_prototype_bank_strict_mode_rejects_failed_qa_and_manifest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    for name in ["rgb", "depth", "mask", "meta", "camera"]:
        (root / name).mkdir(parents=True)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "rgb" / "view0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / "view0.npy", np.full((32, 32), 0.95, dtype=np.float32))
    cv2.imwrite(str(root / "mask" / "view0.png"), np.ones((32, 32), dtype=np.uint8) * 255)
    (root / "camera" / "view0.json").write_text('{"view_id":"view0"}', encoding="utf-8")
    (root / "meta" / "manifest.json").write_text('{"part_name":"demo","num_views":99,"qa_passed":false}', encoding="utf-8")
    (root / "meta" / "qa_report.json").write_text('{"qa_passed":false,"errors":["alignment_failed"]}', encoding="utf-8")
    (root / "meta" / "shape_stats.json").write_text('{"mean_area_ratio":0.5,"mean_bbox_aspect_ratio":1.0}', encoding="utf-8")
    (root / "meta" / "preview_contact_sheet.png").write_bytes(b"fake")

    with pytest.raises(PrototypeBankContractError) as exc_info:
        load_prototype_bank(root, image_size=64, contract_mode="strict")

    error_text = str(exc_info.value)
    assert "qa_passed=false" in error_text
    assert "num_views=99 expected=1" in error_text


def test_load_prototype_bank_strict_mode_rejects_modality_stem_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    for name in ["rgb", "depth", "mask", "meta", "camera"]:
        (root / name).mkdir(parents=True)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "rgb" / "view0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(root / "rgb" / "view1.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / "view0.npy", np.full((32, 32), 0.95, dtype=np.float32))
    cv2.imwrite(str(root / "mask" / "view0.png"), np.ones((32, 32), dtype=np.uint8) * 255)
    cv2.imwrite(str(root / "mask" / "view1.png"), np.ones((32, 32), dtype=np.uint8) * 255)
    (root / "camera" / "view0.json").write_text('{"view_id":"view0"}', encoding="utf-8")
    (root / "camera" / "view1.json").write_text('{"view_id":"view1"}', encoding="utf-8")
    (root / "meta" / "manifest.json").write_text('{"part_name":"demo","num_views":2,"qa_passed":true}', encoding="utf-8")
    (root / "meta" / "qa_report.json").write_text('{"qa_passed":true,"errors":[]}', encoding="utf-8")
    (root / "meta" / "shape_stats.json").write_text('{"mean_area_ratio":0.5,"mean_bbox_aspect_ratio":1.0}', encoding="utf-8")
    (root / "meta" / "preview_contact_sheet.png").write_bytes(b"fake")

    with pytest.raises(PrototypeBankContractError) as exc_info:
        load_prototype_bank(root, image_size=64, contract_mode="strict")

    assert "rgb/depth/mask stem mismatch" in str(exc_info.value)


def test_load_prototype_bank_accepts_legacy_views_field_with_deprecation_warning(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    for name in ["rgb", "depth", "mask", "meta", "camera"]:
        (root / name).mkdir(parents=True)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "rgb" / "view0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / "view0.npy", np.full((32, 32), 0.95, dtype=np.float32))
    cv2.imwrite(str(root / "mask" / "view0.png"), np.ones((32, 32), dtype=np.uint8) * 255)
    (root / "camera" / "view0.json").write_text('{"view_id":"view0"}', encoding="utf-8")
    (root / "meta" / "manifest.json").write_text('{"part_name":"demo","views":1,"qa_passed":true}', encoding="utf-8")
    (root / "meta" / "qa_report.json").write_text('{"qa_passed":true,"errors":[]}', encoding="utf-8")
    (root / "meta" / "shape_stats.json").write_text('{"mean_area_ratio":0.5,"mean_bbox_aspect_ratio":1.0}', encoding="utf-8")
    (root / "meta" / "preview_contact_sheet.png").write_bytes(b"fake")

    with pytest.warns(DeprecationWarning, match="num_views"):
        bank = load_prototype_bank(root, image_size=64, contract_mode="strict")

    assert bank.manifest.view_count == 1
