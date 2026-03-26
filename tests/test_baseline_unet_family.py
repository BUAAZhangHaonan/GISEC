from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.unet.eval import evaluate_unet_baseline
from baseline.unet.model import build_unet_family_model
from baseline.unet.train import train_unet_baseline


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = (60, 80, 120)
        cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", np.full((64, 64), 0.9, dtype=np.float32))
        ann = {
            "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [16, 16, 32, 32],
                    "area": 1024,
                    "iscrowd": 0,
                    "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
                }
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def test_unet_family_model_factory_builds_expected_variants() -> None:
    for name in ["unet", "unetpp", "attention_unet"]:
        model = build_unet_family_model(
            name,
            in_channels=3,
            encoder_name="resnet34",
            pretrained_backbone=False,
            decoder_channels=64,
        )
        outputs = model(torch.randn(2, 3, 64, 64))
        assert set(outputs) == {"fg_logits", "center_heatmap", "offsets", "boundary_logits", "feature_map"}
        assert outputs["fg_logits"].shape == (2, 1, 64, 64)
        assert outputs["center_heatmap"].shape == (2, 1, 64, 64)
        assert outputs["offsets"].shape == (2, 2, 64, 64)
        assert outputs["boundary_logits"].shape == (2, 1, 64, 64)
        assert outputs["feature_map"].shape == (2, 64, 64, 64)


def test_attention_unet_smoke_uses_shared_export_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    train_unet_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        threshold=0.5,
        model_name="attention_unet",
        task_mode="semantic_smoke",
    )

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "attention_unet"
    assert summary["variant"] == "rgb_smoke"
    assert summary["modality"] == "rgb"


def test_unet_family_supports_instance_training_options_in_summary(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    train_unet_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        threshold=0.5,
        model_name="unet",
        encoder_name="resnet34",
        pretrained_backbone=False,
        task_mode="instance",
        amp=True,
        grad_accum_steps=2,
    )

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "unet"
    assert summary["variant"] == "rgb_instance"


def test_unet_family_depth_wall_variant_is_reflected_in_summary(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    train_unet_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        threshold=0.5,
        model_name="unet",
        encoder_name="resnet34",
        pretrained_backbone=False,
        task_mode="instance",
        use_depth_split_walls=True,
        watershed_enabled=True,
    )

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["variant"] == "rgb_depth_wall_instance"


def test_unet_instance_training_single_epoch_skips_duplicate_final_eval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    calls: list[dict[str, object]] = []

    def _fake_eval(**kwargs):
        calls.append(kwargs)
        return {"segm/AP": 0.25}, {"status": "ok", "timed_images": 1}

    monkeypatch.setattr("baseline.unet.train.evaluate_unet_baseline", _fake_eval)

    train_unet_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        threshold=0.5,
        model_name="unet",
        encoder_name="resnet34",
        pretrained_backbone=False,
        task_mode="instance",
        amp=False,
        grad_accum_steps=1,
    )

    assert len(calls) == 1


def test_unet_instance_training_respects_eval_every_epochs(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    calls: list[dict[str, object]] = []

    def _fake_eval(**kwargs):
        calls.append(kwargs)
        return {"segm/AP": 0.25, "bbox/AP": 0.25}, {"status": "ok", "timed_images": 1}

    monkeypatch.setattr("baseline.unet.train.evaluate_unet_baseline", _fake_eval)

    train_unet_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=3,
        batch_size=1,
        num_workers=0,
        max_train_steps=0,
        max_val_images=1,
        threshold=0.5,
        model_name="unet",
        encoder_name="resnet34",
        pretrained_backbone=False,
        task_mode="instance",
        amp=False,
        grad_accum_steps=1,
        eval_every_epochs=2,
    )

    assert len(calls) == 2


def test_unet_instance_training_saves_model_final_once(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    def _fake_eval(**kwargs):
        return {"segm/AP": 0.25, "bbox/AP": 0.25}, {"status": "ok", "timed_images": 1}

    save_paths: list[str] = []
    original_save = torch.save

    def _record_save(obj, path, *args, **kwargs):
        save_paths.append(str(path))
        return original_save(obj, path, *args, **kwargs)

    monkeypatch.setattr("baseline.unet.train.evaluate_unet_baseline", _fake_eval)
    monkeypatch.setattr("baseline.unet.train.torch.save", _record_save)

    train_unet_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=3,
        batch_size=1,
        num_workers=0,
        max_train_steps=0,
        max_val_images=1,
        threshold=0.5,
        model_name="unet",
        encoder_name="resnet34",
        pretrained_backbone=False,
        task_mode="instance",
        amp=False,
        grad_accum_steps=1,
        eval_every_epochs=2,
    )

    model_final_saves = [path for path in save_paths if path.endswith("model_final.pth")]
    assert len(model_final_saves) == 1


def test_evaluate_unet_baseline_can_skip_overlay_rendering(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)
    model = build_unet_family_model(
        "unet",
        in_channels=3,
        encoder_name="resnet34",
        pretrained_backbone=False,
        decoder_channels=64,
    )
    calls: list[dict[str, object]] = []

    def _record_overlay(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("baseline.unet.eval.render_fragment_merge_preview", _record_overlay)

    evaluate_unet_baseline(
        model=model,
        model_name="unet",
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        num_workers=0,
        threshold=0.5,
        max_images=1,
        input_mode="rgb",
        task_mode="instance",
        render_overlay_limit=0,
    )

    assert calls == []
