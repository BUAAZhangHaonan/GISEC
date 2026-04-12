from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

from baseline.yolo_seg.adapter import export_yolo_seg_dataset
from baseline.yolo_seg.train import train_yolo_seg_baseline


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[12:32, 8:28] = (60, 80, 120)
        image[28:56, 32:54] = (140, 90, 40)
        cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", np.full((64, 64), 0.9, dtype=np.float32))
        ann = {
            "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [8, 12, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                    "segmentation": [[8, 12, 28, 12, 28, 32, 8, 32]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [32, 28, 22, 28],
                    "area": 616,
                    "iscrowd": 0,
                    "segmentation": [[32, 28, 54, 28, 54, 56, 32, 56]],
                },
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


class _FakeYOLO:
    def __init__(self, _source: str) -> None:
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, kernel_size=1),
        )

    def train(self, **_kwargs) -> dict[str, str]:
        return {"status": "ok"}

    def predict(self, source, **_kwargs):
        height, width = source.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)
        mask[16 : height - 12, 12 : width - 16] = 1.0
        prediction = SimpleNamespace(
            masks=SimpleNamespace(data=torch.from_numpy(mask[None, ...])),
            boxes=SimpleNamespace(conf=torch.tensor([0.8], dtype=torch.float32)),
        )
        return [prediction]


class _FakeYOLOWithDownloads(_FakeYOLO):
    def train(self, **_kwargs) -> dict[str, str]:
        Path("yolon-seg.pt").write_text("downloaded\n", encoding="utf-8")
        Path("yolo26n.pt").write_text("downloaded\n", encoding="utf-8")
        return {"status": "ok"}


def test_yolo_adapter_exports_dataset_yaml_and_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    export_root = tmp_path / "export"
    _write_dataset(dataset_root)

    exported = export_yolo_seg_dataset(dataset_root=str(dataset_root), output_dir=str(export_root))

    yaml_payload = yaml.safe_load((Path(exported["root"]) / "dataset.yaml").read_text(encoding="utf-8"))
    assert yaml_payload["train"] == "images/train"
    assert yaml_payload["val"] == "images/val"
    assert yaml_payload["names"][0] == "component"
    train_labels = (Path(exported["root"]) / "labels" / "train" / "000001.txt").read_text(encoding="utf-8").strip().splitlines()
    val_labels = (Path(exported["root"]) / "labels" / "val" / "000001.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_labels) == 2
    assert len(val_labels) == 2
    assert all(line.startswith("0 ") for line in train_labels + val_labels)


def test_yolo_seg_rgb_baseline_smoke_exports_shared_artifacts(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)
    monkeypatch.setattr("baseline.yolo_seg.train.get_ultralytics_yolo_class", lambda: _FakeYOLO)

    train_yolo_seg_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_val_images=1,
        score_threshold=0.05,
        model_source="yolon-seg.yaml",
    )

    assert (output_root / "run_summary.json").exists()
    assert (output_root / "metrics.cocoeval.json").exists()
    assert (output_root / "inference_speed.json").exists()
    assert (output_root / "coco_instances_results.json").exists()
    assert (output_root / "params_trainable.txt").exists()
    assert (output_root / "wall_time_sec.txt").exists()
    assert (output_root / "peak_memory_mb.txt").exists()
    assert list((output_root / "visualizations" / "overlay").glob("*.png"))

    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "yolo_seg"
    assert summary["variant"] == "rgb_smoke"
    assert summary["modality"] == "rgb"
    assert summary["checkpoint"] == str((output_root / "model_final.pth").resolve())
    assert summary["results_json"] == str((output_root / "coco_instances_results.json").resolve())
    assert summary["params_trainable"] > 0
    assert summary["wall_time_sec"] >= 0


def test_yolo_seg_smoke_cleans_transient_downloaded_weight_files(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)

    unrelated_weight = repo_root / "yolo_keep.pt"
    unrelated_weight.write_text("keep\n", encoding="utf-8")
    dataset_root = repo_root / "dataset"
    output_root = repo_root / "out"
    _write_dataset(dataset_root)
    monkeypatch.setattr("baseline.yolo_seg.train.get_ultralytics_yolo_class", lambda: _FakeYOLOWithDownloads)

    train_yolo_seg_baseline(
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        image_size=64,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_val_images=1,
        score_threshold=0.05,
        model_source="yolon-seg.pt",
    )

    assert not (repo_root / "yolon-seg.pt").exists()
    assert not (repo_root / "yolo26n.pt").exists()
    assert unrelated_weight.exists()
