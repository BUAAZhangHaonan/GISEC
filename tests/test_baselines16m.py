"""Tests for experiments/ugnn/baselines16m data plumbing.

Covers the two supervision-path bugs fixed after the first baseline
round: (1) packed masks must round-trip np.packbits exactly (MSB-first
bit order; a LSB-first decode mirrors every 8-pixel block), and (2)
the single-class label convention (dataset keeps the raw COCO
category id 1; the Mask2Former collate remaps to 0-based, torchvision
keeps 1-based with 0 as background).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
BASELINES16M = REPO / "experiments" / "ugnn" / "baselines16m"


def _module(name: str):
    for entry in (BASELINES16M, REPO / "src"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    return importlib.import_module(name)


def _make_fake_split(tmp_path: Path) -> Path:
    """Single 1024-space square annotation + tiny RGB and depth files."""
    root = tmp_path / "32254"
    (root / "images" / "train").mkdir(parents=True)
    (root / "depth" / "depth_npy" / "train").mkdir(parents=True)
    cv2.imwrite(
        str(root / "images" / "train" / "fake.png"),
        np.full((48, 64, 3), 127, np.uint8),
    )
    np.save(
        root / "depth" / "depth_npy" / "train" / "fake.npy",
        np.full((48, 64), 0.45, np.float32),
    )
    payload = {
        "images": [{"id": 1, "file_name": "fake.png"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 100.0, 120.0, 80.0],
                "segmentation": [
                    [100.0, 100.0, 220.0, 100.0, 220.0, 180.0, 100.0, 180.0]
                ],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component", "supercategory": "part"}],
    }
    (root / "annotations").mkdir()
    (root / "annotations" / "instances_train.json").write_text(json.dumps(payload))
    return root


def test_unpack_masks_asymmetric_pattern() -> None:
    common = _module("common")
    # Asymmetric pattern inside each 8-pixel block: first 4 pixels set.
    # A LSB-first decode would mirror it onto the last 4 pixels.
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask.reshape(1024, 128, 8)[:, :, :4] = 1
    packed = torch.from_numpy(np.packbits(mask, axis=None)[None])
    unpacked = common.unpack_masks(packed)
    assert unpacked.shape == (1, 1024, 1024)
    assert np.array_equal(unpacked[0].numpy(), mask)


def test_unpack_masks_random_roundtrip_matches_unpackbits() -> None:
    common = _module("common")
    rng = np.random.default_rng(0)
    masks = (rng.random((3, 1024, 1024)) < 0.25).astype(np.uint8)
    packed = np.stack([np.packbits(m, axis=None) for m in masks])
    unpacked = common.unpack_masks(torch.from_numpy(packed))
    reference = np.unpackbits(packed, axis=None).reshape(3, 1024, 1024)
    assert np.array_equal(unpacked.numpy(), reference)
    assert np.array_equal(unpacked.numpy(), masks)


def test_dataset_mask_area_bbox_match_ann_to_mask(tmp_path: Path) -> None:
    common = _module("common")
    from gisec.datasets.coco_utils import ann_to_mask

    root = _make_fake_split(tmp_path)
    ds = common.Baseline16mDataset("train", data_root=root)
    item = ds[0]

    assert item["labels"].tolist() == [1]  # raw COCO category id
    assert item["boxes"].tolist() == [[100.0, 100.0, 220.0, 180.0]]

    ann = ds.coco.loadAnns(ds.coco.getAnnIds(imgIds=[1], iscrowd=False))[0]
    direct = ann_to_mask(ann, 1024, 1024)
    masks = common.unpack_masks(item["packed_masks"])
    assert np.array_equal(masks[0].numpy(), direct)
    assert int(masks.sum()) == int(direct.sum())

    ys, xs = np.nonzero(masks[0].numpy())
    assert (xs.min(), ys.min()) == (100, 100)
    assert abs(int(xs.max()) - 219) <= 1
    assert abs(int(ys.max()) - 179) <= 1


def test_collate_label_conventions(tmp_path: Path) -> None:
    common = _module("common")
    root = _make_fake_split(tmp_path)
    item = common.Baseline16mDataset("train", data_root=root)[0]

    _images, targets = common.collate_mrcnn([item])
    assert targets[0]["labels"].tolist() == [1]  # torchvision: background=0

    images, _pm, _packed, class_labels = common.collate_m2f([item])
    assert class_labels[0].tolist() == [0]  # M2F: 0-based single class
    assert images.shape == (1, 3, 48, 64)


def test_collate_mrcnn_concatenates_depth_channel(tmp_path: Path) -> None:
    common = _module("common")
    root = _make_fake_split(tmp_path)
    item = common.Baseline16mDataset("train", data_root=root, include_depth=True)[0]

    assert item["depth"] is not None
    assert item["depth"].shape == (1, 48, 64)
    images, _targets = common.collate_mrcnn([item])
    assert images[0].shape == (4, 48, 64)
    assert torch.allclose(images[0][3], item["depth"][0])


def test_foreground_keep_drops_nonzero_classes() -> None:
    eval_mod = _module("eval")
    labels = torch.tensor([0, 1, 2, 0])
    scores = torch.tensor([0.9, 0.95, 0.99, 0.5])
    keep = eval_mod.foreground_keep(labels, scores, score_thr=0.05)
    assert keep.tolist() == [True, False, False, True]
    keep = eval_mod.foreground_keep(labels, scores, score_thr=0.6)
    assert keep.tolist() == [True, False, False, False]


class _StubM2FOutput:
    def __init__(self, class_logits, mask_logits):
        self.class_queries_logits = class_logits
        self.masks_queries_logits = mask_logits


class _StubM2F(torch.nn.Module):
    """Three queries over four class logits: class 0, class 1, class 3."""

    def forward(self, pixel_values=None, output_hidden_states=True, **_):
        class_logits = torch.tensor(
            [[[5.0, 0.0, 0.0, -5.0], [0.0, 5.0, 0.0, -5.0], [-5.0, 0.0, 0.0, 5.0]]]
        )
        mask_logits = torch.full((1, 3, 8, 8), -12.0)
        mask_logits[0, 0, :4, :] = 12.0  # the class-0 query covers the top half
        return _StubM2FOutput(class_logits, mask_logits)


def test_predict_m2f_keeps_only_class_zero() -> None:
    eval_mod = _module("eval")
    loader = [
        (
            torch.zeros((1, 3, 64, 64)),
            torch.ones((1, 64, 64), dtype=torch.long),
            [torch.zeros((1, 1024 * 1024 // 8), dtype=torch.uint8)],
            [torch.zeros((1,), dtype=torch.int64)],
        )
    ]
    collected = []
    eval_mod.predict_m2f(
        _StubM2F(),
        loader,
        "cpu",
        lambda scores, masks: collected.append((scores, masks)),
        score_thr=0.05,
        mask_thr=0.5,
    )
    assert len(collected) == 1
    scores, masks = collected[0]
    assert scores.shape == (1,)  # only the class-0 query survives
    assert scores[0] > 0.9
    assert masks.shape == (1, 1024, 1024)
    mask = masks[0]
    assert mask[:448].all()
    assert mask[576:].sum() == 0


_R18_CACHE = (
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
)


@pytest.mark.skipif(not _R18_CACHE.exists(), reason="resnet18 V1 weights not cached")
def test_build_mrcnn_4ch_conv1_init() -> None:
    build_models = _module("build_models")

    m3 = build_models.build_mrcnn()
    m4 = build_models.build_mrcnn(in_chans=4)

    w3 = m3.backbone.body.conv1.weight
    w4 = m4.backbone.body.conv1.weight
    assert w4.shape == (64, 4, 7, 7)
    assert torch.allclose(w4[:, :3], w3)
    assert torch.allclose(w4[:, 3:4], w3.mean(dim=1, keepdim=True))

    # The transform must accept the 4-channel input (4-entry mean/std).
    image_list, _ = m4.transform([torch.zeros(4, 64, 64)], None)
    assert image_list.tensors.shape[1] == 4


@pytest.mark.skipif(not _R18_CACHE.exists(), reason="resnet18 V1 weights not cached")
def test_mrcnn_param_budget_strict() -> None:
    """Expert round-2 fix: box-head width 192 measured 17.0003M (3ch)
    / 17.0034M (4ch) - both over the strict <=17,000,000 ceiling.
    Width 191 must put BOTH arms strictly under it (train.py asserts
    the same bound at runtime)."""
    build_models = _module("build_models")

    def params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    m3 = build_models.build_mrcnn()
    m4 = build_models.build_mrcnn(in_chans=4)
    assert params(m3) < 17_000_000
    assert params(m4) < 17_000_000
    assert params(m4) - params(m3) == 64 * 7 * 7  # one extra input channel


def test_family_data_flags_table() -> None:
    """Single source for the per-family input pipeline (train / eval /
    calibrate share common.family_data_flags)."""
    common = _module("common")
    flags = {f: common.family_data_flags(f) for f in common.FAMILIES}
    assert flags["mrcnn16"] == (False, False)
    assert flags["mrcnn16d"] == (True, False)  # depth, no ImageNet norm
    assert flags["m2f16"] == (False, False)  # historical: bare [0,1] RGB
    assert flags["m2f16cat"] == (True, False)  # historical: bare [0,1] RGB
    assert flags["m2f16fix"] == (False, True)
    assert flags["m2f16v2"] == (False, True)  # clean m2f16 replacement
    assert flags["m2f16catfix"] == (True, True)  # ImageNet on RGB only


def test_dataset_imagenet_norm_touches_rgb_only(tmp_path: Path) -> None:
    """m2f16v2 / m2f16catfix input pipeline: the RGB tensor gets the
    ImageNet mean/std; the depth tensor keeps the global DEPTH_LO/HI
    calibration and is never ImageNet-normalised."""
    common = _module("common")
    root = _make_fake_split(tmp_path)

    raw = common.Baseline16mDataset("train", data_root=root)[0]
    fixed = common.Baseline16mDataset(
        "train", data_root=root, include_depth=True, imagenet_norm=True
    )[0]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    assert torch.allclose(fixed["image"], (raw["image"] - mean) / std, atol=1e-6)

    calibrated = np.clip(
        (0.45 - common.DEPTH_LO) / (common.DEPTH_HI - common.DEPTH_LO), -1.0, 2.0
    )
    assert fixed["depth"] is not None
    assert torch.allclose(
        fixed["depth"][0], torch.tensor(calibrated, dtype=torch.float32)
    )
    # not the ImageNet-normalised version of anything:
    assert not torch.allclose(
        fixed["depth"], (fixed["depth"] - mean[0, 0, 0]) / std[0, 0, 0]
    )
    # plain (no-norm) depth arm is bit-identical
    plain = common.Baseline16mDataset("train", data_root=root, include_depth=True)[0]
    assert torch.equal(fixed["depth"], plain["depth"])


def test_parse_scene_key_and_edge_dirs() -> None:
    """calibrate_and_report.py pre-registered helpers: the scene key
    matches lib/eval_scale.scene_key, and R4 edge detection reports the
    violated direction only."""
    cal = _module("calibrate_and_report")
    assert (
        cal.parse_scene_key("76030110x_50_scene_000028_000001_v0.png")
        == "76030110x_50_000028"
    )
    assert cal.parse_scene_key("no_scene_marker.png") == "no_scene_marker.png"

    score = list(cal.CORE_SCORE_THRS)
    mask = list(cal.CORE_MASK_THRS)
    assert cal.edge_dirs((12, 0.03, 0.5), score, mask) == ("lo", None)
    assert cal.edge_dirs((12, 0.2, 0.5), score, mask) == ("hi", None)
    assert cal.edge_dirs((12, 0.05, 0.3), score, mask) == (None, "lo")
    assert cal.edge_dirs((12, 0.05, 0.7), score, mask) == (None, "hi")
    assert cal.edge_dirs((12, 0.1, 0.5), score, mask) == (None, None)

    assert cal.parse_epochs("10-19") == tuple(range(10, 20))
    assert cal.parse_epochs("10,12") == (10, 12)
