from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from gisec.eval.coco_eval import evaluate_json
from gisec.eval.coco_export import masks_to_coco_results
from gisec.train.args import (
    EVAL_SCORE_THRESHOLD_DEFAULT,
    parse_eval_args,
    parse_train_args,
)


def _write_minimal_coco_annotations(path) -> None:
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
        "annotations": [],
        "categories": [{"id": 1, "name": "part", "supercategory": "part"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_json_returns_zero_ap_for_empty_results(tmp_path) -> None:
    ann_file = tmp_path / "instances_val.json"
    _write_minimal_coco_annotations(ann_file)

    metrics = evaluate_json(ann_file, [])

    assert metrics["note"] == "no predictions"
    for metric in ("bbox", "segm"):
        for suffix in ("AP", "AP50", "AP75"):
            assert metrics[f"{metric}/{suffix}"] == 0.0


def test_eval_defaults_to_standard_coco_score_threshold() -> None:
    args = parse_eval_args(
        [
            "--dataset-root", "datasets/x",
            "--output-dir", "output/x",
            "--checkpoint", "model_best.pth",
        ]
    )
    assert float(args.eval_score_threshold) == 0.05
    assert float(args.score_threshold) == 0.5


def test_train_epoch_val_shares_the_eval_score_threshold_default() -> None:
    train_args = parse_train_args(
        [
            "--dataset-root", "datasets/x",
            "--output-dir", "output/x",
            "--variant", "base_rgb_1024",
        ]
    )
    eval_args = parse_eval_args(
        [
            "--dataset-root", "datasets/x",
            "--output-dir", "output/x",
            "--checkpoint", "model_best.pth",
        ]
    )

    assert float(train_args.eval_score_threshold) == EVAL_SCORE_THRESHOLD_DEFAULT
    assert float(eval_args.eval_score_threshold) == EVAL_SCORE_THRESHOLD_DEFAULT
    assert float(train_args.score_threshold) == 0.5


def _gt_annotation(ann_id: int, image_id: int, mask, category_id: int = 1):
    from gisec.eval.coco_export import encode_binary_mask

    ys, xs = np.nonzero(mask)
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": encode_binary_mask(mask.astype(np.uint8)),
        "bbox": [
            int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min()), int(ys.max() - ys.min()),
        ],
        "area": float(int(mask.sum())),
        "iscrowd": 0,
    }


class _StubModel:
    def eval(self) -> None:
        pass


def _run_stubbed_evaluate(tmp_path, monkeypatch, *, save_score_threshold):
    """Drive evaluate_gisec over one stubbed baseline-decode image.

    The decode stub returns a fixed candidate set and records the score
    threshold the pipeline handed it; the two ground-truth instances are
    recovered exactly, one candidate at score 0.9 and one at 0.3.
    """
    from gisec.train import evaluate as evaluate_module

    left = np.zeros((8, 8), dtype=np.uint8)
    left[:, :4] = 1
    right = np.zeros((8, 8), dtype=np.uint8)
    right[:, 4:] = 1
    ann_file = tmp_path / "instances_val.json"
    ann_file.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
        "annotations": [
            _gt_annotation(1, 1, left), _gt_annotation(2, 1, right)],
        "categories": [
            {"id": 1, "name": "part", "supercategory": "part"}],
    }), encoding="utf-8")

    candidates = [(left, 0.9), (right, 0.3)]
    decode_thresholds: list[float] = []

    def _fake_run_backbone(**kwargs):
        return SimpleNamespace()

    def _fake_decode(outputs, *, processor, target_size, score_threshold,
                     mask_threshold):
        decode_thresholds.append(float(score_threshold))
        masks = [mask for mask, _ in candidates]
        scores = [score for _, score in candidates]
        return masks, scores

    monkeypatch.setattr(evaluate_module, "run_backbone", _fake_run_backbone)
    monkeypatch.setattr(
        evaluate_module, "outputs_to_instance_masks", _fake_decode)

    (tmp_path / "out").mkdir()
    loader = [[{"image": torch.zeros(1, 8, 8), "image_id": 1}]]
    metrics, _speed = evaluate_module.evaluate_gisec(
        model=_StubModel(),
        loader=loader,
        device=torch.device("cpu"),
        variant_name="base_rgb_1024",
        reference_source=None,
        ann_file=ann_file,
        output_dir=tmp_path / "out",
        score_threshold=0.05,
        mask_threshold=0.5,
        graph_merge_threshold=0.5,
        crop_size=256,
        crop_pad=16,
        boundary_band_width=4,
        max_images=0,
        save_raw=save_score_threshold is not None,
        depth_mode="rgb",
        component_class_index=1,
        save_score_threshold=save_score_threshold,
    )
    return metrics, decode_thresholds, candidates, ann_file


def test_infer_metrics_use_eval_protocol_and_saves_use_score_threshold(
    tmp_path, monkeypatch,
) -> None:
    metrics, decode_thresholds, candidates, ann_file = _run_stubbed_evaluate(
        tmp_path, monkeypatch, save_score_threshold=0.5)

    # Decode ran on the standard eval candidate threshold.
    assert decode_thresholds == [0.05]
    masks = [mask for mask, _ in candidates]
    scores = [score for _, score in candidates]
    full_results = masks_to_coco_results(
        image_id=1, masks=masks, scores=scores, category_id=1)
    kept_results = [full_results[0]]
    # Metrics come from the full candidate set, not the saved subset.
    assert metrics["segm/AP"] == evaluate_json(ann_file, full_results)[
        "segm/AP"]
    assert metrics["segm/AP"] > evaluate_json(ann_file, kept_results)[
        "segm/AP"]
    # Saved predictions keep the --score-threshold semantics.
    saved = json.loads((tmp_path / "out" / "coco_instances_results.json")
                       .read_text(encoding="utf-8"))
    assert [row["score"] for row in saved] == [0.9]
    raw = json.loads(
        (tmp_path / "out" / "coco_instances_results.raw.json")
        .read_text(encoding="utf-8"))
    assert [row["score"] for row in raw["rows"]] == [0.9]


def test_eval_writes_the_full_candidate_set_to_disk(
    tmp_path, monkeypatch,
) -> None:
    metrics, decode_thresholds, _candidates, _ann = _run_stubbed_evaluate(
        tmp_path, monkeypatch, save_score_threshold=None)

    assert decode_thresholds == [0.05]
    assert metrics["segm/AP"] == 1.0
    saved = json.loads((tmp_path / "out" / "coco_instances_results.json")
                       .read_text(encoding="utf-8"))
    assert [row["score"] for row in saved] == [0.9, 0.3]
    assert not (tmp_path / "out" / "coco_instances_results.raw.json").exists()


def test_exported_bbox_matches_pycocotools_rle_bbox() -> None:
    from pycocotools import mask as mask_utils

    shapes = [
        ((16, 16), (slice(3, 9), slice(5, 12))),
        ((8, 8), (slice(0, 8), slice(0, 1))),
        ((20, 10), (slice(7, 8), slice(2, 3))),
    ]
    for size, (ys, xs) in shapes:
        mask = np.zeros(size, dtype=np.uint8)
        mask[ys, xs] = 1

        row = masks_to_coco_results(
            image_id=1, masks=[mask], scores=[0.9], category_id=1)[0]

        rle = mask_utils.encode(np.asfortranarray(mask))
        # toBbox uses the inclusive pixel span (a 1-px mask has width 1),
        # so the exported x1 - x0 + 1 must match it exactly; dropping the
        # +1 would shrink every predicted box by a pixel relative to both
        # pycocotools and the dataset GT boxes.
        assert row["bbox"] == [float(v) for v in mask_utils.toBbox(rle)]
