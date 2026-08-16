from __future__ import annotations

import json

from gisec.eval.coco_eval import evaluate_json
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
    results_file = tmp_path / "coco_instances_results.json"
    results_file.write_text("[]", encoding="utf-8")

    metrics = evaluate_json(ann_file, results_file)

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
