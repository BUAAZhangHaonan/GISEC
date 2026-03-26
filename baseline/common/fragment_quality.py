from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

import numpy as np


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _bbox_gap(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = bbox_a
    bx0, by0, bw, bh = bbox_b
    ax1 = ax0 + aw
    ay1 = ay0 + ah
    bx1 = bx0 + bw
    by1 = by0 + bh
    gap_x = max(0, max(bx0 - ax1, ax0 - bx1))
    gap_y = max(0, max(by0 - ay1, ay0 - by1))
    return float(max(gap_x, gap_y))


def _majority_instance_and_purity(mask: np.ndarray, instance_map: np.ndarray | None) -> tuple[int, float]:
    if instance_map is None:
        return 0, 1.0
    values = instance_map[mask]
    values = values[values > 0]
    if values.size == 0:
        return 0, 0.0
    unique, counts = np.unique(values, return_counts=True)
    best_idx = int(counts.argmax())
    return int(unique[best_idx]), float(counts[best_idx]) / float(values.size)


def _contact_pairs(label_map: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        y0 = max(0, -dy)
        y1 = label_map.shape[0] - max(0, dy)
        x0 = max(0, -dx)
        x1 = label_map.shape[1] - max(0, dx)
        src = label_map[y0:y1, x0:x1]
        dst = label_map[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        mask = (src > 0) & (dst > 0) & (src != dst)
        if not mask.any():
            continue
        pair_values = np.stack([src[mask], dst[mask]], axis=1)
        pair_values.sort(axis=1)
        for value in np.unique(pair_values, axis=0):
            pairs.add((int(value[0]), int(value[1])))
    return pairs


def build_fragment_records(label_map: np.ndarray, instance_map: np.ndarray | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    image_area = float(label_map.shape[0] * label_map.shape[1])
    labels = [int(value) for value in np.unique(label_map).tolist() if int(value) > 0]
    for fragment_id in labels:
        mask = label_map == int(fragment_id)
        gt_instance, purity = _majority_instance_and_purity(mask, instance_map)
        bbox = _mask_bbox(mask)
        centroid = _mask_centroid(mask)
        records.append(
            {
                "fragment_id": int(fragment_id),
                "area": int(mask.sum()),
                "area_ratio": float(mask.sum()) / image_area if image_area > 0 else 0.0,
                "bbox": [int(value) for value in bbox],
                "centroid": [float(value) for value in centroid],
                "gt_instance": int(gt_instance),
                "purity": float(purity),
            }
        )
    records.sort(key=lambda row: int(row["fragment_id"]))
    return records


def build_fragment_pair_records(
    label_map: np.ndarray,
    fragment_records: list[dict[str, Any]],
    *,
    max_gap: float = 4.0,
) -> list[dict[str, Any]]:
    contact_pairs = _contact_pairs(label_map)
    records_by_id = {int(row["fragment_id"]): row for row in fragment_records}
    labels = sorted(records_by_id)
    pair_records: list[dict[str, Any]] = []

    def _append_pair(a: int, b: int, *, pair_type: str, bbox_gap: float) -> None:
        row_a = records_by_id[a]
        row_b = records_by_id[b]
        gt_a = int(row_a["gt_instance"])
        gt_b = int(row_b["gt_instance"])
        pair_records.append(
            {
                "fragment_a": int(a),
                "fragment_b": int(b),
                "pair_type": str(pair_type),
                "bbox_gap": float(bbox_gap),
                "gt_same_instance": int(gt_a > 0 and gt_a == gt_b),
                "purity_a": float(row_a["purity"]),
                "purity_b": float(row_b["purity"]),
                "gt_instance_a": gt_a,
                "gt_instance_b": gt_b,
            }
        )

    seen: set[tuple[int, int]] = set()
    for a, b in sorted(contact_pairs):
        seen.add((int(a), int(b)))
        _append_pair(int(a), int(b), pair_type="contact", bbox_gap=0.0)

    for index, a in enumerate(labels):
        bbox_a = tuple(int(value) for value in records_by_id[a]["bbox"])
        for b in labels[index + 1 :]:
            if (int(a), int(b)) in seen:
                continue
            bbox_b = tuple(int(value) for value in records_by_id[b]["bbox"])
            gap = _bbox_gap(bbox_a, bbox_b)
            if gap <= float(max_gap):
                seen.add((int(a), int(b)))
                _append_pair(int(a), int(b), pair_type="bridge", bbox_gap=gap)

    pair_records.sort(key=lambda row: (int(row["fragment_a"]), int(row["fragment_b"])))
    return pair_records


def summarize_fragment_quality(
    fragment_records: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
    *,
    purity_threshold: float = 0.5,
) -> dict[str, Any]:
    purities = [float(row["purity"]) for row in fragment_records]
    by_instance: dict[tuple[int | None, int], list[int]] = defaultdict(list)
    for row in fragment_records:
        gt_instance = int(row["gt_instance"])
        if gt_instance <= 0 or float(row["purity"]) < float(purity_threshold):
            continue
        image_id = None if row.get("image_id") is None else int(row["image_id"])
        by_instance[(image_id, gt_instance)].append(int(row["fragment_id"]))

    same_instance_total_pairs = 0
    for fragment_ids in by_instance.values():
        count = len(fragment_ids)
        if count >= 2:
            same_instance_total_pairs += (count * (count - 1)) // 2

    same_instance_recalled_pairs = 0
    for row in pair_records:
        if int(row["gt_same_instance"]) != 1:
            continue
        if float(row["purity_a"]) < float(purity_threshold) or float(row["purity_b"]) < float(purity_threshold):
            continue
        same_instance_recalled_pairs += 1

    contact_pairs = sum(int(row["pair_type"] == "contact") for row in pair_records)
    bridge_pairs = sum(int(row["pair_type"] == "bridge") for row in pair_records)

    return {
        "fragment_count": int(len(fragment_records)),
        "pair_count": int(len(pair_records)),
        "contact_pair_count": int(contact_pairs),
        "bridge_pair_count": int(bridge_pairs),
        "fragment_purity_mean": 0.0 if not purities else float(sum(purities) / len(purities)),
        "fragment_purity_median": 0.0 if not purities else float(median(purities)),
        "same_instance_total_pairs": int(same_instance_total_pairs),
        "same_instance_recalled_pairs": int(same_instance_recalled_pairs),
        "same_instance_recall": (
            0.0
            if same_instance_total_pairs <= 0
            else float(same_instance_recalled_pairs) / float(same_instance_total_pairs)
        ),
    }
