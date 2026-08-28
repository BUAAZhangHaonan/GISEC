"""Calibration + report for the 16M-baseline retrain queue (expert
round-2 protocol, 2026-08-28).  No training happens here.

calibrate
    Joint (epoch, score_thr, mask_thr) selection on the FROZEN
    500-image calibration set - the same 500 images as the E20 sweep;
    image ids + file names are read read-only from
    exp20_band8/decode_fix/_cache_fwd/metas.json.  Candidates are the
    epoch_10..19 checkpoints saved by train.py.  Selection is
    scene-disjoint (calibration half picks, gating half only scores).

    Pre-registered selection rule (frozen before any run):
      R1  Scenes (part+scene key parsed from the file name, same key
          as lib/eval_scale.scene_key) are split ONCE, seed 0, into
          disjoint calibration / gating halves.
      R2  The winner is the argmax segm AP on the FULL calibration
          half (unit multiplicity), jointly over
            epoch {10..19} x score {0.03,0.05,0.1,0.2}
            x mask {0.3,0.4,0.5,0.6,0.7}.
          The gating half never enters any selection.
      R3  Cross-fit gate estimate (winner's-curse repair, same
          estimator family as lib/scene_boot.cross_fit_threshold):
          2000 draws; each draw resamples scenes within each half
          independently, re-picks the argmax on the calibration
          replicate, and scores ONLY that pick on the gating replicate.
      R4  Edge extension: if the R2 winner sits on a boundary of the
          score grid or the mask grid, that grid is extended ONE step
          in the violated direction (score -> 0.02 or 0.3, mask ->
          0.2 or 0.8; the extension values are decoded in the same
          forward pass, so extension never reruns inference) and
          R2+R3 are re-run over the extended grid.  Exactly one
          extension round - a winner landing on the extended edge
          stays there.  The epoch grid {10..19} is hard-bounded by
          the checkpoint protocol (train.py saves exactly these) and
          is NEVER extended; an epoch-edge winner is recorded with
          epoch_at_edge=true.
      R5  Output run_dir/calibration.json (winner, calib-half AP,
          gate-half point AP, cross-fit gate distribution, pick
          histogram, extension record).  The queue freezes this
          winner for the full-3276 eval.

report
    Multiplicity-aware paired scene bootstrap (lib/scene_boot,
    2000 draws) of the arm's full-val predictions against the E20
    canonical predictions (export_e20_results.py output), plus the
    RESULT.md template row.  CPU-only; requires
    baselines16m/e20_fullval_results.json (see export_e20_results.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
UGNN = HERE.parent
REPO = UGNN.parents[1]
for _p in (HERE, UGNN / "lib", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_models import build_model  # noqa: E402
from common import (  # noqa: E402
    DATA,
    FAMILIES,
    Baseline16mDataset,
    collate_m2f,
    collate_mrcnn,
    family_data_flags,
)
from eval import foreground_keep, masks_to_rle  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from scene_boot import ApWeighted, SceneResampler, paired_scene_bootstrap  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

FROZEN_METAS = UGNN / "exp20_band8" / "decode_fix" / "_cache_fwd" / "metas.json"
DEFAULT_EPOCHS = tuple(range(10, 20))
CORE_SCORE_THRS = (0.03, 0.05, 0.1, 0.2)
CORE_MASK_THRS = (0.3, 0.4, 0.5, 0.6, 0.7)
EXT_SCORE_THRS = {"lo": 0.02, "hi": 0.3}
EXT_MASK_THRS = {"lo": 0.2, "hi": 0.8}
E20_CANONICAL_AP = 0.84880
SEED = 0
N_BOOT = 2000


def parse_scene_key(file_name: str) -> str:
    """part+scene cluster key, identical to lib/eval_scale.scene_key
    (kept local so this script does not drag eval_scale's imports)."""
    m = re.match(r"(.+?)_scene_(\d+)_", file_name)
    return f"{m.group(1)}_{m.group(2)}" if m else file_name


def parse_epochs(spec: str) -> tuple[int, ...]:
    epochs: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-"))
            epochs.extend(range(lo, hi + 1))
        else:
            epochs.append(int(part))
    if not epochs or any(e not in DEFAULT_EPOCHS for e in epochs):
        raise SystemExit(f"--epochs must subset {DEFAULT_EPOCHS}, got {spec!r}")
    return tuple(epochs)


def scene_subset_mult(resampler: SceneResampler, scenes: np.ndarray) -> np.ndarray:
    """Unit multiplicity on the images of the given scenes, 0 elsewhere."""
    mult = np.zeros(resampler.n_scenes, dtype=np.int64)
    mult[scenes] = 1
    return mult[resampler.scene_of_img]


@torch.no_grad()
def decode_epoch(
    model: torch.nn.Module,
    family: str,
    loader: DataLoader,
    device: str,
    image_ids: list[int],
    score_floor: float,
    mask_thrs: tuple[float, ...],
) -> dict[int, dict]:
    """One forward per image; per image keep every instance with
    score > score_floor and pre-encode its binary mask RLE at every
    mask_thr, so all (score, mask) combos decode without rerunning
    inference."""
    model.eval()
    store: dict[int, dict] = {}
    pos = 0  # loader has shuffle=False; emit-style alignment (eval.py)
    is_mrcnn = family.startswith("mrcnn16")
    for batch in loader:
        if is_mrcnn:
            images = [im.to(device) for im in batch[0]]
            outputs = model(images)
            for out in outputs:
                keep = out["scores"] > score_floor
                scores = out["scores"][keep].detach().cpu().numpy()
                probs = out["masks"][keep, 0]
                store[image_ids[pos]] = _rle_per_thr(probs, mask_thrs, scores)
                pos += 1
        else:
            pixel_values = batch[0].to(device)
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            class_probs = F.softmax(outputs.class_queries_logits, dim=-1)
            scores_t, labels_t = class_probs.max(-1)
            mask_logits = outputs.masks_queries_logits
            for i in range(pixel_values.shape[0]):
                keep = foreground_keep(labels_t[i], scores_t[i], score_floor)
                idx = torch.nonzero(keep).flatten()
                if idx.numel() == 0:
                    store[image_ids[pos]] = {
                        "scores": np.zeros(0, np.float32),
                        "rles": {m: [] for m in mask_thrs},
                    }
                    pos += 1
                    continue
                probs = F.interpolate(
                    mask_logits[i : i + 1, idx].sigmoid(),
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                scores = scores_t[i][idx].detach().cpu().numpy()
                store[image_ids[pos]] = _rle_per_thr(probs, mask_thrs, scores)
                pos += 1
    return store


def _rle_per_thr(
    probs: torch.Tensor, mask_thrs: tuple[float, ...], scores: np.ndarray
) -> dict:
    rles = {m: [] for m in mask_thrs}
    if scores.shape[0]:
        for m in mask_thrs:
            masks = (probs > m).to(torch.uint8).cpu().numpy()
            rles[m] = masks_to_rle(masks)
            del masks
    return {"scores": scores.astype(np.float32), "rles": rles}


def build_results(
    store: dict[int, dict],
    image_ids: list[int],
    score_thr: float,
    mask_thr: float,
    category_id: int,
) -> list[dict]:
    results = []
    for image_id in image_ids:
        entry = store[image_id]
        for score, rle in zip(entry["scores"], entry["rles"][mask_thr], strict=True):
            if score > score_thr:
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": rle,
                        "score": float(score),
                    }
                )
    return results


def accumulators_for_epoch(
    coco_gt: COCO,
    store: dict[int, dict],
    epoch: int,
    image_ids: list[int],
    score_thrs: tuple[float, ...],
    mask_thrs: tuple[float, ...],
    category_id: int,
) -> dict[tuple[int, float, float], ApWeighted | None]:
    accs: dict[tuple[int, float, float], ApWeighted | None] = {}
    for score_thr in score_thrs:
        for mask_thr in mask_thrs:
            results = build_results(store, image_ids, score_thr, mask_thr, category_id)
            # pycocotools loadRes cannot digest an empty list; a combo
            # with zero predictions has AP exactly 0 at any multiplicity.
            accs[(epoch, score_thr, mask_thr)] = (
                ApWeighted(coco_gt, coco_gt.loadRes(results), image_ids, "segm")
                if results
                else None
            )
    return accs


def ap_of(accs: dict, combo: tuple[int, float, float], mult) -> float:
    acc = accs[combo]
    return acc.ap(mult) if acc is not None else 0.0


def edge_dirs(combo: tuple[int, float, float], score_grid, mask_grid):
    """Pre-registered R4 directions: 'lo'/'hi' when the winner sits on
    that boundary of a core grid, else None."""
    _e, s, m = combo
    s_dir = "lo" if s == score_grid[0] else "hi" if s == score_grid[-1] else None
    m_dir = "lo" if m == mask_grid[0] else "hi" if m == mask_grid[-1] else None
    return s_dir, m_dir


def run_calibrate(args: argparse.Namespace) -> None:
    epochs = parse_epochs(args.epochs)
    score_grid = list(CORE_SCORE_THRS)
    mask_grid = list(CORE_MASK_THRS)
    # extension values ride along in the decode pass (R4: extension
    # never reruns inference)
    decode_scores = sorted(set(score_grid) | set(EXT_SCORE_THRS.values()))
    decode_masks = sorted(set(mask_grid) | set(EXT_MASK_THRS.values()))

    metas = json.loads(Path(args.metas).read_text())
    file_name_of = {int(m["image_id"]): m["file_name"] for m in metas}
    image_ids = sorted(file_name_of)
    if args.max_images:
        image_ids = image_ids[: args.max_images]
    scene_keys = [parse_scene_key(file_name_of[i]) for i in image_ids]
    resampler = SceneResampler(image_ids, scene_keys)
    print(
        f"frozen set: {len(image_ids)} images / {resampler.n_scenes} scenes "
        f"from {args.metas}",
        flush=True,
    )

    include_depth, imagenet_norm = family_data_flags(args.family)
    dataset = Baseline16mDataset(
        "val", include_depth=include_depth, imagenet_norm=imagenet_norm
    )
    missing = [i for i in image_ids if i not in set(dataset.image_ids)]
    if missing:
        raise SystemExit(f"calibration ids missing from val split: {missing[:5]}")
    dataset.image_ids = image_ids
    collate = collate_mrcnn if args.family.startswith("mrcnn16") else collate_m2f
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
    )
    coco_gt = COCO(str(DATA / "annotations" / "instances_val.json"))
    category_id = int(dataset.coco.categories[0]["id"])

    score_floor = min(decode_scores)
    accs: dict[tuple[int, float, float], ApWeighted | None] = {}
    for epoch in epochs:
        ckpt = Path(args.run_dir) / f"epoch_{epoch}.pth"
        if not ckpt.exists():
            raise SystemExit(f"missing calibration checkpoint {ckpt}")
        model = build_model(args.family)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.to("cuda")
        t0 = time.time()
        store = decode_epoch(
            model,
            args.family,
            loader,
            "cuda",
            image_ids,
            score_floor,
            tuple(decode_masks),
        )
        accs.update(
            accumulators_for_epoch(
                coco_gt,
                store,
                epoch,
                image_ids,
                tuple(decode_scores),
                tuple(decode_masks),
                category_id,
            )
        )
        del store, model
        print(
            f"epoch {epoch}: decoded {len(image_ids)} imgs, "
            f"score floor {score_floor} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # R1: one fixed scene split, seed 0
    rng0 = np.random.default_rng(SEED)
    perm = rng0.permutation(resampler.n_scenes)
    half = resampler.n_scenes // 2
    calib_scenes, gate_scenes = perm[:half], perm[half:]
    calib_unit = scene_subset_mult(resampler, calib_scenes)
    gate_unit = scene_subset_mult(resampler, gate_scenes)

    def pick(combos):
        return max(combos, key=lambda c: ap_of(accs, c, calib_unit))

    def cross_fit(combos):
        rng = np.random.default_rng(SEED + 1)
        gate_aps = np.empty(args.n_boot)
        picks = []
        for d in range(args.n_boot):
            cm = resampler.draw(rng, calib_scenes)
            gm = resampler.draw(rng, gate_scenes)
            star = max(combos, key=lambda c: ap_of(accs, c, cm))
            gate_aps[d] = ap_of(accs, star, gm)
            picks.append(star)
        hist: dict[str, int] = {}
        for c in picks:
            key = f"{c[0]}|{c[1]}|{c[2]}"
            hist[key] = hist.get(key, 0) + 1
        return {
            "gate_ap": {
                "mean": float(gate_aps.mean()),
                "ci95": [
                    float(np.percentile(gate_aps, 2.5)),
                    float(np.percentile(gate_aps, 97.5)),
                ],
            },
            "pick_hist": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:10]),
        }

    # R2 on the core grids
    core = [(e, s, m) for e in epochs for s in score_grid for m in mask_grid]
    winner_core = pick(core)
    # R4: one extension round in the violated directions only
    s_dir, m_dir = edge_dirs(winner_core, score_grid, mask_grid)
    ext_score = sorted(set(score_grid) | ({EXT_SCORE_THRS[s_dir]} if s_dir else set()))
    ext_mask = sorted(set(mask_grid) | ({EXT_MASK_THRS[m_dir]} if m_dir else set()))
    final_combos = [(e, s, m) for e in epochs for s in ext_score for m in ext_mask]
    winner = pick(final_combos) if (s_dir or m_dir) else winner_core
    s_dir2, m_dir2 = edge_dirs(winner, ext_score, ext_mask)
    report = {
        "family": args.family,
        "run_dir": str(args.run_dir),
        "metas": str(args.metas),
        "n_images": len(image_ids),
        "n_scenes": resampler.n_scenes,
        "n_scenes_calib": len(calib_scenes),
        "n_scenes_gate": int(resampler.n_scenes - len(calib_scenes)),
        "seed": SEED,
        "n_boot": args.n_boot,
        "epochs": list(epochs),
        "score_grid": ext_score,
        "mask_grid": ext_mask,
        "winner": {
            "epoch": winner[0],
            "score_thr": winner[1],
            "mask_thr": winner[2],
            "calib_ap": ap_of(accs, winner, calib_unit),
            "gate_ap": ap_of(accs, winner, gate_unit),
        },
        "epoch_at_edge": winner[0] in (epochs[0], epochs[-1]),
        "edge_extension": {
            "applied": bool(s_dir or m_dir),
            "score_dir": s_dir,
            "mask_dir": m_dir,
            "winner_before": {
                "epoch": winner_core[0],
                "score_thr": winner_core[1],
                "mask_thr": winner_core[2],
                "calib_ap": ap_of(accs, winner_core, calib_unit),
            },
        },
        "winner_on_extended_edge": bool(s_dir2 or m_dir2),
        "cross_fit": cross_fit(final_combos),
    }
    out = Path(args.run_dir) / "calibration.json"
    out.write_text(json.dumps(report, indent=2))
    w = report["winner"]
    print(
        f"winner ep{w['epoch']} score {w['score_thr']} mask {w['mask_thr']}: "
        f"calib AP {w['calib_ap']:.5f} / gate AP {w['gate_ap']:.5f}; "
        f"cross-fit gate {report['cross_fit']['gate_ap']['mean']:.5f} "
        f"CI95 {report['cross_fit']['gate_ap']['ci95']} -> {out}",
        flush=True,
    )


def run_report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    base_results = json.loads((run_dir / "coco_instances_results.json").read_text())
    e20_results = json.loads(Path(args.e20_results).read_text())
    coco_gt = COCO(str(DATA / "annotations" / "instances_val.json"))
    img_ids = sorted(int(i) for i in coco_gt.getImgIds())
    file_name_of = {int(im["id"]): im["file_name"] for im in coco_gt.loadImgs(img_ids)}
    resampler = SceneResampler(
        img_ids, [parse_scene_key(file_name_of[i]) for i in img_ids]
    )
    acc_base = ApWeighted(coco_gt, coco_gt.loadRes(base_results), img_ids, "segm")
    acc_e20 = ApWeighted(coco_gt, coco_gt.loadRes(e20_results), img_ids, "segm")
    paired = paired_scene_bootstrap(
        acc_e20, acc_base, resampler, n_boot=args.n_boot, seed=SEED
    )
    params = None
    hist_path = run_dir / "history.jsonl"
    if hist_path.exists():
        for line in hist_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("event") == "start":
                params = rec.get("params")
                break
    out = {
        "family": args.family,
        "run_dir": str(run_dir),
        "e20_results": str(args.e20_results),
        "n_images": len(img_ids),
        "n_scenes": resampler.n_scenes,
        "n_boot": args.n_boot,
        "seed": SEED,
        "params": params,
        "point_ap": {
            "e20": acc_e20.ap(resampler.unit()),
            "baseline": acc_base.ap(resampler.unit()),
            "e20_canonical_reference": E20_CANONICAL_AP,
        },
        "paired_e20_minus_baseline": paired,
    }
    (run_dir / "paired_vs_e20.json").write_text(json.dumps(out, indent=2))
    d = paired["delta"]
    calib = run_dir / "calibration.json"
    winner = json.loads(calib.read_text())["winner"] if calib.exists() else None
    cfg = (
        f"ep{winner['epoch']} score {winner['score_thr']} mask {winner['mask_thr']}"
        if winner
        else f"score {metrics.get('score_thr')} mask {metrics.get('mask_thr')}"
    )
    params_s = f"{params / 1e6:.2f}M" if params else "n/a"
    row = (
        f"- {run_dir.name} (family {args.family}, {params_s}, {cfg}): "
        f"segm AP {metrics['segm/AP']:.4f} AP50 {metrics['segm/AP50']:.4f} "
        f"AP75 {metrics['segm/AP75']:.4f} | bbox AP {metrics['bbox/AP']:.4f} | "
        f"paired E20-minus-this {d['mean']:+.4f} CI95 "
        f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
        f"(scene bootstrap, {args.n_boot} draws, seed {SEED})"
    )
    print(row, flush=True)
    print(f"-> {run_dir / 'paired_vs_e20.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="joint (epoch,score,mask) selection")
    p_cal.add_argument("--family", required=True, choices=FAMILIES)
    p_cal.add_argument("--run-dir", required=True)
    p_cal.add_argument("--metas", default=str(FROZEN_METAS))
    p_cal.add_argument("--epochs", default="10-19")
    p_cal.add_argument("--max-images", type=int, default=0)
    p_cal.add_argument("--batch-size", type=int, default=8)
    p_cal.add_argument("--workers", type=int, default=8)
    p_cal.add_argument("--n-boot", type=int, default=N_BOOT)
    p_cal.set_defaults(func=run_calibrate)

    p_rep = sub.add_parser("report", help="paired bootstrap vs E20 + result row")
    p_rep.add_argument("--family", required=True, choices=FAMILIES)
    p_rep.add_argument("--run-dir", required=True)
    p_rep.add_argument("--e20-results", required=True)
    p_rep.add_argument("--n-boot", type=int, default=N_BOOT)
    p_rep.set_defaults(func=run_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
