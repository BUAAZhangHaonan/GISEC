"""Deployment-metric monitor inside training (E25 monitor design).

Same call contract as the canonical eval chain (gisec.inference
forward + gisec.decode markers + gisec.postproc_fast.process with
the shared rank/RGB caches), EMA weights, frozen first-N val
images, segm AP at SEM_THR 0.90/0.95 via evaluate_json(img_ids),
plus overlay PNGs (predictions colored, GT contours in white).
Read-only w.r.t. caches; ~5 min for N=500 every 8K steps.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch

_EVAL_STATE: dict = {}
_VIZ_PALETTE = [
    (255, 60, 60),
    (60, 220, 60),
    (60, 120, 255),
    (255, 220, 40),
    (255, 60, 220),
    (40, 220, 220),
    (200, 130, 40),
    (130, 60, 200),
] * 8


def deploy_eval(model_eval, runs_dir, eval_imgs, viz_imgs, tag, engine="auto"):
    """One monitor pass; returns the jsonl row (dict).

    engine="gpu" routes forward+decode+rank through the gpu_fast
    pipeline (bitwise-equal by construction, ~2x faster per pass) —
    valid in-process because training already holds the CUDA context;
    engine="cpu" is the historical forkless path (identical outputs);
    "auto" (default) picks gpu when CUDA is up, else cpu."""
    st = _EVAL_STATE
    if "dec" not in st:
        from pathlib import Path

        from pycocotools.coco import COCO

        from gisec import decode as dec
        from gisec import inference as inf
        from gisec import postproc_fast as ppf
        from gisec.datasets.coco_utils import ann_to_mask, load_depth_array
        from gisec.datasets.split import DATA, load_split
        from gisec.eval.coco_eval import evaluate_json

        inf.load_rgb_index()
        inf._gpu_divisors()
        metas_all, _ = load_split("val")
        st.update(
            dec=dec,
            inf=inf,
            ppf=ppf,
            metas_all=metas_all,
            ej=evaluate_json,
            ann=DATA / "annotations" / "instances_val.json",
            load_depth=load_depth_array,
            ann_to_mask=ann_to_mask,
            COCO=COCO,
            Path=Path,
        )
    dec, inf, ppf = st["dec"], st["inf"], st["ppf"]
    if engine == "auto":
        from gisec import gpu_pipeline as _gpl

        engine = "gpu" if _gpl._backend() is not False else "cpu"
    if engine == "gpu":
        from gisec import gpu_pipeline as gpipeline

        if "gp" not in st:
            st["gp"] = gpipeline.GpuPipeline(model_eval)
    else:
        st["gp"] = None
    gp = st["gp"]
    metas = st["metas_all"][:eval_imgs]
    thrs = (0.90, 0.95)
    results = {t: [] for t in thrs}
    viz_store = {}
    t0 = time.time()
    model_eval.eval()
    with torch.no_grad():
        for meta in metas:
            img = inf.load_rgb_cached(meta)
            depth = st["load_depth"](st["Path"](meta["dpath"]))
            if gp is not None:
                # one GPU stage per threshold (forward shared cost is
                # small at 1024; sem/rank/markers are threshold-bound)
                payload = None
                for t in thrs:
                    payload = gp.gpu_stage(img, depth, t)
                    _, coco = gpipeline.cpu_stage(payload, meta["image_id"])
                    results[t].extend(coco)
                if len(viz_store) < viz_imgs:
                    viz_store[meta["image_id"]] = (
                        "gpu",
                        img,
                        depth,
                        payload,
                        None,
                        meta,
                    )
                continue
            sem_logit, hm, off = inf._forward(model_eval, img, depth)
            coords, cells = dec._cn_markers_with_cells(hm, off)
            peaks = dec._marker_peaks(hm, coords, cells)
            for t in thrs:
                sem = dec.sem_binary(sem_logit, t)
                _, coco = ppf.process(
                    meta["image_id"], coords, sem, depth, sem_logit, peaks
                )
                results[t].extend(coco)
            if len(viz_store) < viz_imgs:
                viz_store[meta["image_id"]] = (
                    "cpu",
                    img,
                    depth,
                    sem_logit,
                    coords,
                    meta,
                )
    ids = [mm["image_id"] for mm in metas]
    out = {
        "event": "deploy_eval",
        "tag": tag,
        "n": len(ids),
        "sec": round(time.time() - t0, 1),
        "engine": engine,
    }
    for t in thrs:
        r = st["ej"](st["ann"], results[t], img_ids=ids)
        out[f"segm_AP@{t:.2f}"] = round(float(r.get("segm/AP", 0.0)), 5)

    vdir = runs_dir / "visualizations"
    vdir.mkdir(exist_ok=True)
    if st.get("_coco_val") is None:
        st["_coco_val"] = st["COCO"](str(st["ann"]))
    coco = st["_coco_val"]
    for iid, (mode, img, depth, sem_logit, coords, _meta) in viz_store.items():
        if mode == "gpu":
            insts, _ = gpipeline.cpu_stage(sem_logit, iid)  # the 0.95 payload
        else:
            sem = dec.sem_binary(sem_logit, 0.95)
            # insts (uncapped instance masks) is peak-independent; the discarded
            # COCO dicts from this call are not used for scoring.
            insts, _ = ppf.process(
                iid,
                coords,
                sem,
                depth,
                sem_logit,
                np.zeros(max(len(coords), 1)),
            )
        canvas = img.astype(np.float32)
        for k, (mask, _a) in enumerate(insts):
            c = np.array(_VIZ_PALETTE[k % len(_VIZ_PALETTE)], dtype=np.float32)
            canvas[mask] = 0.45 * canvas[mask] + 0.55 * c
        H, W = img.shape[:2]
        ann_ids = coco.getAnnIds(imgIds=[iid])
        for a in coco.loadAnns(ann_ids):
            gtm = st["ann_to_mask"](a, H, W).astype(np.uint8)
            cnts, _ = cv2.findContours(gtm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
        cv2.imwrite(
            str(vdir / f"{tag}_id{iid}.png"),
            cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
    return out
