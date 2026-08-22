"""Judge integration bench: real Dataset+DataLoader, data loop only.

Usage: python integration_bench.py {none,ref,a,b,c}

Clones the E8 CenterDataset recipe (batch 8, num_workers 16,
pin_memory, persistent_workers) over the seed=42 64-image train
subset, swapping only the heatmap synthesis. Reports per-epoch
s/step and batches/s (epoch 1 = cold, epochs 2-3 = warm; C's
self-warming cache makes epoch 2+ skip rasterization). A
num_workers=0 correctness pass first checks each impl's heatmap
against the reference path bit-for-bit, and a final check confirms
multi-worker epoch-3 output equals the single-process reference.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "datasets" / "20260318_1K_32254"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "ugnn" / "exp06_center_split"))
sys.path.insert(0, str(REPO / "experiments" / "ugnn" / "heatmap_colosseum"))

from train_center import make_heatmap as ref_make_heatmap  # noqa: E402

from gisec.datasets.coco_utils import (  # noqa: E402
    LiteCOCO,
    ann_to_mask,
    load_depth_array,
)

DEPTH_LO, DEPTH_HI = 0.3, 0.7  # E3/E8 constants


def load_impl(name: str):
    """Import <team>/solution.py as top-level ``solution`` (team_c's
    numba JIT cache pickles the env under that module name)."""
    name = {"a": "team_a", "b": "team_b", "c": "team_c"}.get(name, name)
    if name in ("ref", "none"):
        return None
    team_dir = str(Path(__file__).resolve().parent.parent / name)
    sys.path.insert(0, team_dir)
    mod = importlib.import_module("solution")
    if name != "team_c":
        del sys.modules["solution"]
    if name == "team_a":
        mod.init_cache()
    return mod


class BenchDataset(Dataset):
    """E8 CenterDataset clone over a fixed 64-image subset."""

    def __init__(self, impl: str, img_infos: list[dict], coco):
        self.impl = impl
        self.mod = load_impl(impl)  # fork-inherited by workers
        self.coco = coco
        self.img_dir = DATA / "images" / "train"
        self.depth_dir = DATA / "depth" / "depth_npy" / "train"
        self.infos = img_infos

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx: int):
        info = self.infos[idx]
        stem = info["file_name"].rsplit(".", 1)[0]
        img = cv2.imread(str(self.img_dir / info["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_array(self.depth_dir / f"{stem}.npy")
        depth = np.clip((depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), -1.0, 2.0)
        x = np.concatenate(
            [img.astype(np.float32) / 255.0, depth[..., None].astype(np.float32)],
            axis=-1,
        )
        h, w = info["height"], info["width"]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[info["id"]]))
        gt = np.zeros((h, w), dtype=np.float32)
        insts = []
        for ann in anns:
            m = ann_to_mask(ann, h, w)
            if m.sum() <= 0:
                continue
            insts.append(m)
            gt[m > 0] = 1.0
        if self.impl == "none":
            hm = np.zeros((h, w), dtype=np.float32)
        elif self.impl == "ref":
            hm = ref_make_heatmap(insts, h, w)
        else:
            hm = self.mod.build_heatmap(anns, (h, w))
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.stack([gt, hm]))
        return x, y


def ref_hms(coco, infos):
    """Single-process reference heatmaps keyed by image id."""
    out = {}
    for info in infos:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[info["id"]]))
        insts = [
            m
            for m in (ann_to_mask(a, info["height"], info["width"]) for a in anns)
            if m.sum() > 0
        ]
        out[info["id"]] = ref_make_heatmap(insts, info["height"], info["width"])
    return out


def main() -> None:
    impl = sys.argv[1]
    coco = LiteCOCO(DATA / "annotations" / "instances_train.json")
    ids = coco.getImgIds()
    rng = np.random.default_rng(42)
    sel = rng.choice(len(ids), size=64, replace=False)
    infos = [coco.loadImgs([ids[i]])[0] for i in sel]

    if impl != "none":
        refs = ref_hms(coco, infos)
        # single-process correctness
        ds0 = BenchDataset(impl, infos, coco)
        maxd = 0.0
        for i in range(len(ds0)):
            _, y = ds0[i]
            maxd = max(maxd, float(np.abs(y[1].numpy() - refs[infos[i]["id"]]).max()))
        print(f"single-process max|delta| vs ref: {maxd:.3e}")

    ds = BenchDataset(impl, infos, coco)
    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    for epoch in range(1, 4):
        t0 = time.perf_counter()
        nb = 0
        hms = {}
        for x, y in dl:
            nb += 1
        dt = time.perf_counter() - t0
        print(
            f"epoch{epoch}: {dt:.2f} s, {nb} batches, "
            f"{dt / nb * 1e3:.1f} ms/step, {nb / dt:.2f} batches/s"
        )

    # multi-worker output vs single-process reference
    if impl != "none":
        ds_ck = BenchDataset(impl, infos, coco)
        dl_ck = DataLoader(ds_ck, batch_size=8, shuffle=False, num_workers=16)
        maxd = 0.0
        i = 0
        for _, y in dl_ck:
            for k in range(y.shape[0]):
                maxd = max(
                    maxd, float(np.abs(y[k, 1].numpy() - refs[infos[i]["id"]]).max())
                )
                i += 1
        print(f"multi-worker max|delta| vs ref: {maxd:.3e}")


if __name__ == "__main__":
    main()
