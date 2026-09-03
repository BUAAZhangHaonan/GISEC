"""Training dataset over the precomputed GT records (E9b + E17 + E24).

``CNDataset`` reads, per split:
  - exp09 GT records: {split}_items.pkl / _stats.pkl / _sem.dat
    (compact per-image stats + packbits union semantic)
  - the E17 band record: {split}_band.dat (packbits boundary band)
  - optionally the E24 projected-anchor records:
    {split}_projanchor.pkl (in-mask projections p*), which replace
    the (fy, fx) columns of the stats slice before target stamping

anchor="centroid" (default) is bitwise E20; anchor="projected" is
the E24/E25 canonical. Input assembly: RGB [0,1] + globally
calibrated depth channel (DEPTH_LO/HI fixed across the dataset).
"""

from __future__ import annotations

import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gisec.datasets.coco_utils import load_depth_array
from gisec.paths import BAND_RECORDS, DATA_ROOT, GT_RECORDS, PROJANCHOR_RECORDS
from gisec.targets import STRIDE, build_seed_targets_from_stats

DEPTH_LO, DEPTH_HI = 0.245, 0.686
SIDE = 1024
PACK = SIDE * SIDE // 8


class CNDataset(Dataset):
    """E20 CNDataset + optional projected-anchor stats injection.

    anchor modes (E24 anchor-ablation ladder, 2026-09-02 review):
      centroid (default) -- the exp09 stats stream passes through
        untouched: the original float arithmetic centroid (bitwise E20).
      projected -- columns (fy, fx) are replaced with the precomputed
        discrete in-mask projections p* for EVERY instance (bitwise
        E24/E25; support-domain-constrained discrete anchor).
      invproj -- p* only for the instances whose rounded centroid
        falls outside the mask; in-mask instances keep their float
        centroid (isolates "fix invalid anchors" from "discretize all
        anchors"; synthesized from the same projanchor.pkl, zero new
        precompute).
    """

    def __init__(self, split: str, anchor: str = "centroid", hit_counter=None) -> None:
        if anchor not in ("centroid", "projected", "invproj"):
            raise ValueError(f"unknown anchor mode: {anchor!r}")
        self.split = split
        self.anchor = anchor
        self.hit_counter = hit_counter
        rec = GT_RECORDS
        band = BAND_RECORDS / f"{split}_band.dat"
        if not (rec / f"{split}_items.pkl").exists():
            raise FileNotFoundError(
                f"{rec}/{split}_items.pkl missing; run "
                "gisec.datasets.build_gt_records once"
            )
        if not band.exists():
            raise FileNotFoundError(
                f"{band} missing; run gisec.datasets.build_band_records once"
            )
        with open(rec / f"{split}_items.pkl", "rb") as f:
            self.items = pickle.load(f)
        with open(rec / f"{split}_stats.pkl", "rb") as f:
            ids, self.offsets, self.flat = pickle.load(f)
        assert list(ids) == [i for i, _ in self.items]
        self.sem = np.memmap(
            rec / f"{split}_sem.dat",
            dtype=np.uint8,
            mode="r",
            shape=(len(self.items), PACK),
        )
        self.band = np.memmap(
            band, dtype=np.uint8, mode="r", shape=(len(self.items), PACK)
        )
        self.img_dir = DATA_ROOT / "images" / split
        self.depth_dir = DATA_ROOT / "depth" / "depth_npy" / split
        self.ids = [i for i, _ in self.items]
        self.proj = None
        self.inside = None
        self.cell_moved = None
        if anchor in ("projected", "invproj"):
            paf = PROJANCHOR_RECORDS / f"{split}_projanchor.pkl"
            if not paf.exists():
                raise FileNotFoundError(
                    f"{paf} missing; run gisec.datasets.build_proj_anchor_records once"
                )
            with open(paf, "rb") as f:
                pa = pickle.load(f)
            assert list(pa["ids"]) == self.ids, "proj/items id order mismatch"
            assert np.array_equal(pa["offsets"], self.offsets), (
                "proj/stats offsets mismatch"
            )
            self.inside = pa["inside"]
            if anchor == "projected":
                self.proj = pa["proj"]
                cell = np.floor(pa["cent"] / STRIDE + 0.5).astype(np.int64)
                cell_p = np.floor(pa["proj"] / STRIDE + 0.5).astype(np.int64)
                self.cell_moved = (cell[:, 0] != cell_p[:, 0]) | (
                    cell[:, 1] != cell_p[:, 1]
                )
            else:  # invproj: keep the float centroid where it is valid
                self.proj = np.where(pa["inside"][:, None], pa["cent"], pa["proj"])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        _img_id, file_name = self.items[idx]
        img = cv2.imread(str(self.img_dir / file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_array(self.depth_dir / f"{file_name.rsplit('.', 1)[0]}.npy")
        depth = np.clip((depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), -1.0, 2.0)
        x = np.concatenate(
            [img.astype(np.float32) / 255.0, depth[..., None].astype(np.float32)],
            axis=-1,
        )
        gt = np.unpackbits(self.sem[idx]).astype(np.float32).reshape(SIDE, SIDE)
        bd = np.unpackbits(self.band[idx]).astype(np.float32).reshape(SIDE, SIDE)
        stats = self.flat[self.offsets[idx] : self.offsets[idx + 1]]
        if self.anchor in ("projected", "invproj"):
            o0, o1 = int(self.offsets[idx]), int(self.offsets[idx + 1])
            stats = stats.copy()
            stats[:, 0] = self.proj[o0:o1, 0]
            stats[:, 1] = self.proj[o0:o1, 1]
            if self.hit_counter is not None:
                with self.hit_counter.get_lock():
                    self.hit_counter.value += int((~self.inside[o0:o1]).sum())
        hm, off = build_seed_targets_from_stats(stats)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y_sem = torch.from_numpy(gt)
        y_band = torch.from_numpy(bd)
        y_seed = torch.from_numpy(np.concatenate([hm[None], off]))
        return x, y_sem, y_seed, y_band
