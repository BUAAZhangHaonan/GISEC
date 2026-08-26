"""E19 flip correctness check (data-side, no GPU).

For 2 val samples: fetch the same item with epoch set so flip=False and
flip=True, then assert bit-exact mirror consistency:
  - x (4ch), sem, band, hm flipped == np.flip(W) of unflipped
  - off_y (ch0) flipped == np.flip of unflipped
  - off_x (ch1) flipped == -np.flip of unflipped
Prints numbers, exits nonzero on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_hflip import CNDataset  # noqa: E402


def main() -> None:
    ds = CNDataset("val")
    for idx in (0, 1):
        # flip when (epoch + idx) % 2 == 0
        ep_no = 0 if idx % 2 == 1 else 1
        ep_yes = 1 - ep_no
        assert (ep_no + idx) % 2 == 1 and (ep_yes + idx) % 2 == 0
        ds.epoch.value = ep_no
        x0, sem0, seed0, band0 = ds[idx]
        ds.epoch.value = ep_yes
        x1, sem1, seed1, band1 = ds[idx]
        hm0, off_y0, off_x0 = seed0.numpy()
        hm1, off_y1, off_x1 = seed1.numpy()
        checks = {
            "x": np.array_equal(x1.numpy(), np.flip(x0.numpy(), axis=2)),
            "sem": np.array_equal(sem1.numpy(), np.flip(sem0.numpy(), axis=1)),
            "band": np.array_equal(band1.numpy(), np.flip(band0.numpy(), axis=1)),
            "hm": np.array_equal(hm1, np.flip(hm0, axis=1)),
            "off_y": np.array_equal(off_y1, np.flip(off_y0, axis=1)),
        }
        dx_pred = np.flip(off_x0, axis=1)
        nz = dx_pred != 0
        neg_ok = bool(np.allclose(off_x1[nz], -dx_pred[nz], atol=1e-7))
        mx = float(np.max(np.abs(off_x1[nz] + dx_pred[nz]))) if nz.any() else 0.0
        hm_peak_dev = float(np.max(np.abs(hm1 - np.flip(hm0, axis=1))))
        print(
            f"idx {idx}: mirror exact x/sem/band/hm/off_y = {checks} | "
            f"off_x negation ok={neg_ok} max|err|={mx:.2e} "
            f"hm mirror max dev {hm_peak_dev:.1e}"
        )
        if not all(checks.values()) or not neg_ok:
            raise SystemExit(f"FLIP CHECK FAILED at idx {idx}")
    print("flip check PASS (2 samples, bit-exact mirror + dx negation)")


if __name__ == "__main__":
    main()
