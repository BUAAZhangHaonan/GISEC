"""Post-integration determinism check: N val images (default 200),
forward once, postproc_fast.process twice per image in the same process;
per-image CRC32 of the serialized instances must be identical.

Usage: python determinism_check.py OUT_JSON [N_IMAGES]
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))

import eval_pipeline as ep  # noqa: E402
import postproc_fast  # noqa: E402
from eval_centernet import _cn_markers, _forward  # noqa: E402
from eval_scale import DATA, HM_THR, load_split  # noqa: E402
from train_centernet import SeedNet  # noqa: E402

ep.DATA = DATA


def insts_crc(insts) -> int:
    h = zlib.crc32(b"")
    for mask, area in insts:
        h = zlib.crc32(struct.pack("<i", area), h)
        h = zlib.crc32(np.ascontiguousarray(mask).tobytes(), h)
    return h


def main() -> None:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ckpt = torch.load(HERE / "runs" / "best.pth", map_location="cpu")
    model = SeedNet()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    metas, _ = load_split("val")
    rows = []
    for meta in metas[:n]:
        img = ep.cv2.imread(str(DATA / "images" / "val" / meta["file_name"]))
        img = ep.cv2.cvtColor(img, ep.cv2.COLOR_BGR2RGB)
        depth = ep.load_depth_array(Path(meta["dpath"]))
        sem, hm, off = _forward(model, img, depth)
        coords = _cn_markers(hm, off, HM_THR)
        r1 = postproc_fast.process(meta["image_id"], coords, sem, depth)
        r2 = postproc_fast.process(meta["image_id"], coords, sem, depth)
        rows.append(
            {
                "image_id": meta["image_id"],
                "n_inst": len(r1[0]),
                "crc1": insts_crc(r1[0]),
                "crc2": insts_crc(r2[0]),
                "n_results": len(r1[1]),
            }
        )
    out = {
        "n": len(rows),
        "in_process_identical": all(r["crc1"] == r["crc2"] for r in rows),
        "crcs": {r["image_id"]: r["crc1"] for r in rows},
    }
    Path(sys.argv[1]).write_text(json.dumps(out))
    print(f"n={out['n']} in_process_identical={out['in_process_identical']}")


if __name__ == "__main__":
    main()
