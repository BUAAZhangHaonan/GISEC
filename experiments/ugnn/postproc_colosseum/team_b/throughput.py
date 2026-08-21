"""Throughput: 8 worker processes, each runs the bench hot path."""
import sys, time, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import team_b.solution as S

metas = {int(m["image_id"]): m for m in json.loads((HERE/"data/dumps/metajs.json").read_text())}
ids = sorted(metas)

def one(iid):
    d = np.load(HERE/f"data/dumps/{iid}.npz")
    m = metas[iid]
    return len(S.run(iid, d["sem"].astype(np.uint8), d["hm"].astype(np.float32),
                     d["off"].astype(np.float32), d["depth"], m["height"], m["width"]))

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with Pool(n) as p:
        for _ in p.map(one, ids[:24]):  # warmup + compile in each worker
            pass
        t0 = time.perf_counter()
        res = p.map(one, ids)
        dt = time.perf_counter() - t0
    print(f"workers={n} imgs={len(ids)} wall={dt:.2f}s -> {len(ids)/dt:.2f} imgs/s "
          f"({dt/len(ids)*1000:.1f} ms/img wall)")
