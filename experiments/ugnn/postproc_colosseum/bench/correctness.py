#!/usr/bin/env python
"""Correctness gate: contestant output vs reference output.

Usage:
  python bench/correctness.py --module <mod> [--fn run]   # importable
  python bench/correctness.py --json <results.json>       # prefetched
Module contract (or JSON equivalent of its outputs):
  fn(image_id:int, sem:u8[H,W], hm:f32[h/4,w/4], off:f32[2,h/4,w/4],
     depth:f32[H,W], h:int, w:int) -> list[COCO result dict]
     (RLE + score + image_id + category_id=1), deterministic.
JSON format: {"<image_id>": [result dicts]}.

Checks (PROBLEM.md section 5):
  C1 per-image instance count: frac(|n-n_ref|==0) >= 0.95 and max dev <= 1
  C2 center-matched mean IoU >= 0.995
  C3 probe COCO segm AP |delta| <= 0.01 (probe = 50 imgs of the 250)
  C4 determinism: module run twice on same dump -> identical RLE bytes
"""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
REF = json.loads((HERE/'data/reference_outputs/reference_outputs.json').read_text())
DUMPS = HERE/'data/dumps'
PROBE = sorted(int(k) for k in REF)[:50]  # deterministic probe set


def load_dump(image_id):
    d = np.load(DUMPS/f'{image_id}.npz')
    return (d['sem'].astype(np.uint8), d['hm'].astype(np.float32),
            d['off'].astype(np.float32), d['depth'])


def rle_to_mask(res, H, W):
    from pycocotools import mask as m
    return m.decode(res['segmentation'])


def match_iou(ref_res, usr_res, H, W):
    """Mean over ref instances of best-IoU vs usr, via RLE IoU."""
    import pycocotools.mask as m
    if not ref_res:
        return 1.0
    if not usr_res:
        return 0.0
    r = [x['segmentation'] for x in ref_res]
    u = [x['segmentation'] for x in usr_res]
    iou = m.iou(r, u, [0] * len(u))  # [n_ref, n_usr]
    return float(np.mean(iou.max(axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module'); ap.add_argument('--fn', default='run')
    ap.add_argument('--json', dest='jjson')
    args = ap.parse_args()
    metas = {int(m['image_id']): m for m in json.loads((DUMPS/'metajs.json').read_text())}

    if args.module:
        mod = importlib.import_module(args.module)
        fn = getattr(mod, args.fn)
        def get_results(iid, second=False):
            sem, hm, off, depth = load_dump(iid)
            m = metas[iid]
            return fn(iid, sem, hm, off, depth, m['height'], m['width'])
    else:
        data = json.loads(Path(args.jjson).read_text())
        def get_results(iid, second=False):
            return data[str(iid)]

    devs, ious = [], []
    det_ok = True
    usr_results = {}
    for iid in sorted(int(k) for k in REF):
        r = get_results(iid)
        usr_results[iid] = r
        ref = REF[str(iid)]['results']
        m = metas[iid]
        devs.append(abs(len(r) - len(ref)))
        ious.append(match_iou(ref, r, m['height'], m['width']))
        if args.module and iid in PROBE:
            r2 = get_results(iid, second=True)
            if json.dumps(r, sort_keys=True) != json.dumps(r2, sort_keys=True):
                det_ok = False

    # C3 probe AP
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ann = HERE.parents[2]/'datasets/20260318_1K_32254/annotations/instances_val.json'
    gt = COCO(str(ann))
    flat = lambda d: [x for iid in PROBE for x in d[iid]]
    dt_ref = gt.loadRes(flat({i: REF[str(i)]['results'] for i in PROBE}))
    dt_usr = gt.loadRes(flat({i: usr_results[i] for i in PROBE}))
    aps = []
    for dt in (dt_ref, dt_usr):
        ev = COCOeval(gt, dt, 'segm'); ev.params.imgIds = PROBE
        ev.evaluate(); ev.accumulate(); ev.summarize(); aps.append(float(ev.stats[0]))

    c1 = (np.mean([d == 0 for d in devs]) >= 0.95) and (max(devs) <= 1)
    c2 = np.mean(ious) >= 0.995
    c3 = abs(aps[0] - aps[1]) <= 0.01
    c4 = det_ok
    print(f'C1 count: zero-dev frac={np.mean([d==0 for d in devs]):.4f} max_dev={max(devs)} -> {"PASS" if c1 else "FAIL"}')
    print(f'C2 IoU:   mean={np.mean(ious):.6f} -> {"PASS" if c2 else "FAIL"}')
    print(f'C3 AP:    ref={aps[0]:.4f} usr={aps[1]:.4f} |d|={abs(aps[0]-aps[1]):.5f} -> {"PASS" if c3 else "FAIL"}')
    print(f'C4 det:   {"PASS" if c4 else "FAIL"}')
    print('VERDICT:', 'PASS' if (c1 and c2 and c3 and c4) else 'FAIL')


if __name__ == '__main__':
    main()
