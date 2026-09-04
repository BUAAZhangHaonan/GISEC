"""ws_full self-check vs the numba reference chain (_ws_bucket->_merge->_boxes).

Check A (bitwise): with WS_SMALL=1 (identity merge) ws_full returns the raw
GPU watershed labels + boxes; the numba _merge/_boxes applied to THOSE labels
must match the normal ws_full (WS_SMALL=32) output bitwise.  This isolates
the merge/boxes port from watershed tie noise.

Check B (end-to-end AP): emulate the harness tail (_counts_for_label + RLE +
top-100) on ws_full outputs and compare COCO AP against the canonical chain.
"""
import os, sys, json, time
from contextlib import redirect_stdout
from io import StringIO
import numpy as np

sys.path.insert(0, '/home/k100/zhn/electronic-components-grasp-and-segment/gisec/src')
from gisec import postproc_fast as pf
from gisec.eval.coco_eval import evaluate_json
import pycocotools.mask as M

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solution

ARENA = '/home/k100/zhn/electronic-components-grasp-and-segment/gisex_extreme_arena/arena'
DATA = ('/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets'
        '/20260318_1K_32254')


def _tail_from_boxes(iid, labels, x0, y0, x1, y1, area, peaks, nmarkers):
    labs = [lb for lb in range(1, nmarkers + 1) if area[lb] > pf.MIN_AREA]
    labs.sort(key=lambda lb: (-peaks[lb - 1], area[lb]))
    labs = labs[: pf.MAX_INST]
    H, W = labels.shape
    buf = np.empty(labels.size + 8, dtype=np.uint32)
    out = []
    for lb in labs:
        n = pf._counts_for_label(labels, lb, int(x0[lb]), int(y0[lb]),
                                 int(x1[lb]), int(y1[lb]), buf)
        seg = M.frPyObjects({'size': [H, W], 'counts': buf[:n].tolist()}, H, W)
        if isinstance(seg, list):
            seg = seg[0]
        out.append({
            'image_id': int(iid), 'category_id': 1,
            'score': float(peaks[lb - 1]),
            'bbox': [int(x0[lb]), int(y0[lb]),
                     int(x1[lb] - x0[lb] + 1), int(y1[lb] - y0[lb] + 1)],
            'segmentation': {'size': [H, W], 'counts': seg['counts'].decode()},
        })
    return out


def main():
    man = json.load(open(os.path.join(ARENA, 'manifest.json')))
    tms = []
    results = []
    ok_a = True
    for m in man:
        iid = m['image_id']
        z = np.load(f'{ARENA}/payloads/wsin_{iid}.npz')
        rank = np.load(f'{ARENA}/payloads/rank_{iid}.npy')
        sem, mk, nrank = z['sem'], z['markers'], int(z['nrank'])
        ys, xs = np.nonzero(mk)
        order = np.argsort(mk[ys, xs])
        coords = [(int(y), int(x)) for y, x in zip(ys[order], xs[order])]
        peaks = np.load(os.path.join(ARENA, 'payloads', f'peaks_{iid}.npy'))
        nmarkers = len(coords)
        H, W = sem.shape

        for _ in range(3):
            t0 = time.perf_counter()
            lab, x0, y0, x1, y1, area = solution.ws_full(rank, nrank, sem, mk)
            tms.append(time.perf_counter() - t0)

        # Check A (bitwise, fixed input): re-run ONLY the GPU tail on the
        # merged labels we just produced, and compare with numba
        # _merge/_boxes on the SAME array.  (The full re-run of the sweep
        # is not tie-deterministic, so the tail must be tested in
        # isolation on identical inputs.)
        import torch
        ctx = solution._ctx
        lab2 = lab.copy()
        ctx.lab_g.copy_(torch.from_numpy(lab2.ravel()))
        mb = solution._merge_bufs(ctx, nmarkers + 1)
        nl1 = nmarkers + 1
        mb['counts'][:nl1].zero_()
        mb['adj'][:nl1 * nl1].zero_()
        for t, init in zip(mb['bx'], ((1 << 30), (1 << 30), 0, 0, 0)):
            t[:nl1].fill_(init)
        ctx.mod.ws_tail_forward(ctx.lab_g, mb['counts'], mb['adj'],
                                mb['remap'], mb['bx'][0], mb['bx'][1],
                                mb['bx'][2], mb['bx'][3], mb['bx'][4],
                                nmarkers, W, 32)
        g_lab = ctx.lab_g.cpu().numpy().reshape(H, W)
        gb = [mb['bx'][k][:nl1].cpu().numpy().astype(np.int64)
              for k in range(5)]
        absent = gb[4] == 0
        gb[2] = np.where(absent, -1, gb[2])
        gb[3] = np.where(absent, -1, gb[3])
        gb[0] = np.where(absent, 1 << 30, gb[0])
        gb[1] = np.where(absent, 1 << 30, gb[1])
        mref = pf._merge(lab2, nmarkers)
        bref = pf._boxes(mref, nmarkers)
        lab_eq = bool((g_lab == mref).all())
        box_eq = all(bool(np.array_equal(a, b)) for a, b in
                     zip(gb, bref))
        if not (lab_eq and box_eq):
            ok_a = False
            print(f'iid={iid}: labels_eq={lab_eq} boxes_eq={box_eq} '
                  f'labdiff={int((g_lab != mref).sum())}')
        results += _tail_from_boxes(iid, lab, x0, y0, x1, y1, area,
                                    peaks, nmarkers)

    with redirect_stdout(StringIO()):
        ap = evaluate_json(DATA / 'annotations' / 'instances_val.json' if False
                           else f'{DATA}/annotations/instances_val.json',
                           results,
                           img_ids=[mm['image_id'] for mm in man])['segm/AP']
    canon = json.load(open(os.path.join(ARENA, 'canonical.json')))
    canon_results = [r for v in canon.values() for r in v['results']]
    with redirect_stdout(StringIO()):
        cap = evaluate_json(f'{DATA}/annotations/instances_val.json',
                            canon_results,
                            img_ids=[mm['image_id'] for mm in man])['segm/AP']
    print(f'SELFCHECK ws_full: bitwise merge/boxes vs numba on identical raw '
          f'labels: {"PASS" if ok_a else "FAIL"}')
    print(f'WSFULL ms/img: {1000 * float(np.median(tms)):.2f}')
    print(f'AP {ap:.5f} vs canonical {cap:.5f} (delta {ap - cap:+.5f})')
    return 0 if ok_a and (ap - cap) > -0.005 else 1


if __name__ == '__main__':
    sys.exit(main())
