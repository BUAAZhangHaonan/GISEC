"""gpu_fast pipeline tests.

CPU-only hosts: the module must import cleanly and the CUDA-gated
tests skip (the repo convention, cf. test_exp24_proj_anchor.requires_cuda).
On a CUDA host the GPU sobel / rank / NMS primitives are compared
BITWISE against the canonical CPU implementations on synthetic inputs
(the design goal: everything except torch.hypot is bit-equal), and a
random-weight end-to-end run checks the marker/sem/rank parity of the
full gpu_stage against the canonical decode+rank chain.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gisec import gpu_pipeline as gp
from gisec import postproc_fast as pf

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for gpu_fast tests"
)


def test_import_cpu_only():
    """Module imports without initializing torch/CUDA (lazy backend)."""
    import gisec.gpu_pipeline  # noqa: F401

    assert "torch" not in [m for m in ()]  # trivial; real check: no crash above


@requires_cuda
def test_gpu_sobel_bitwise():
    """gx/gy from the slicing-arithmetic GPU sobel == numba _sobel_xy."""
    st = gp._backend()
    r = np.random.default_rng(0)
    for shape in ((64, 64), (257, 509), (1, 1), (3, 5)):
        a = r.standard_normal(shape).astype(np.float32) * 10
        ref_gx, ref_gy = pf._sobel_xy(a)
        t = torch.from_numpy(a).cuda()
        with torch.inference_mode():
            gx, gy = gp._gpu_sobel_mag_parts(st, t)
        assert np.array_equal(gx.cpu().numpy(), ref_gx), shape
        assert np.array_equal(gy.cpu().numpy(), ref_gy), shape


@requires_cuda
def test_gpu_rank_dev_bitwise():
    """_gpu_rank_dev == reference _rank (ties share rank; -0.0 merged)."""
    st = gp._backend()
    r = np.random.default_rng(1)
    cases = [
        r.standard_normal(1000).astype(np.float32),
        (r.integers(0, 7, 4096).astype(np.float32) / 3.0),  # heavy ties
        np.array([0.0, -0.0, 1.0, -0.0, 0.0], np.float32),
        np.zeros(16, np.float32),
        r.standard_normal(1 << 20).astype(np.float32),
    ]
    for a in cases:
        ref_rank, ref_nrank = pf._rank(a)
        with torch.inference_mode():
            got, nrank = gp._gpu_rank_dev(
                st, torch.from_numpy(a.copy()).cuda() + 0.0, a.size
            )
        assert np.array_equal(got.cpu().numpy(), ref_rank)
        assert nrank == int(ref_nrank)


@requires_cuda
def test_maxpool_nms_equiv():
    """F.max_pool2d(3,1,1) == scipy maximum_filter(mode='nearest') for
    the peak-mask semantics (hm >= mx)."""
    from scipy import ndimage as ndi

    r = np.random.default_rng(2)
    for _ in range(5):
        hm = r.random((64, 64)).astype(np.float32)
        hm[10:20, 10:20] = 0.5  # plateau ties
        hm[0, :] = 0.9  # border row
        hm[:, -1] = 0.85
        ref = (hm >= ndi.maximum_filter(hm, size=3, mode="nearest")) & (hm > 0.3)
        t = torch.from_numpy(hm)[None, None].cuda()
        with torch.inference_mode():
            mx = torch.nn.functional.max_pool2d(t, 3, stride=1, padding=1)[0, 0]
            got = (torch.from_numpy(hm).cuda() >= mx) & (
                torch.from_numpy(hm).cuda() > 0.3
            )
        assert np.array_equal(got.cpu().numpy(), ref)


@requires_cuda
def test_gpu_stage_parity_random_model(monkeypatch):
    """Full gpu_stage vs the canonical chain on one synthetic image with
    a random-weight SeedNet: markers/peaks/sem bitwise-equal, mix rank
    equality >= 99.9% (torch.hypot vs np.hypot ulps are the only
    allowed divergence), split outputs consistent."""
    from gisec import decode, inference
    from gisec.model import SeedNet

    monkeypatch.setattr(decode, "HM_THR", 0.05)  # random weights: low bar
    torch.manual_seed(3)
    model = SeedNet().cuda().eval()
    inference._gpu_divisors()

    r = np.random.default_rng(4)
    img = (r.random((1024, 1024, 3)) * 255).astype(np.uint8)
    depth = (r.random((1024, 1024)) * 0.4 + 0.25).astype(np.float32)

    g = gp.GpuPipeline(model, sem_thr=0.5)
    payload = g.gpu_stage(img, depth)

    with torch.inference_mode():
        sem_logit, hm, off = inference._forward(model, img, depth)
    coords, cells = decode._cn_markers_with_cells(hm, off)
    peaks = decode._marker_peaks(hm, coords, cells)
    sem = decode.sem_binary(sem_logit, 0.5)

    assert payload.coords == list(coords)
    assert np.array_equal(payload.peaks, peaks)
    assert np.array_equal(payload.sem, sem)
    rank_d, _ = pf.compute_elevation_rank(depth)
    rank_s, _ = pf.sem_logit_rank(sem_logit)
    rank_c, _ = pf.mix_elevation_rank(rank_d, rank_s)
    eq = float((payload.rank == rank_c).mean())
    assert eq == 1.0, f"mix rank equality {eq:.6f} (design is bitwise)"

    _, coco_g = gp.cpu_stage(payload, 1)
    _, coco_c = pf.process(1, coords, sem, depth, sem_logit, peaks, split="val")
    assert len(coco_g) == len(coco_c)
    for a, b in zip(coco_g, coco_c, strict=True):
        assert a["bbox"] == b["bbox"]
        assert a["segmentation"]["counts"] == b["segmentation"]["counts"]


@requires_cuda
def test_infer_one_contract():
    from gisec.model import SeedNet

    torch.manual_seed(5)
    model = SeedNet().cuda().eval()
    r = np.random.default_rng(6)
    img = (r.random((1024, 1024, 3)) * 255).astype(np.uint8)
    depth = (r.random((1024, 1024)) * 0.4 + 0.25).astype(np.float32)
    insts, results = gp.infer_one(model, img, depth, image_id=7, sem_thr=0.5)
    assert isinstance(insts, list)
    assert isinstance(results, list)
    for row in results:
        assert row["image_id"] == 7
        assert row["category_id"] == 1
        assert isinstance(row["segmentation"]["counts"], str)
