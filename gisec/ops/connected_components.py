from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


def _source_root() -> Path:
    return Path(__file__).resolve().parent / "csrc"


@lru_cache(maxsize=1)
def _load_extension():
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    source_root = _source_root()
    return load(
        name="gisec_connected_components_cuda",
        sources=[
            str(source_root / "registry.cu"),
            str(source_root / "buf_2d.cu"),
            str(source_root / "buf_3d.cu"),
        ],
        extra_include_paths=[str(source_root)],
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
        with_cuda=True,
        verbose=False,
    )


def _pad_even_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if x.ndim != 2:
        raise ValueError(f"connected_components_labeling expects a 2D tensor, got {tuple(x.shape)}")
    pad_h = int(x.shape[0] % 2)
    pad_w = int(x.shape[1] % 2)
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h)), (pad_h, pad_w)


def connected_components_labeling(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"connected_components_labeling expects torch.Tensor, got {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"connected_components_labeling expects a 2D tensor, got {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("connected_components_labeling requires a CUDA tensor")
    tensor = x.to(dtype=torch.uint8, device=x.device).contiguous()
    padded, (pad_h, pad_w) = _pad_even_2d(tensor)
    ext = _load_extension()
    labels = ext.cc_2d(padded)
    if pad_h:
        labels = labels[:-pad_h, :]
    if pad_w:
        labels = labels[:, :-pad_w]
    return labels.to(dtype=torch.int32)
