from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gisec.datasets.coco_utils import load_depth_array
from gisec.models.gisec_model import prepare_reference_depth


def _resize_linear(array: np.ndarray, image_size: int) -> np.ndarray:
    # Depth is a continuous field, so it resamples bilinearly like RGB;
    # nearest would leave stair-step artifacts along resample boundaries.
    return cv2.resize(array, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


@dataclass
class ReferenceBank:
    root: Path
    image_size: int
    view_records: list["ReferenceViewRecord"]
    _images: torch.Tensor | None = field(default=None, init=False, repr=False)
    _depths: torch.Tensor | None = field(default=None, init=False, repr=False)
    _masks: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _load_record(self, record: "ReferenceViewRecord") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = cv2.imread(str(record.rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(record.rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = load_depth_array(record.depth_path)
        mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(record.mask_path)
        mask = (mask > 0).astype(np.uint8)
        rgb = _resize_linear(rgb, self.image_size)
        depth = _resize_linear(depth, self.image_size).astype(np.float32)
        mask = _resize_mask(mask, self.image_size).astype(np.uint8)
        return (
            torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0,
            torch.from_numpy(depth[None, ...]).float(),
            torch.from_numpy(mask[None, ...]).float(),
        )

    def materialize_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._images is None or self._depths is None or self._masks is None:
            images, depths, masks = [], [], []
            for record in self.view_records:
                image, depth, mask = self._load_record(record)
                images.append(image)
                depths.append(depth)
                masks.append(mask)
            self._images = torch.stack(images, dim=0)
            self._depths = torch.stack(depths, dim=0)
            self._masks = torch.stack(masks, dim=0)
        return self._images, self._depths, self._masks

    @property
    def images(self) -> torch.Tensor:
        return self.materialize_tensors()[0]

    @property
    def depths(self) -> torch.Tensor:
        return self.materialize_tensors()[1]

    @property
    def masks(self) -> torch.Tensor:
        return self.materialize_tensors()[2]


@dataclass(frozen=True)
class ReferenceViewRecord:
    view_id: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path


@dataclass
class ReferenceBankSource:
    root: Path
    image_size: int
    max_views: int = 0
    view_sampler: str = "all"

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self._is_single_bank = _is_bank_root(self.root)
        self._bank_cache: dict[Path, ReferenceBank] = {}
        self._available_parts = sorted(
            [
                path.name
                for path in self.root.iterdir()
                if path.is_dir() and _is_bank_root(path)
            ],
            key=lambda item: (-len(item), item),
        ) if not self._is_single_bank else []

    @property
    def available_parts(self) -> list[str]:
        return list(self._available_parts)

    @property
    def is_single_bank(self) -> bool:
        return self._is_single_bank

    def resolve_root_for_query(self, file_name: str) -> Path:
        if self._is_single_bank:
            return self.root
        part_key = extract_reference_part_key(file_name, self._available_parts)
        return (self.root / part_key).resolve()

    def resolve_root_for_part(self, part_key: str) -> Path:
        if self._is_single_bank:
            return self.root
        candidate = (self.root / str(part_key)).resolve()
        if candidate.name not in self._available_parts or not candidate.exists():
            raise ValueError(
                f"Unknown part key for reference bank source: {part_key}; "
                f"available parts: {self._available_parts}")
        return candidate

    def load_for_query(self, file_name: str) -> ReferenceBank:
        resolved_root = self.resolve_root_for_query(file_name)
        if resolved_root not in self._bank_cache:
            self._bank_cache[resolved_root] = load_reference_bank(
                resolved_root,
                image_size=self.image_size,
                max_views=self.max_views,
                view_sampler=self.view_sampler,
            )
        return self._bank_cache[resolved_root]

    def load_for_part(self, part_key: str) -> ReferenceBank:
        resolved_root = self.resolve_root_for_part(part_key)
        if resolved_root not in self._bank_cache:
            self._bank_cache[resolved_root] = load_reference_bank(
                resolved_root,
                image_size=self.image_size,
                max_views=self.max_views,
                view_sampler=self.view_sampler,
            )
        return self._bank_cache[resolved_root]


def reference_tensors_from_bank(
    *,
    bank: ReferenceBank,
    crop_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_ref_depth = prepare_reference_depth(
        depth=bank.depths.float(),
        mask=bank.masks.float(),
    )
    return (
        F.interpolate(bank.images.float().to(device), size=(
            crop_size, crop_size), mode="bilinear", align_corners=False).unsqueeze(0),
        F.interpolate(normalized_ref_depth.to(device), size=(
            crop_size, crop_size), mode="bilinear", align_corners=False).unsqueeze(0),
        F.interpolate(bank.masks.float().to(device), size=(
            crop_size, crop_size), mode="nearest").unsqueeze(0),
    )


def prepare_reference_tensors(
    *,
    sample: dict[str, Any],
    source: ReferenceBankSource | None,
    crop_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if source is None:
        return None, None, None
    bank = source.load_for_query(str(sample["file_name"]))
    return reference_tensors_from_bank(bank=bank, crop_size=crop_size, device=device)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_bank_root(root: Path) -> bool:
    return all((root / name).exists() for name in ["rgb", "depth", "mask"])


def extract_reference_part_key(file_name: str, available_parts: list[str]) -> str:
    for part_key in sorted(available_parts, key=lambda item: (-len(item), item)):
        if file_name.startswith(part_key + "_"):
            return part_key
    raise ValueError(
        f"Could not resolve part key from reference file name: {file_name}; "
        f"available parts: {sorted(available_parts)}")


def load_reference_bank(
    reference_root: str | Path,
    image_size: int,
    max_views: int = 0,
    view_sampler: str = "all",
) -> ReferenceBank:
    if view_sampler not in {"all", "uniform", "pose_farthest"}:
        raise ValueError(f"Unsupported view_sampler: {view_sampler}")

    root = Path(reference_root).resolve()
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    mask_dir = root / "mask"

    missing = [
        path.relative_to(root).as_posix()
        for path in [rgb_dir, depth_dir, mask_dir]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Reference directory not found: {missing[0]}")

    rgb_files = {p.stem: p for p in sorted(rgb_dir.glob("*")) if p.is_file()}
    depth_files = {p.stem: p for p in sorted(
        depth_dir.glob("*.npy")) if p.is_file()}
    mask_files = {p.stem: p for p in sorted(mask_dir.glob("*")) if p.is_file()}
    view_ids = sorted(set(rgb_files) & set(depth_files) & set(mask_files))
    if not view_ids:
        raise FileNotFoundError(
            f"No matched rgb/depth/mask reference views found under {root}")
    view_ids = _sample_view_ids(
        root=root,
        view_ids=view_ids,
        max_views=max_views,
        view_sampler=view_sampler,
    )

    view_records: list[ReferenceViewRecord] = []
    for view_id in view_ids:
        view_records.append(
            ReferenceViewRecord(
                view_id=view_id,
                rgb_path=rgb_files[view_id],
                depth_path=depth_files[view_id],
                mask_path=mask_files[view_id],
            )
        )

    return ReferenceBank(
        root=root,
        image_size=int(image_size),
        view_records=view_records,
    )


def _sample_view_ids(
    *,
    root: Path,
    view_ids: list[str],
    max_views: int,
    view_sampler: str,
) -> list[str]:
    if int(max_views) <= 0 or len(view_ids) <= int(max_views):
        return list(view_ids)
    if view_sampler == "uniform":
        return _uniform_sample_view_ids(view_ids, max_views)
    if view_sampler == "pose_farthest":
        sampled = _pose_farthest_sample_view_ids(root, view_ids, max_views)
        if sampled:
            return sampled
        return _uniform_sample_view_ids(view_ids, max_views)
    return list(view_ids[:max_views])


def _uniform_sample_view_ids(view_ids: list[str], max_views: int) -> list[str]:
    if max_views <= 0 or len(view_ids) <= max_views:
        return list(view_ids)
    if max_views == 1:
        return [view_ids[0]]
    selected = {
        int(round(index))
        for index in np.linspace(0, len(view_ids) - 1, num=max_views)
    }
    # linspace rounding can collapse to fewer than max_views distinct
    # indices; fill the remaining slots with the lowest unselected ones so
    # the caller always receives exactly min(max_views, available) views.
    for index in range(len(view_ids)):
        if len(selected) >= int(max_views):
            break
        selected.add(index)
    return [view_ids[index] for index in sorted(selected)]


def _pose_farthest_sample_view_ids(root: Path, view_ids: list[str], max_views: int) -> list[str]:
    camera_dir = root / "camera"
    positions = []
    for view_id in view_ids:
        path = camera_dir / f"{view_id}.json"
        if not path.exists():
            print(
                f"[gisec] reference bank: camera pose {path} is missing; "
                "falling back to uniform view sampling", flush=True)
            return []
        payload = _read_json(path)
        position = payload.get("position")
        if not isinstance(position, list):
            print(
                f"[gisec] reference bank: camera pose {path} has no position "
                "list; falling back to uniform view sampling", flush=True)
            return []
        # Distances use the camera position only: position is meters while
        # quaternions are unitless, so mixing them into one euclidean
        # distance compares incompatible units.
        positions.append(np.asarray(list(position), dtype=np.float32))
    if not positions:
        return []
    selected = [0]
    while len(selected) < min(int(max_views), len(view_ids)):
        best_index = None
        best_distance = -1.0
        for index, position in enumerate(positions):
            if index in selected:
                continue
            distance = min(
                float(np.linalg.norm(position - positions[selected_index]))
                for selected_index in selected
            )
            if distance > best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
    return [view_ids[index] for index in sorted(selected)]
