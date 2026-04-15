from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List

import cv2
import numpy as np
import torch


def _resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def _load_depth_array(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def _mask_to_bbox_aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 1.0
    width = max(1, int(xs.max()) - int(xs.min()) + 1)
    height = max(1, int(ys.max()) - int(ys.min()) + 1)
    return float(width) / float(height)


def _quantile_stats(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {
            f"{prefix}_q10": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q90": 0.0,
        }
    return {
        f"{prefix}_q10": float(np.quantile(array, 0.10)),
        f"{prefix}_q50": float(np.quantile(array, 0.50)),
        f"{prefix}_q90": float(np.quantile(array, 0.90)),
    }


class ReferenceBankContractError(ValueError):
    def __init__(self, root: Path, missing_items: list[str]):
        self.root = root
        self.missing_items = tuple(missing_items)
        joined = ", ".join(self.missing_items)
        super().__init__(
            f"Reference bank contract check failed under {root}: missing {joined}")


@dataclass(frozen=True)
class ReferenceBankManifest:
    root: Path
    contract_mode: str
    view_count: int
    has_camera: bool
    has_manifest: bool
    has_shape_stats: bool
    has_qa_report: bool
    has_preview_contact_sheet: bool
    qa_passed: bool
    qa_errors: tuple[str, ...]
    missing_items: tuple[str, ...]


@dataclass
class ReferenceBank:
    root: Path
    view_ids: List[str]
    shape_stats: Dict[str, float]
    meta: Dict[str, Any]
    manifest: ReferenceBankManifest
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
        depth = _load_depth_array(record.depth_path)
        mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(record.mask_path)
        mask = (mask > 0).astype(np.uint8)
        rgb = _resize_rgb(rgb, self.image_size)
        depth = _resize_mask(depth, self.image_size).astype(np.float32)
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

    def iter_views(self, batch_size: int = 0) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
        resolved_batch_size = int(batch_size)
        if resolved_batch_size <= 0 or resolved_batch_size >= len(self.view_records):
            images, depths, masks = self.materialize_tensors()
            yield images, depths, masks, list(self.view_ids)
            return
        for start in range(0, len(self.view_records), resolved_batch_size):
            batch_records = self.view_records[start:start + resolved_batch_size]
            batch_images, batch_depths, batch_masks = [], [], []
            batch_ids: list[str] = []
            for record in batch_records:
                image, depth, mask = self._load_record(record)
                batch_images.append(image)
                batch_depths.append(depth)
                batch_masks.append(mask)
                batch_ids.append(record.view_id)
            yield (
                torch.stack(batch_images, dim=0),
                torch.stack(batch_depths, dim=0),
                torch.stack(batch_masks, dim=0),
                batch_ids,
            )


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
    contract_mode: str = "compat"
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
            raise KeyError(f"Unknown part key for reference bank source: {part_key}")
        return candidate

    def load_for_query(self, file_name: str) -> ReferenceBank:
        resolved_root = self.resolve_root_for_query(file_name)
        if resolved_root not in self._bank_cache:
            self._bank_cache[resolved_root] = load_reference_bank(
                resolved_root,
                image_size=self.image_size,
                contract_mode=self.contract_mode,
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
                contract_mode=self.contract_mode,
                max_views=self.max_views,
                view_sampler=self.view_sampler,
            )
        return self._bank_cache[resolved_root]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_num_views(meta: Dict[str, Any]) -> int | None:
    if "num_views" in meta and meta["num_views"] is not None:
        return int(meta["num_views"])
    if "views" in meta and meta["views"] is not None:
        warnings.warn(
            "Reference bank manifest field 'views' is deprecated; use 'num_views' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return int(meta["views"])
    return None


def _is_bank_root(root: Path) -> bool:
    return all((root / name).exists() for name in ["rgb", "depth", "mask"])


def extract_reference_part_key(file_name: str, available_parts: list[str]) -> str:
    for part_key in sorted(available_parts, key=lambda item: (-len(item), item)):
        if file_name.startswith(part_key + "_"):
            return part_key
    raise KeyError(f"Could not resolve part key from reference file name: {file_name}")


def _validate_contract(root: Path, contract_mode: str) -> ReferenceBankManifest:
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    mask_dir = root / "mask"
    camera_dir = root / "camera"
    meta_dir = root / "meta"
    manifest_path = meta_dir / "manifest.json"
    qa_report_path = meta_dir / "qa_report.json"
    shape_stats_path = meta_dir / "shape_stats.json"
    preview_path = meta_dir / "preview_contact_sheet.png"

    missing = []
    for path in [rgb_dir, depth_dir, mask_dir]:
        if not path.exists():
            missing.append(path.relative_to(root).as_posix())
    if contract_mode == "strict":
        for path in [camera_dir, meta_dir, manifest_path, qa_report_path, shape_stats_path, preview_path]:
            if not path.exists():
                missing.append(path.relative_to(root).as_posix())
    if missing:
        if contract_mode == "compat":
            raise FileNotFoundError(
                f"Reference directory not found: {missing[0]}")
        raise ReferenceBankContractError(root, missing)

    qa_payload = _read_json(qa_report_path) if qa_report_path.exists() else {}
    qa_errors = tuple(str(item) for item in qa_payload.get("errors", []))
    qa_passed = bool(qa_payload.get("qa_passed", not qa_errors))

    return ReferenceBankManifest(
        root=root,
        contract_mode=contract_mode,
        view_count=0,
        has_camera=camera_dir.exists(),
        has_manifest=manifest_path.exists(),
        has_shape_stats=shape_stats_path.exists(),
        has_qa_report=qa_report_path.exists(),
        has_preview_contact_sheet=preview_path.exists(),
        qa_passed=qa_passed,
        qa_errors=qa_errors,
        missing_items=tuple(missing),
    )


def load_reference_bank(
    reference_root: str | Path,
    image_size: int,
    contract_mode: str = "compat",
    max_views: int = 0,
    view_sampler: str = "all",
) -> ReferenceBank:
    if contract_mode not in {"compat", "strict"}:
        raise ValueError(f"Unsupported contract_mode: {contract_mode}")
    if view_sampler not in {"all", "uniform", "pose_farthest"}:
        raise ValueError(f"Unsupported view_sampler: {view_sampler}")

    root = Path(reference_root).resolve()
    manifest = _validate_contract(root, contract_mode)
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    mask_dir = root / "mask"
    meta_dir = root / "meta"
    camera_dir = root / "camera"

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
    area_ratios, aspect_ratios = [], []
    for view_id in view_ids:
        rgb_path = rgb_files[view_id]
        depth_path = depth_files[view_id]
        mask_path = mask_files[view_id]
        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)
        mask = cv2.imread(str(mask_files[view_id]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_files[view_id])
        mask = (mask > 0).astype(np.uint8)
        mask = _resize_mask(mask, image_size).astype(np.uint8)
        view_records.append(
            ReferenceViewRecord(
                view_id=view_id,
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_path=mask_path,
            )
        )
        area_ratios.append(float(mask.mean()))
        aspect_ratios.append(_mask_to_bbox_aspect(mask))

    shape_stats_path = meta_dir / "shape_stats.json"
    if shape_stats_path.exists():
        shape_stats = _read_json(shape_stats_path)
    else:
        shape_stats = {}
    shape_stats.setdefault("mean_area_ratio", float(np.mean(area_ratios)))
    shape_stats.setdefault("mean_aspect_ratio", float(np.mean(aspect_ratios)))
    shape_stats.setdefault("mean_bbox_aspect_ratio",
                           float(np.mean(aspect_ratios)))
    for key, value in _quantile_stats(area_ratios, "area").items():
        shape_stats.setdefault(key, value)
    for key, value in _quantile_stats(aspect_ratios, "aspect").items():
        shape_stats.setdefault(key, value)

    meta: Dict[str, Any] = {}
    manifest_path = meta_dir / "manifest.json"
    if manifest_path.exists():
        meta = _read_json(manifest_path)
    manifest_views = _manifest_num_views(meta) if meta else None

    missing = list(manifest.missing_items)
    if contract_mode == "strict":
        rgb_stems = set(rgb_files)
        depth_stems = set(depth_files)
        mask_stems = set(mask_files)
        if not (rgb_stems == depth_stems == mask_stems):
            missing.append(
                "rgb/depth/mask stem mismatch "
                f"rgb_only={sorted(rgb_stems - depth_stems - mask_stems)} "
                f"depth_only={sorted(depth_stems - rgb_stems - mask_stems)} "
                f"mask_only={sorted(mask_stems - rgb_stems - depth_stems)}"
            )
        if manifest_views is not None and int(manifest_views) != len(view_ids):
            missing.append(
                f"meta/manifest.json num_views={manifest_views} expected={len(view_ids)}")
        if not manifest.qa_passed:
            missing.append("meta/qa_report.json qa_passed=false")
        if manifest.qa_errors:
            missing.append("meta/qa_report.json errors_present")
        if not shape_stats_path.exists():
            missing.append("meta/shape_stats.json")
        if not (meta_dir / "preview_contact_sheet.png").exists():
            missing.append("meta/preview_contact_sheet.png")
        missing.extend(
            f"camera/{view_id}.json"
            for view_id in view_ids
            if not (camera_dir / f"{view_id}.json").exists()
        )
        if missing:
            raise ReferenceBankContractError(root, sorted(set(missing)))

    manifest = ReferenceBankManifest(
        root=root,
        contract_mode=contract_mode,
        view_count=len(view_ids),
        has_camera=camera_dir.exists(),
        has_manifest=manifest_path.exists(),
        has_shape_stats=shape_stats_path.exists(),
        has_qa_report=(meta_dir / "qa_report.json").exists(),
        has_preview_contact_sheet=(
            meta_dir / "preview_contact_sheet.png").exists(),
        qa_passed=manifest.qa_passed,
        qa_errors=manifest.qa_errors,
        missing_items=tuple(sorted(set(missing))),
    )

    return ReferenceBank(
        root=root,
        view_ids=view_ids,
        shape_stats={
            key: float(value) for key, value in shape_stats.items() if isinstance(value, (int, float))
        },
        meta=meta,
        manifest=manifest,
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
    return [view_ids[index] for index in sorted(selected)]


def _pose_farthest_sample_view_ids(root: Path, view_ids: list[str], max_views: int) -> list[str]:
    camera_dir = root / "camera"
    vectors = []
    for view_id in view_ids:
        path = camera_dir / f"{view_id}.json"
        if not path.exists():
            return []
        payload = _read_json(path)
        position = payload.get("position")
        quat = payload.get("quat_xyzw")
        if not isinstance(position, list) or not isinstance(quat, list):
            return []
        vectors.append(np.asarray(list(position) + list(quat), dtype=np.float32))
    if not vectors:
        return []
    selected = [0]
    while len(selected) < min(int(max_views), len(view_ids)):
        best_index = None
        best_distance = -1.0
        for index, vector in enumerate(vectors):
            if index in selected:
                continue
            distance = min(
                float(np.linalg.norm(vector - vectors[selected_index]))
                for selected_index in selected
            )
            if distance > best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
    return [view_ids[index] for index in sorted(selected)]
