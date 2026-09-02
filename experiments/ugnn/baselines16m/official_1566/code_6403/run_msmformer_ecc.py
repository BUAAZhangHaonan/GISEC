#!/usr/bin/env python3
"""
Run MSMFormer (vendored baseline) on ECC datasets (0831 / 0909) with COCOeval (segm/bbox AP).

This wrapper:
- registers the ECC RGBD COCO dataset in-process (adds `depth_file_name`)
- wires a minimal RGBD DatasetMapper (reads `.npy` depth)
- runs Detectron2 training/eval with MSMFormer code vendored under:
  `baselines/msmformer/`
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as d2_utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# Ensure sibling baseline utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))


def _ensure_msmformer_import_paths(msmformer_root: str | Path | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    root = Path(msmformer_root) if msmformer_root is not None else repo_root / "baselines" / "msmformer" / "MSMFormer"
    root = root.resolve()
    if root.exists():
        for path in (root, root.parent, root.parent / "tools"):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    return root


_ensure_msmformer_import_paths()

from depth_stats import (
    load_0831_1k_depth_stats,
    load_0909_512_depth_stats,
    load_depth_stats_for_dataset_root,
)
from ecc_datasets import normalize_register, register_ecc_coco_rgbd
from rgbd_geometry import depth_to_xyz


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_msmformer_pretrained_path(repo_root: Path | None = None) -> Path:
    root = repo_root if repo_root is not None else _repo_root()
    return root / "output" / "pretrained" / "norm_RGBD_pretrained.pth"


def _find_model_weights_opt(opts: List[str]) -> str | None:
    for i in range(0, len(opts) - 1, 2):
        if opts[i] == "MODEL.WEIGHTS":
            return opts[i + 1]
    return None


def _with_default_model_weights(
    opts: List[str] | tuple[str, ...],
    *,
    explicit_pretrained: str | None,
    repo_root: Path | None = None,
) -> list[str]:
    resolved_opts = list(opts)
    existing = _find_model_weights_opt(resolved_opts)
    if existing is not None:
        return resolved_opts

    if explicit_pretrained is not None and explicit_pretrained.strip().lower() == "none":
        return resolved_opts

    weights_path = (
        Path(explicit_pretrained).expanduser()
        if explicit_pretrained
        else _default_msmformer_pretrained_path(repo_root=repo_root)
    )
    if not weights_path.exists():
        if explicit_pretrained:
            raise FileNotFoundError(f"MSMFormer pretrained checkpoint not found: {weights_path}")
        return resolved_opts

    resolved_opts.extend(["MODEL.WEIGHTS", str(weights_path.resolve())])
    return resolved_opts


@dataclass(frozen=True)
class MSMFormerRecipe:
    use_depth: bool
    use_other_backbone: bool
    num_classes: int
    convs_dim: int
    mask_dim: int
    pixel_decoder_name: str
    transformer_in_feature: str
    transformer_decoder_name: str
    use_meanshift_cross_attention: bool
    use_meanshift_self_attention: bool
    disable_attention_mask: bool
    decoder_block_norm: bool
    class_weight: float
    mask_weight: float
    dice_weight: float
    dropout: float
    dec_layers: int
    object_mask_threshold: float
    overlap_threshold: float
    ucn_input_type: str
    ucn_fusion_type: str
    ucn_embedding_pretrain: bool
    ucn_embedding_metric: str
    ucn_embedding_normalization: bool
    ucn_embedding_lambda_intra: float
    ucn_embedding_lambda_inter: float


def build_msmformer_recipe(register: str) -> MSMFormerRecipe:
    _ = register
    return MSMFormerRecipe(
        use_depth=True,
        use_other_backbone=False,
        num_classes=1,
        convs_dim=64,
        mask_dim=256,
        pixel_decoder_name="SimpleBasePixelDecoder",
        transformer_in_feature="multi_scale_pixel_decoder",
        transformer_decoder_name="PretrainedMeanShiftTransformerDecoder",
        use_meanshift_cross_attention=True,
        use_meanshift_self_attention=True,
        disable_attention_mask=False,
        decoder_block_norm=True,
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        dropout=0.0,
        dec_layers=7,
        object_mask_threshold=0.8,
        overlap_threshold=0.8,
        ucn_input_type="RGBD",
        ucn_fusion_type="add",
        ucn_embedding_pretrain=False,
        ucn_embedding_metric="cosine",
        ucn_embedding_normalization=True,
        ucn_embedding_lambda_intra=10.0,
        ucn_embedding_lambda_inter=10.0,
    )


def _split_args(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def _default_dataset_root(register: str) -> str:
    ws_root = Path(__file__).resolve().parents[2]
    if register == "0831":
        return str(ws_root / "magformer_datasets" / "0831_1K")
    return str(ws_root / "magformer_datasets" / "0909_512_0.12K")


_DEPTH_MIN = 0.0
_DEPTH_MAX = 1.0


def _set_global_depth_range(register: str, dataset_root: str | None = None) -> None:
    global _DEPTH_MIN, _DEPTH_MAX
    if register == "0831":
        stats = load_0831_1k_depth_stats()
    elif register == "0909":
        stats = load_0909_512_depth_stats()
    elif dataset_root is not None:
        stats = load_depth_stats_for_dataset_root(dataset_root)
    else:
        raise ValueError(f"Custom register requires explicit dataset_root for depth stats: {register}")
    _DEPTH_MIN = float(stats.p1)
    _DEPTH_MAX = float(stats.p99)


class DatasetMapperRGBD:
    """
    Minimal Detectron2 dataset mapper that provides:
    - image: float32 tensor (3,H,W), normalized according to the active backbone path
    - depth: float32 tensor (3,H,W), loaded from `.npy` and resized/flipped identically
    """

    def __init__(self, cfg, is_train: bool):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.use_other_backbone = bool(cfg.MODEL.USE_OTHER_BACKBONE)
        self.pixel_mean = np.asarray(cfg.MODEL.PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.pixel_std = np.asarray(cfg.MODEL.PIXEL_STD, dtype=np.float32).reshape(1, 1, 3)
        self.depth_min = float(_DEPTH_MIN)
        self.depth_max = float(_DEPTH_MAX)

        # Fixed-size baseline protocol: resize to 512x512; apply the same transform to RGB+depth.
        target = (int(cfg.INPUT.MIN_SIZE_TEST), int(cfg.INPUT.MIN_SIZE_TEST))
        augs: List[T.Augmentation] = [T.Resize(target)]
        if is_train:
            augs.insert(0, T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
        self.augmentations = T.AugmentationList(augs)

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        dataset_dict = copy.deepcopy(dataset_dict)

        image = d2_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        depth_path = dataset_dict.get("depth_file_name")
        if not depth_path:
            raise KeyError("Missing `depth_file_name` in dataset dict (did you register RGBD dataset?)")

        depth = np.load(depth_path).astype(np.float32)
        depth_hw = depth[:, :, 0] if depth.ndim == 3 and depth.shape[2] == 1 else depth
        if depth_hw.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Depth shape {depth_hw.shape[:2]} does not match RGB shape {image.shape[:2]} for: {dataset_dict['file_name']}"
            )

        if depth_hw.ndim != 2:
            raise ValueError(f"Expected scalar depth map from dataset, got shape={depth_hw.shape}")

        combo = np.concatenate([image.astype(np.float32), depth_hw[:, :, None]], axis=2)  # (H,W,4)
        aug_input = T.AugInput(combo)
        transforms = self.augmentations(aug_input)
        combo = aug_input.image

        image = combo[:, :, :3]
        depth = combo[:, :, 3]
        image_shape = image.shape[:2]  # (H,W)

        depth = np.clip(depth, self.depth_min, self.depth_max)
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min + 1e-6)
        depth = depth_to_xyz(depth)

        # The UCN backbone expects mean subtraction in pixel space followed by /255.
        # Other backbones keep standard Detectron2 mean/std normalization.
        if self.use_other_backbone:
            image = (image - self.pixel_mean) / self.pixel_std
        else:
            image = (image - self.pixel_mean) / 255.0

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)
        dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth.transpose(2, 0, 1)), dtype=torch.float32)

        if "annotations" in dataset_dict:
            annos = [
                d2_utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = d2_utils.annotations_to_instances(annos, image_shape, mask_format=self.mask_format)
            instances = d2_utils.filter_empty_instances(instances)
            # MSMFormer target preparation expects dense masks as a tensor (N,H,W).
            # Convert after filtering to keep Detectron2's `nonempty()` checks working.
            if hasattr(instances, "gt_masks"):
                mask_tensor = instances.gt_masks.tensor if hasattr(instances.gt_masks, "tensor") else instances.gt_masks
                label_map = torch.full(image_shape, -1, dtype=torch.int64)
                for inst_id, mask in enumerate(mask_tensor):
                    label_map[mask > 0] = int(inst_id)
                dataset_dict["label"] = label_map.unsqueeze(0)
                if hasattr(instances.gt_masks, "tensor"):
                    instances.gt_masks = mask_tensor
            dataset_dict["instances"] = instances

        return dataset_dict


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapperRGBD(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapperRGBD(cfg, is_train=False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=True, output_dir=output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_msmformer_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)


def build_msmformer_optimizer(cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {
        "lr": cfg.SOLVER.BASE_LR,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad or value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
            if (
                "relative_position_bias_table" in module_param_name
                or "absolute_pos_embed" in module_param_name
            ):
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def setup(args) -> Any:
    cfg = get_cfg()

    # MSMFormer config extension.
    # NOTE: The vendored code path is injected in `cli()` before calling setup().
    from meanshiftformer.config import add_meanshiftformer_config

    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    recipe = build_msmformer_recipe(str(cfg.DATASETS.TRAIN[0]) if len(cfg.DATASETS.TRAIN) else "0831")
    cfg.MODEL.USE_DEPTH = recipe.use_depth
    cfg.MODEL.USE_OTHER_BACKBONE = recipe.use_other_backbone
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = recipe.num_classes
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = recipe.convs_dim
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = recipe.mask_dim
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = recipe.pixel_decoder_name
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = recipe.transformer_in_feature
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = recipe.transformer_decoder_name
    cfg.MODEL.MASK_FORMER.USE_MEANSHIFT_CROSS_ATTENTION = recipe.use_meanshift_cross_attention
    cfg.MODEL.MASK_FORMER.USE_MEANSHIFT_SELF_ATTENTION = recipe.use_meanshift_self_attention
    cfg.MODEL.MASK_FORMER.DISABLE_MEANSHIFT_ATTENTION_MASK = recipe.disable_attention_mask
    cfg.MODEL.MASK_FORMER.DECODER_BLOCK_NORM = recipe.decoder_block_norm
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = recipe.class_weight
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = recipe.mask_weight
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = recipe.dice_weight
    cfg.MODEL.MASK_FORMER.DROPOUT = recipe.dropout
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = recipe.dec_layers
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = recipe.object_mask_threshold
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = recipe.overlap_threshold
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.freeze()

    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=0, name="msmformer")
    return cfg


def _write_trainable_params(model: torch.nn.Module, out_dir: str) -> None:
    p = Path(out_dir) / "params_trainable.txt"
    n = sum(int(x.numel()) for x in model.parameters() if x.requires_grad)
    p.write_text(str(n) + "\n", encoding="utf-8")


def main(args, dataset_root: str) -> Dict[str, Any] | None:
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if args.eval_only and args.config_file:
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    _write_trainable_params(trainer.model, cfg.OUTPUT_DIR)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def cli() -> None:
    wrapper_argv, passthrough = _split_args(sys.argv[1:])

    ap = argparse.ArgumentParser()
    ap.add_argument("--register", type=str, default="0831", help="ECC dataset id: 0831 | 0909")
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/<dataset> root (default: env + workspace-relative).",
    )
    ap.add_argument(
        "--msmformer-root",
        type=str,
        default="baselines/msmformer/MSMFormer",
        help="Path to vendored MSMFormer code root (contains `meanshiftformer/`).",
    )
    ap.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help=(
            "Path to official MSMFormer pretrained weights. "
            "Defaults to output/pretrained/norm_RGBD_pretrained.pth when present. "
            "Use 'none' to force scratch."
        ),
    )
    argsw = ap.parse_args(wrapper_argv)

    register_id = normalize_register(argsw.register)
    recipe = build_msmformer_recipe(register_id)

    if argsw.dataset_root is None:
        argsw.dataset_root = _default_dataset_root(register_id)

    _set_global_depth_range(register_id, argsw.dataset_root)

    msm_root = _ensure_msmformer_import_paths(argsw.msmformer_root)
    if not msm_root.exists():
        raise FileNotFoundError(f"msmformer root not found: {msm_root}")

    # Populate Detectron2 registries (meta-arch, heads, etc.).
    import meanshiftformer  # noqa: F401

    # Register datasets in-process (RGBD).
    register_ecc_coco_rgbd(register_id, argsw.dataset_root)

    # MSMFormer uses an internal UCN-style backbone configured via `lib/fcn/config.py`.
    # Keep the RGBD / embedding settings aligned with the canonical ECC recipe.
    from fcn.config import cfg as ucn_cfg

    ucn_cfg.INPUT = recipe.ucn_input_type
    ucn_cfg.TRAIN.FUSION_TYPE = recipe.ucn_fusion_type
    ucn_cfg.TRAIN.EMBEDDING_PRETRAIN = recipe.ucn_embedding_pretrain
    ucn_cfg.TRAIN.EMBEDDING_METRIC = recipe.ucn_embedding_metric
    ucn_cfg.TRAIN.EMBEDDING_NORMALIZATION = recipe.ucn_embedding_normalization
    ucn_cfg.TRAIN.EMBEDDING_LAMBDA_INTRA = recipe.ucn_embedding_lambda_intra
    ucn_cfg.TRAIN.EMBEDDING_LAMBDA_INTER = recipe.ucn_embedding_lambda_inter

    args = default_argument_parser().parse_args(passthrough)
    args.opts = _with_default_model_weights(args.opts, explicit_pretrained=argsw.pretrained)
    resolved_weights = _find_model_weights_opt(args.opts)
    if resolved_weights is not None:
        print(f"[msmformer] loading pretrained checkpoint: {resolved_weights}")
    else:
        print("[msmformer] no pretrained checkpoint found, training from scratch")
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, argsw.dataset_root),
    )


if __name__ == "__main__":
    cli()
