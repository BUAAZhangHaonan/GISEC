from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models import resnet50


def _dice_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    eps: float = 1.0,
) -> torch.Tensor:
    pred = pred_masks.sigmoid().flatten(1)
    target = target_masks.flatten(1)
    numerator = 2.0 * (pred * target).sum(dim=1) + float(eps)
    denominator = pred.sum(dim=1) + target.sum(dim=1) + float(eps)
    return 1.0 - (numerator / denominator)


def _batch_pairwise_dice_cost(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    eps: float = 1.0,
) -> torch.Tensor:
    pred = pred_masks.flatten(1)
    target = target_masks.flatten(1)
    numerator = 2.0 * torch.einsum("qc,mc->qm", pred, target) + float(eps)
    denominator = pred.sum(dim=1, keepdim=True) + target.sum(dim=1).unsqueeze(0) + float(eps)
    return 1.0 - (numerator / denominator)


def _batch_sigmoid_bce_cost(
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    pred = pred_logits.flatten(1)
    target = target_masks.flatten(1)
    num_pixels = max(1, int(pred.shape[1]))
    # BCEWithLogits(x, y) = softplus(x) - x*y. This is exact and avoids
    # allocating the old Q x M x pixels matcher tensor.
    softplus_term = F.softplus(pred).mean(dim=1, keepdim=True)
    target_term = torch.matmul(pred, target.transpose(0, 1)) / float(num_pixels)
    return softplus_term - target_term


class CoordConv(nn.Module):
    """Append normalized x/y coordinate maps before a convolutional block."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        yy = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype).view(1, 1, 1, width)
        yy = yy.expand(batch, 1, height, width)
        xx = xx.expand(batch, 1, height, width)
        return torch.cat([x, xx, yy], dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class IAUNetPixelDecoderBlock(nn.Module):
    """CoordConv + double point-wise conv + BN/ReLU + SE refinement."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.coordconv = CoordConv()
        self.main = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcitationBlock(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.main(self.coordconv(x)))


class MaskFeatureUpdateBlock(nn.Module):
    """Stacked convolutions used to update mask features at each decoder stage."""

    def __init__(self, hidden_dim: int, mask_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, mask_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IAUNetR50Encoder(nn.Module):
    """ResNet-50 encoder that returns 1/4, 1/8, 1/16, and 1/32 features."""

    out_channels = [256, 512, 1024, 2048]

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        backbone = resnet50(weights=None)
        if int(in_channels) != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(images)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class IAUNetPixelDecoder(nn.Module):
    def __init__(
        self,
        in_channels_list: Sequence[int],
        hidden_dim: int,
        mask_dim: int,
        pooled_size: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mask_dim = int(mask_dim)
        self.num_stages = len(in_channels_list)
        self.pooled_size = int(pooled_size)
        self.skip_projections = nn.ModuleList(
            [nn.Conv2d(in_channels, hidden_dim, kernel_size=1) for in_channels in in_channels_list]
        )
        self.decoder_blocks = nn.ModuleList([IAUNetPixelDecoderBlock(hidden_dim) for _ in in_channels_list])
        self.mask_updates = nn.ModuleList([MaskFeatureUpdateBlock(hidden_dim, mask_dim) for _ in in_channels_list])

    def forward(self, features: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if len(features) != self.num_stages:
            raise ValueError(f"expected {self.num_stages} encoder features, got {len(features)}")

        main: torch.Tensor | None = None
        stage_mask_features: List[torch.Tensor] = []
        stage_memories: List[torch.Tensor] = []
        zipped = zip(
            reversed(features),
            reversed(self.skip_projections),
            reversed(self.decoder_blocks),
            reversed(self.mask_updates),
        )
        for skip, projection, decoder_block, mask_update in zipped:
            projected = projection(skip)
            if main is not None:
                main = F.interpolate(main, size=projected.shape[-2:], mode="bilinear", align_corners=False)
                projected = projected + main
            main = decoder_block(projected)
            mask_features = mask_update(main)
            stage_mask_features.append(mask_features)
            memory = (
                F.adaptive_avg_pool2d(mask_features, output_size=(self.pooled_size, self.pooled_size))
                .flatten(2)
                .permute(2, 0, 1)
            )
            stage_memories.append(memory)

        high_res_mask_features = stage_mask_features[-1]
        return high_res_mask_features, stage_memories


class IAUNetQueryDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_stages: int,
        blocks_per_stage: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_stages = int(num_stages)
        self.blocks_per_stage = int(blocks_per_stage)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)
        self.stage_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.TransformerDecoderLayer(
                            d_model=hidden_dim,
                            nhead=num_heads,
                            dim_feedforward=hidden_dim * 4,
                            dropout=0.0,
                            batch_first=False,
                            activation="relu",
                        )
                        for _ in range(blocks_per_stage)
                    ]
                )
                for _ in range(num_stages)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, stage_memories: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(stage_memories) != self.num_stages:
            raise ValueError(f"expected {self.num_stages} decoder memories, got {len(stage_memories)}")
        batch_size = int(stage_memories[0].shape[1])
        query = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)
        hidden_states: List[torch.Tensor] = []
        tgt = query
        for memory, blocks in zip(stage_memories, self.stage_blocks):
            for block in blocks:
                tgt = block(tgt=tgt + query_pos, memory=memory)
            # Deep supervision per-stage (not per-block): one hidden state per stage
            hidden_states.append(self.norm(tgt).permute(1, 0, 2))
        return hidden_states


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(0, int(num_layers) - 1) + [out_dim]
        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IAUNetInstanceModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 32,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int | None = None,
        num_decoder_stages: int = 4,
        transformer_blocks_per_stage: int = 3,
        num_heads: int = 8,
        mask_dim: int | None = None,
    ) -> None:
        super().__init__()
        del base_channels
        if num_decoder_layers is not None:
            num_decoder_stages = int(num_decoder_layers)
        if int(num_decoder_stages) != 4:
            raise ValueError("IAUNet-R50 uses exactly four pixel/transformer decoder stages")
        mask_dim = int(mask_dim or hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_decoder_stages = int(num_decoder_stages)
        self.transformer_blocks_per_stage = int(transformer_blocks_per_stage)
        self.paper_faithful = True

        self.encoder = IAUNetR50Encoder(in_channels=in_channels)
        self.pixel_decoder = IAUNetPixelDecoder(
            in_channels_list=self.encoder.out_channels,
            hidden_dim=hidden_dim,
            mask_dim=mask_dim,
        )
        self.query_decoder = IAUNetQueryDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_stages=num_decoder_stages,
            blocks_per_stage=transformer_blocks_per_stage,
            num_heads=num_heads,
        )
        self.class_head = nn.Linear(hidden_dim, 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim)
        self.maskness_head = nn.Linear(hidden_dim, 1)

    def _decode_queries(
        self,
        hidden_states: Sequence[torch.Tensor],
        mask_features: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        predictions: List[Dict[str, torch.Tensor]] = []
        for hidden in hidden_states:
            class_logits = self.class_head(hidden)
            mask_embed = self.mask_embed(hidden)
            mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            maskness_logits = self.maskness_head(hidden)
            predictions.append(
                {
                    "pred_logits": class_logits,
                    "pred_masks": mask_logits,
                    "pred_maskness": maskness_logits,
                }
            )
        final = predictions[-1]
        return final, predictions[:-1]

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        features = self.encoder(images)
        mask_features, stage_memories = self.pixel_decoder(features)
        hidden_states = self.query_decoder(stage_memories)
        final, aux_outputs = self._decode_queries(hidden_states, mask_features)
        if aux_outputs:
            final["aux_outputs"] = aux_outputs
        return final


def _resize_target_masks(
    targets: Sequence[Dict[str, torch.Tensor]],
    mask_size: Tuple[int, int],
    device: torch.device,
) -> List[torch.Tensor]:
    resized: List[torch.Tensor] = []
    for target in targets:
        masks = target["masks"].to(device=device, dtype=torch.float32)
        if masks.numel() == 0:
            resized.append(masks.reshape(0, mask_size[0], mask_size[1]))
            continue
        if masks.shape[-2:] != mask_size:
            masks = F.interpolate(masks[:, None, ...], size=mask_size, mode="nearest").squeeze(1)
        resized.append(masks)
    return resized


class IAUNetHungarianMatcher:
    def __init__(
        self,
        *,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
    ) -> None:
        self.cost_class = float(cost_class)
        self.cost_mask = float(cost_mask)
        self.cost_dice = float(cost_dice)

    @torch.no_grad()
    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        device = pred_logits.device
        target_masks = _resize_target_masks(targets, pred_masks.shape[-2:], device)
        out_prob = pred_logits.softmax(dim=-1)

        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for batch_idx, masks in enumerate(target_masks):
            if masks.numel() == 0:
                empty = torch.zeros((0,), dtype=torch.int64, device=device)
                indices.append((empty, empty))
                continue

            object_cost = -out_prob[batch_idx, :, 1].unsqueeze(1).repeat(1, masks.shape[0])
            mask_cost = _batch_sigmoid_bce_cost(pred_masks[batch_idx], masks)
            dice_cost = _batch_pairwise_dice_cost(pred_masks[batch_idx].sigmoid(), masks)
            cost_matrix = (
                self.cost_class * object_cost
                + self.cost_mask * mask_cost
                + self.cost_dice * dice_cost
            ).detach().cpu()
            src_idx, tgt_idx = linear_sum_assignment(cost_matrix)
            indices.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.int64, device=device),
                    torch.as_tensor(tgt_idx, dtype=torch.int64, device=device),
                )
            )
        return indices


class IAUNetCriterion(nn.Module):
    def __init__(
        self,
        *,
        matcher: IAUNetHungarianMatcher,
        eos_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.register_buffer("class_weights", torch.tensor([float(eos_coef), 1.0], dtype=torch.float32))

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
        indices: list | None = None,
    ) -> Dict[str, torch.Tensor]:
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        pred_maskness = outputs.get("pred_maskness")
        device = pred_logits.device
        if indices is None:
            indices = self.matcher(outputs, targets)
        target_masks = _resize_target_masks(targets, pred_masks.shape[-2:], device)

        target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.int64, device=device)
        matched_pred_masks: List[torch.Tensor] = []
        matched_target_masks: List[torch.Tensor] = []
        maskness_targets = torch.zeros(pred_logits.shape[:2], dtype=torch.float32, device=device)

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[batch_idx, src_idx] = 1
            pred_match = pred_masks[batch_idx, src_idx]
            target_match = target_masks[batch_idx][tgt_idx]
            matched_pred_masks.append(pred_match)
            matched_target_masks.append(target_match)
            with torch.no_grad():
                pred_bin = pred_match.sigmoid()
                inter = (pred_bin * target_match).flatten(1).sum(dim=1)
                union = pred_bin.flatten(1).sum(dim=1) + target_match.flatten(1).sum(dim=1) - inter
                maskness_targets[batch_idx, src_idx] = inter / union.clamp_min(1e-6)

        class_weights = self.class_weights.to(device)
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=class_weights)
        if matched_pred_masks:
            src_masks = torch.cat(matched_pred_masks, dim=0)
            tgt_masks = torch.cat(matched_target_masks, dim=0)
            loss_mask = F.binary_cross_entropy_with_logits(src_masks, tgt_masks)
            loss_dice = _dice_loss(src_masks, tgt_masks).mean()
        else:
            zero = pred_logits.sum() * 0.0
            loss_mask = zero
            loss_dice = zero
        # Paper-faithful loss weights: w_ce=1.0, w_mask=5.0, w_dice=2.0, w_maskness=1.0
        losses = {
            "loss_ce": loss_ce * 1.0,
            "loss_mask": loss_mask * 5.0,
            "loss_dice": loss_dice * 2.0,
        }
        if pred_maskness is not None:
            losses["loss_maskness"] = F.binary_cross_entropy_with_logits(
                pred_maskness.squeeze(-1),
                maskness_targets,
            ) * 1.0
        return losses

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        # Compute matching ONCE on main output
        indices = self.matcher(outputs, targets)
        losses = self._compute_losses(outputs, targets, indices=indices)

        aux_outputs = outputs.get("aux_outputs", [])
        for layer_idx, aux_output in enumerate(aux_outputs):
            # Reuse same indices for aux losses
            aux_losses = self._compute_losses(aux_output, targets, indices=indices)
            for key, value in aux_losses.items():
                losses[f"{key}_aux{layer_idx}"] = value
        return losses


@torch.no_grad()
def iaunet_inference(
    outputs: Dict[str, torch.Tensor],
    *,
    original_sizes: Sequence[Tuple[int, int]],
    score_threshold: float = 0.4,
    mask_threshold: float = 0.5,
    min_area: int = 20,
    max_instances: int | None = None,
) -> List[Dict[str, torch.Tensor]]:
    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    class_scores = pred_logits.softmax(dim=-1)[..., 1]
    pred_maskness = outputs.get("pred_maskness")
    if pred_maskness is None:
        maskness = torch.ones_like(class_scores)
    else:
        maskness = pred_maskness.squeeze(-1).sigmoid()
    scores = class_scores * maskness

    results: List[Dict[str, torch.Tensor]] = []
    for batch_idx, orig_size in enumerate(original_sizes):
        sample_scores = scores[batch_idx]
        sample_masks = pred_masks[batch_idx]
        keep = sample_scores >= float(score_threshold)
        if keep.any():
            sample_scores = sample_scores[keep]
            sample_masks = sample_masks[keep]
        else:
            empty_masks = torch.zeros((0, int(orig_size[0]), int(orig_size[1])), dtype=torch.uint8)
            results.append(
                {
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "category_ids": torch.zeros((0,), dtype=torch.int64),
                    "masks": empty_masks,
                }
            )
            continue

        order = torch.argsort(sample_scores, descending=True)
        if max_instances is not None and order.numel() > int(max_instances):
            order = order[: int(max_instances)]
        sample_scores = sample_scores[order]
        sample_masks = sample_masks[order]

        sample_masks = F.interpolate(
            sample_masks[:, None, ...],
            size=(int(orig_size[0]), int(orig_size[1])),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        binary_masks = (sample_masks.sigmoid() >= float(mask_threshold)).to(torch.uint8)
        flat_area = binary_masks.flatten(1).sum(dim=1)
        keep_area = flat_area >= int(min_area)

        if keep_area.any():
            sample_scores = sample_scores[keep_area].detach().cpu()
            binary_masks = binary_masks[keep_area].detach().cpu()
            category_ids = torch.zeros((binary_masks.shape[0],), dtype=torch.int64)
        else:
            sample_scores = torch.zeros((0,), dtype=torch.float32)
            binary_masks = torch.zeros((0, int(orig_size[0]), int(orig_size[1])), dtype=torch.uint8)
            category_ids = torch.zeros((0,), dtype=torch.int64)

        results.append(
            {
                "scores": sample_scores,
                "category_ids": category_ids,
                "masks": binary_masks,
            }
        )
    return results


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))
