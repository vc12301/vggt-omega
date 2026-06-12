"""Training utilities: LR scheduler + GSDPT output-layer (re)initialization.

Unlike Depth-Anything-3, VGGT-Omega is pose-free, so there is no
``normalize_extrinsics`` here. The GSDPT output-layer init reuses the existing
``vggt_omega.models.heads.gsdpt_head._init_gs_prediction_head`` (near-zero conv
weight — a tiny Gaussian perturbation with std ``init_weight_std`` to break
per-pixel symmetry — plus logit-space biases, identity quaternion, opacity
logit) so the channel layout stays the single source of truth
(``gs_channel_layout``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from vggt_omega.models.heads.gsdpt_head import GSDPTHead, _init_gs_prediction_head


def _final_conv(gs_head: GSDPTHead) -> nn.Conv2d:
    """Return the final Conv2d of the GSDPT head (``output_conv2``'s last layer)."""
    last_conv = None
    for m in gs_head.output_conv2.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Could not find output_conv2's last Conv2d in gs_head")
    return last_conv


def initialize_gsdpt_output_layer(gs_head: GSDPTHead) -> None:
    """Re-apply the precise logit-space init to the GSDPT head's final conv.

    Keeps the pretrained/trained middle layers; resets only the output layer to
    the documented initialization (zero offsets/SH, identity quaternion,
    ``init_opacity``, ``init_pixel_size`` footprint).
    """
    _init_gs_prediction_head(
        _final_conv(gs_head),
        gs_head.channel_layout,
        gaussian_scale_min=getattr(gs_head, "gaussian_scale_min", 1e-5),
        gaussian_scale_max=getattr(gs_head, "gaussian_scale_max", 30.0),
        scale_multiplier=gs_head.scale_multiplier,
        init_pixel_size=gs_head.init_pixel_size,
        init_opacity=gs_head.init_opacity,
        init_weight_std=gs_head.init_weight_std,
    )


def reinitialize_gsdpt_scratch(gs_head: GSDPTHead) -> None:
    """Full reinitialization for training from scratch.

    All conv/linear layers get kaiming_normal, then the final output layer gets
    the precise logit-space init via :func:`initialize_gsdpt_output_layer`.
    """
    for m in gs_head.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    initialize_gsdpt_output_layer(gs_head)


class CosineAnnealingWarmupScheduler(LRScheduler):
    """Cosine annealing with linear warmup.

    During warmup (step < warmup_steps):
        lr = base_lr * (step + 1) / warmup_steps

    After warmup:
        lr = min_lr + 0.5 * (base_lr - min_lr) *
             (1 + cos(pi * (step - warmup_steps) / (max_steps - warmup_steps)))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * (step + 1) / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(
                    1, self.max_steps - self.warmup_steps
                )
                progress = min(progress, 1.0)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            lrs.append(lr)
        return lrs
