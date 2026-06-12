# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# DPT trunk mirrors dense_head.py; the full-resolution image-feature injection
# follows the GSDPT design of Depth-Anything-3.

import math

import torch
import torch.nn as nn

from .dense_head import (
    _make_dense_resize_layer,
    _make_fusion_block,
    _make_scratch,
    custom_interpolate,
)
from .utils import create_uv_grid, position_grid_to_embed


def gs_channel_layout(sh_degree: int) -> dict[str, slice]:
    """Channel layout of the raw Gaussian parameters (matches the DA3 adapter order)."""
    d_sh = (sh_degree + 1) ** 2
    return {
        "xy_offset": slice(0, 2),
        "scales": slice(2, 5),
        "quaternion": slice(5, 9),  # XYZW, scalar-last
        "sh": slice(9, 9 + 3 * d_sh),
        "depth_offset": slice(9 + 3 * d_sh, 10 + 3 * d_sh),
    }


class GSDPTHead(nn.Module):
    """DPT-style head predicting per-pixel 3D Gaussian parameters.

    Outputs raw (unactivated) parameters in camera space; GaussianAdapter applies
    the activations and transforms them to world space. The opacity channel is
    the only one activated here (sigmoid).
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 16,
        sh_degree: int = 0,
        features: int = 256,
        out_channels: list[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: list[int] = [4, 11, 17, 23],
        gaussian_scale_min: float = 1e-5,
        gaussian_scale_max: float = 30.0,
        scale_multiplier: float = 0.1,
        init_pixel_size: float = 0.25,
        init_opacity: float = 0.12,
    ) -> None:
        super().__init__()

        if not 0 <= sh_degree <= 2:
            raise ValueError(f"GSDPTHead supports sh_degree in [0, 2], got {sh_degree}")

        self.patch_size = patch_size
        self.sh_degree = sh_degree
        self.intermediate_layer_idx = intermediate_layer_idx
        self.scale_multiplier = scale_multiplier
        self.init_pixel_size = init_pixel_size
        self.init_opacity = init_opacity
        self.channel_layout = gs_channel_layout(sh_degree)
        # raw params + 1 opacity channel
        self.output_dim = self.channel_layout["depth_offset"].stop + 1

        self.norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                _make_dense_resize_layer(channels=out_channels[0], resize_scale=4.0),
                _make_dense_resize_layer(channels=out_channels[1], resize_scale=2.0),
                _make_dense_resize_layer(channels=out_channels[2], resize_scale=1.0),
                _make_dense_resize_layer(channels=out_channels[3], resize_scale=0.5),
            ]
        )

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        merger_dim = features // 2
        self.output_conv1 = nn.Conv2d(features, merger_dim, kernel_size=3, stride=1, padding=1)
        # Injects high-frequency RGB detail at full resolution (key for sharp GS colors).
        self.images_merger = nn.Sequential(
            nn.Conv2d(3, merger_dim // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(merger_dim // 4, merger_dim // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(merger_dim // 2, merger_dim, 3, 1, 1),
            nn.GELU(),
        )
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(merger_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.output_dim, kernel_size=1, stride=1, padding=0),
        )
        _init_gs_prediction_head(
            self.output_conv2[-1],
            self.channel_layout,
            gaussian_scale_min=gaussian_scale_min,
            gaussian_scale_max=gaussian_scale_max,
            scale_multiplier=scale_multiplier,
            init_pixel_size=init_pixel_size,
            init_opacity=init_opacity,
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
        frames_chunk_size: int | None = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_token_start is None:
            raise ValueError("patch_token_start is required for GSDPTHead")

        _, num_frames, _, _, _ = images.shape

        if frames_chunk_size is None or frames_chunk_size >= num_frames:
            return self._forward_impl(aggregated_tokens_list, images, patch_token_start)

        assert frames_chunk_size > 0

        raw_gs_chunks = []
        opacity_chunks = []
        for frames_start_idx in range(0, num_frames, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, num_frames)
            raw_gs_chunk, opacity_chunk = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_token_start,
                frames_start_idx,
                frames_end_idx,
            )
            raw_gs_chunks.append(raw_gs_chunk)
            opacity_chunks.append(opacity_chunk)

        return torch.cat(raw_gs_chunks, dim=1), torch.cat(opacity_chunks, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        batch_size, num_frames, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size

        multi_scale_features = []
        for feature_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx]
            if x is None:
                raise ValueError(f"Aggregator did not cache layer {layer_idx}, which GSDPTHead needs.")
            x = x[:, :, patch_token_start:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            if x.dtype != torch.float32:
                x = x.float()

            x = x.reshape(batch_size * num_frames, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[feature_idx](x)
            x = self._apply_pos_embed(x, width, height)
            x = self.resize_layers[feature_idx](x)
            multi_scale_features.append(x)

        fused = self.scratch_forward(multi_scale_features)
        fused = self._apply_pos_embed(fused, width, height)

        fused = self.output_conv1(fused)
        fused = custom_interpolate(fused, (height, width), mode="bilinear", align_corners=True)
        images_flat = images.reshape(batch_size * num_frames, 3, height, width).float()
        fused = fused + self.images_merger(images_flat)
        fused = self._apply_pos_embed(fused, width, height)

        logits = self.output_conv2(fused)
        logits = logits.permute(0, 2, 3, 1)

        raw_gs = logits[..., :-1]
        gs_opacity = torch.sigmoid(logits[..., -1])

        raw_gs = raw_gs.view(batch_size, num_frames, *raw_gs.shape[1:])
        gs_opacity = gs_opacity.view(batch_size, num_frames, *gs_opacity.shape[1:])

        if raw_gs.dtype != torch.float32 or gs_opacity.dtype != torch.float32:
            raise TypeError(f"GSDPTHead outputs must be fp32, got raw_gs={raw_gs.dtype}, opacity={gs_opacity.dtype}")

        return raw_gs, gs_opacity

    def _apply_pos_embed(self, x: torch.Tensor, width: int, height: int, ratio: float = 0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=width / height, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        return self.scratch.refinenet1(out, layer_1_rn, size=layer_1_rn.shape[2:])


def _init_gs_prediction_head(
    proj: nn.Conv2d,
    channel_layout: dict[str, slice],
    gaussian_scale_min: float,
    gaussian_scale_max: float,
    scale_multiplier: float,
    init_pixel_size: float,
    init_opacity: float,
) -> None:
    """Zero the final conv so the head outputs exactly the per-channel biases below.

    All biases live in logit/raw space; after GaussianAdapter activation the
    initial scene is: Gaussians at unprojected depth points (zero xy/depth
    offset), identity rotation, opacity `init_opacity`, color = pixel RGB (zero
    residual SH), and an isotropic scale whose screen-space projection is
    ~`init_pixel_size` pixel.
    """
    nn.init.zeros_(proj.weight)
    if proj.bias is None:
        raise ValueError("GS prediction head init requires a bias term")

    bias = torch.zeros_like(proj.bias)

    # The adapter maps world scale = act(raw) * depth * scale_multiplier * (1/fx + 1/fy),
    # which projects to ~act(raw) * 2 * scale_multiplier pixels (square pixels),
    # so an `init_pixel_size` px footprint needs act(raw) = init_pixel_size / (2 * scale_multiplier).
    target_scale = init_pixel_size / (2.0 * scale_multiplier)
    p = (target_scale - gaussian_scale_min) / (gaussian_scale_max - gaussian_scale_min)
    if not 0.0 < p < 1.0:
        raise ValueError(f"Initial scale {target_scale} is outside ({gaussian_scale_min}, {gaussian_scale_max})")
    bias[channel_layout["scales"]] = math.log(p / (1.0 - p))

    # Identity quaternion in XYZW order; the adapter normalizes, so (0,0,0,1) is exact.
    bias[channel_layout["quaternion"].stop - 1] = 1.0

    # Opacity is sigmoid-activated in the head, so store its logit on the last channel.
    if not 0.0 < init_opacity < 1.0:
        raise ValueError(f"init_opacity must be in (0, 1), got {init_opacity}")
    bias[-1] = math.log(init_opacity / (1.0 - init_opacity))

    # xy_offset, sh (residual), and depth_offset stay 0.

    with torch.no_grad():
        proj.bias.copy_(bias)
