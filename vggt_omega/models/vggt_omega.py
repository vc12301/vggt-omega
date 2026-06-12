# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.gs_adapter import GaussianAdapter
from vggt_omega.models.heads import CameraHead, DenseHead, GSDPTHead, TextAlignmentHead
from vggt_omega.utils.pose_enc import encoding_to_camera


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        enable_gs: bool = False,
        gs_sh_degree: int = 0,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

        self.gs_head = None
        self.gs_adapter = None
        if enable_gs:
            if not (enable_camera and enable_depth):
                raise ValueError("enable_gs requires both the camera and depth heads")
            self.gs_adapter = GaussianAdapter(sh_degree=gs_sh_degree)
            self.gs_head = GSDPTHead(dim_in=2 * embed_dim, patch_size=patch_size, sh_degree=gs_sh_degree)
            assert self.gs_head.output_dim == self.gs_adapter.d_in + 1

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

            if self.gs_head is not None:
                raw_gs, gs_opacity = self.gs_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["raw_gs"] = raw_gs
                predictions["gs_opacity"] = gs_opacity

                extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
                predictions["gaussians"] = self.gs_adapter(
                    extrinsics=extrinsic,
                    intrinsics=intrinsic,
                    depths=predictions["depth"][..., 0],
                    opacities=gs_opacity,
                    raw_gaussians=raw_gs,
                    images=images,
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
