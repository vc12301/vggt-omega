# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Ported from the Depth-Anything-3 GaussianAdapter; the SH branch is changed to
# a residual formulation on top of the per-pixel RGB color.

from dataclasses import dataclass
from math import isqrt

import torch
import torch.nn as nn

from vggt_omega.models.heads.gsdpt_head import gs_channel_layout
from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat

# Degree-0 spherical harmonic basis constant (1 / (2 * sqrt(pi))).
SH_C0 = 0.28209479177387814


@dataclass
class Gaussians:
    """World-space 3D Gaussians, flattened over frames and pixels (M = N*H*W)."""

    means: torch.Tensor  # (B, M, 3)
    harmonics: torch.Tensor  # (B, M, 3, d_sh)
    opacities: torch.Tensor  # (B, M), in [0, 1]
    scales: torch.Tensor  # (B, M, 3), linear (not log)
    rotations: torch.Tensor  # (B, M, 4), unit quaternions, WXYZ (scalar-first)


class GaussianAdapter(nn.Module):
    """Converts raw GSDPTHead outputs (camera space) to world-space Gaussians.

    The predicted scale is a scene-scale quantity independent of depth and FoV;
    it is multiplied by depth and an intrinsics-derived factor here so that its
    screen-space projection matches the prediction regardless of perspective.
    """

    def __init__(
        self,
        sh_degree: int = 0,
        gaussian_scale_min: float = 1e-5,
        gaussian_scale_max: float = 30.0,
        scale_multiplier: float = 0.1,
    ) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        self.scale_multiplier = scale_multiplier
        self.channel_layout = gs_channel_layout(sh_degree)

        # Damps the higher-degree residual SH coefficients so the color stays
        # dominated by the DC (RGB) component early in training.
        sh_mask = torch.ones(self.d_sh, dtype=torch.float32)
        for degree in range(1, sh_degree + 1):
            sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        self.register_buffer("sh_mask", sh_mask, persistent=False)

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return self.channel_layout["depth_offset"].stop

    def forward(
        self,
        extrinsics: torch.Tensor,  # (B, N, 3, 4) or (B, N, 4, 4), camera-from-world (OpenCV)
        intrinsics: torch.Tensor,  # (B, N, 3, 3), pixel units
        depths: torch.Tensor,  # (B, N, H, W)
        opacities: torch.Tensor,  # (B, N, H, W), already in [0, 1]
        raw_gaussians: torch.Tensor,  # (B, N, H, W, d_in)
        images: torch.Tensor,  # (B, N, 3, H, W), RGB in [0, 1]
        eps: float = 1e-8,
    ) -> Gaussians:
        if raw_gaussians.shape[-1] != self.d_in:
            raise ValueError(f"Expected raw_gaussians with {self.d_in} channels, got {raw_gaussians.shape[-1]}")

        batch_size, num_frames, height, width = depths.shape
        layout = self.channel_layout

        extrinsics = extrinsics[..., :3, :].float()
        intrinsics = intrinsics.float()
        raw_gaussians = raw_gaussians.float()

        rot_w2c = extrinsics[..., :3, :3]
        t_w2c = extrinsics[..., :3, 3]
        rot_c2w = rot_w2c.transpose(-1, -2)
        t_c2w = -(rot_c2w @ t_w2c.unsqueeze(-1)).squeeze(-1)

        fx = intrinsics[..., 0, 0][..., None, None]
        fy = intrinsics[..., 1, 1][..., None, None]
        cx = intrinsics[..., 0, 2][..., None, None]
        cy = intrinsics[..., 1, 2][..., None, None]

        # 1) Means: pixel-center grid + predicted offsets, unprojected with z-depth.
        device = raw_gaussians.device
        grid_v, grid_u = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device) + 0.5,
            torch.arange(width, dtype=torch.float32, device=device) + 0.5,
            indexing="ij",
        )
        u = grid_u + raw_gaussians[..., layout["xy_offset"]][..., 0]
        v = grid_v + raw_gaussians[..., layout["xy_offset"]][..., 1]
        gs_depths = depths + raw_gaussians[..., layout["depth_offset"]][..., 0]

        cam_points = torch.stack(
            [(u - cx) / fx * gs_depths, (v - cy) / fy * gs_depths, gs_depths],
            dim=-1,
        )
        means = torch.einsum("bnij,bnhwj->bnhwi", rot_c2w, cam_points) + t_c2w[:, :, None, None, :]

        # 2) Scales: bounded activation, then perspective-aligned scaling.
        scales = raw_gaussians[..., layout["scales"]].sigmoid()
        scales = self.gaussian_scale_min + (self.gaussian_scale_max - self.gaussian_scale_min) * scales
        multiplier = self.scale_multiplier * (1.0 / fx + 1.0 / fy)
        gs_scales = scales * gs_depths[..., None] * multiplier[..., None]

        # 3) Rotations: normalized camera-space XYZW quaternion -> world space WXYZ.
        quats = raw_gaussians[..., layout["quaternion"]]
        quats = quats / (quats.norm(dim=-1, keepdim=True) + eps)
        rot_cam = quat_to_mat(quats)
        rot_world = torch.einsum("bnij,bnhwjk->bnhwik", rot_c2w, rot_cam)
        quats_world = mat_to_quat(rot_world)[..., [3, 0, 1, 2]]

        # 4) Residual SH: predicted coefficients are added on top of the pixel
        # RGB converted to the degree-0 band.
        sh = raw_gaussians[..., layout["sh"]].reshape(batch_size, num_frames, height, width, 3, self.d_sh)
        sh = sh * self.sh_mask
        base_dc = (images.permute(0, 1, 3, 4, 2).float() - 0.5) / SH_C0
        sh = torch.cat([sh[..., :1] + base_dc[..., None], sh[..., 1:]], dim=-1)
        if self.sh_degree > 0:
            sh = rotate_sh(sh, rot_c2w[:, :, None, None, None])

        flat = lambda x: x.reshape(batch_size, num_frames * height * width, *x.shape[4:])  # noqa: E731
        return Gaussians(
            means=flat(means),
            harmonics=flat(sh),
            opacities=opacities.float().reshape(batch_size, -1),
            scales=flat(gs_scales),
            rotations=flat(quats_world),
        )


def rotate_sh(
    sh_coefficients: torch.Tensor,  # (..., d_sh)
    rotations: torch.Tensor,  # (..., 3, 3), broadcastable against the SH batch dims
) -> torch.Tensor:
    """Rotate SH coefficients by the given rotation matrices (requires e3nn).

    Ported from Depth-Anything-3 sh_helpers.rotate_sh:
    https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2452412653
    """
    try:
        from e3nn.o3 import matrix_to_angles, wigner_D
    except ImportError as exc:
        raise ImportError(
            "e3nn is required to rotate SH coefficients with sh_degree > 0. "
            "Install it via: pip install e3nn"
        ) from exc

    device = sh_coefficients.device
    dtype = sh_coefficients.dtype
    n = sh_coefficients.shape[-1]

    with torch.autocast(device_type=device.type, enabled=False):
        rotations = rotations.to(torch.float32)

        # switch axes: yzx -> xyz
        permute = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=rotations.device
        )
        permuted_rotations = permute.T @ rotations @ permute

        alpha, beta, gamma = matrix_to_angles(_project_to_so3(permuted_rotations))
        result = []
        for degree in range(isqrt(n)):
            with torch.device(device):
                sh_rotations = wigner_D(degree, alpha, -beta, gamma).type(dtype)
            sh_rotated = torch.einsum(
                "...ij,...j->...i",
                sh_rotations,
                sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            )
            result.append(sh_rotated)

    return torch.cat(result, dim=-1)


def _project_to_so3(matrix: torch.Tensor) -> torch.Tensor:
    """Project near-rotation matrices onto SO(3) (e3nn asserts det == 1 exactly)."""
    svd_u, _, svd_vh = torch.linalg.svd(matrix)
    svd_v = svd_vh.mH

    correction = torch.where(
        (torch.det(svd_u) * torch.det(svd_v) < 0)[..., None],
        torch.tensor([1.0, 1.0, -1.0], device=matrix.device, dtype=matrix.dtype),
        torch.tensor([1.0, 1.0, 1.0], device=matrix.device, dtype=matrix.dtype),
    )
    rotation = (svd_u @ torch.diag_embed(correction)) @ svd_v.transpose(-2, -1)

    det_correction = torch.pow(torch.det(rotation), -1.0 / 3.0)[..., None, None]
    return rotation * det_correction
