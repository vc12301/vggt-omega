# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Rendering follows the Depth-Anything-3 gsplat wiring; gsplat is an optional
# dependency, so it is imported lazily inside render_gaussians.

import math

import torch

from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat


def render_gaussians(
    gaussians,  # Gaussians dataclass with batch size 1 (see models/gs_adapter.py)
    extrinsics: torch.Tensor,  # (V, 3, 4) or (V, 4, 4), camera-from-world (OpenCV)
    intrinsics: torch.Tensor,  # (V, 3, 3), pixel units
    height: int,
    width: int,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    chunk_size: int = 8,
    render_mode: str = "RGB+D",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render Gaussians with gsplat. Returns colors (V, 3, H, W) and depths (V, H, W)."""
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise ImportError(
            "gsplat is required for rendering 3D Gaussians. "
            "Install it via: pip install gsplat"
        ) from exc

    if gaussians.means.shape[0] != 1:
        raise ValueError("render_gaussians expects batch_size=1 Gaussians")

    device = gaussians.means.device
    means = gaussians.means[0].float()
    quats = gaussians.rotations[0].float()  # WXYZ, as gsplat expects
    scales = gaussians.scales[0].float()
    opacities = gaussians.opacities[0].float()
    # (M, 3, d_sh) -> (M, d_sh, 3)
    sh_coeffs = gaussians.harmonics[0].float().permute(0, 2, 1).contiguous()
    sh_degree = math.isqrt(sh_coeffs.shape[1]) - 1

    num_views = extrinsics.shape[0]
    viewmats = torch.eye(4, device=device).repeat(num_views, 1, 1)
    viewmats[:, :3, :] = extrinsics[:, :3, :].float().to(device)
    intrinsics = intrinsics.float().to(device)
    backgrounds = torch.tensor(background_color, device=device).expand(num_views, 3).contiguous()

    all_colors = []
    all_depths = []
    for start in range(0, num_views, max(1, chunk_size)):
        end = min(start + max(1, chunk_size), num_views)
        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_coeffs,
            viewmats=viewmats[start:end],
            Ks=intrinsics[start:end],
            backgrounds=backgrounds[start:end],
            render_mode=render_mode,
            width=width,
            height=height,
            packed=False,
            sh_degree=sh_degree,
        )
        all_colors.append(render_colors[..., :3].permute(0, 3, 1, 2))
        all_depths.append(render_colors[..., -1])

    return torch.cat(all_colors), torch.cat(all_depths)


def interpolate_camera_path(
    extrinsics: torch.Tensor,  # (V, 3, 4) or (V, 4, 4), camera-from-world
    intrinsics: torch.Tensor,  # (V, 3, 3)
    steps_per_pair: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoothly interpolate between consecutive camera poses.

    Translation is lerped and rotation slerped in camera-to-world space with
    cosine easing per pair. Returns (V', 3, 4) extrinsics and (V', 3, 3)
    intrinsics with V' = (V - 1) * steps_per_pair + 1.
    """
    extrinsics = extrinsics[:, :3, :].float()
    intrinsics = intrinsics.float()
    num_views = extrinsics.shape[0]
    if num_views < 2:
        return extrinsics, intrinsics

    rot_c2w = extrinsics[:, :3, :3].transpose(-1, -2)
    t_c2w = -(rot_c2w @ extrinsics[:, :3, 3:]).squeeze(-1)
    quats = mat_to_quat(rot_c2w)  # XYZW

    t = torch.linspace(0.0, 1.0, steps_per_pair + 1, device=extrinsics.device)[:-1]
    t = (torch.cos(torch.pi * (t + 1.0)) + 1.0) / 2.0  # cosine ease-in-out

    out_quats = []
    out_trans = []
    out_intr = []
    for i in range(num_views - 1):
        out_quats.append(_slerp(quats[i], quats[i + 1], t))
        out_trans.append(t_c2w[i][None] + t[:, None] * (t_c2w[i + 1] - t_c2w[i])[None])
        out_intr.append(intrinsics[i][None] + t[:, None, None] * (intrinsics[i + 1] - intrinsics[i])[None])
    out_quats.append(quats[-1:])
    out_trans.append(t_c2w[-1:])
    out_intr.append(intrinsics[-1:])

    rot_c2w_path = quat_to_mat(torch.cat(out_quats))
    t_c2w_path = torch.cat(out_trans)

    rot_w2c_path = rot_c2w_path.transpose(-1, -2)
    t_w2c_path = -(rot_w2c_path @ t_c2w_path.unsqueeze(-1))
    return torch.cat([rot_w2c_path, t_w2c_path], dim=-1), torch.cat(out_intr)


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation between two unit quaternions; t of shape (T,)."""
    dot = (q0 * q1).sum()
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        out = q0[None] + t[:, None] * (q1 - q0)[None]
        return out / out.norm(dim=-1, keepdim=True)
    theta = torch.acos(dot.clamp(-1.0, 1.0))
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    return w0[:, None] * q0[None] + w1[:, None] * q1[None]
