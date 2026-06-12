"""Target-view valid-visible-pixel mask for GSDPT training.

Ported from Depth-Anything-3 ``gs_training/utils.calculate_in_frustum_mask``.
The three torch geometry helpers (unproject / world->cam / cam->pixel) are
implemented locally to avoid pulling DA3's geometry module; they operate on
pixel-space intrinsics and camera-to-world (c2w) poses.
"""

from __future__ import annotations

import torch


def _unproject_depth(
    depth: torch.Tensor,  # (b, v, h, w, 1)
    intrinsics: torch.Tensor,  # (b, v, 3, 3) pixel-space
    c2w: torch.Tensor,  # (b, v, 4, 4)
) -> torch.Tensor:
    """Unproject per-pixel z-depth to world points. Returns (b, v, h, w, 3)."""
    b, v, h, w, _ = depth.shape
    device = depth.device
    dtype = depth.dtype

    grid_v, grid_u = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device) + 0.5,
        torch.arange(w, dtype=dtype, device=device) + 0.5,
        indexing="ij",
    )
    fx = intrinsics[..., 0, 0][..., None, None]
    fy = intrinsics[..., 1, 1][..., None, None]
    cx = intrinsics[..., 0, 2][..., None, None]
    cy = intrinsics[..., 1, 2][..., None, None]

    z = depth[..., 0]  # (b, v, h, w)
    x = (grid_u - cx) / fx * z
    y = (grid_v - cy) / fy * z
    cam_points = torch.stack([x, y, z], dim=-1)  # (b, v, h, w, 3)

    rot_c2w = c2w[..., :3, :3]  # (b, v, 3, 3)
    t_c2w = c2w[..., :3, 3]  # (b, v, 3)
    world = torch.einsum("bvij,bvhwj->bvhwi", rot_c2w, cam_points) + t_c2w[:, :, None, None, :]
    return world


def _world_to_camera(
    points_world: torch.Tensor,  # (b, v1, h, w, 3)
    c2w_2: torch.Tensor,  # (b, v2, 4, 4)
) -> torch.Tensor:
    """Transform v1's world points into each of v2's camera frames.

    Returns (b, v1, v2, h, w, 3).
    """
    w2c_2 = torch.linalg.inv(c2w_2.float())  # (b, v2, 4, 4)
    rot = w2c_2[..., :3, :3]  # (b, v2, 3, 3)
    t = w2c_2[..., :3, 3]  # (b, v2, 3)
    # rot[b, v2, i, j] @ points[b, v1, h, w, j] -> (b, v1, v2, h, w, i)
    cam = torch.einsum("baij,bvhwj->bvahwi", rot, points_world.to(rot.dtype))
    cam = cam + t[:, None, :, None, None, :]
    return cam


def _camera_to_pixel(
    cam_points: torch.Tensor,  # (b, v1, v2, h, w, 3)
    intrinsics_2: torch.Tensor,  # (b, v2, 3, 3) pixel-space
) -> torch.Tensor:
    """Project camera-space points into v2's pixel coordinates. Returns (..., 2)."""
    fx = intrinsics_2[..., 0, 0][:, None, :, None, None]
    fy = intrinsics_2[..., 1, 1][:, None, :, None, None]
    cx = intrinsics_2[..., 0, 2][:, None, :, None, None]
    cy = intrinsics_2[..., 1, 2][:, None, :, None, None]
    z = cam_points[..., 2].clamp_min(1e-8)
    px = cam_points[..., 0] / z * fx + cx
    py = cam_points[..., 1] / z * fy + cy
    return torch.stack([px, py], dim=-1)


@torch.no_grad()
def calculate_in_frustum_mask(
    depth_1: torch.Tensor,
    intrinsics_1: torch.Tensor,
    c2w_1: torch.Tensor,
    depth_2: torch.Tensor,
    intrinsics_2: torch.Tensor,
    c2w_2: torch.Tensor,
    depth_atol: float = 0.1,
    depth_rtol: float = 0.0,
) -> torch.Tensor:
    """Per-pixel "valid visible" mask for the first set of views w.r.t. the second.

    For every pixel in ``depth_1`` it unprojects to a world point, reprojects that
    point into every view of the second set, and keeps the pixel when ALL three
    conditions hold:

      1. in-frustum: the reprojection lands inside the image of *any* second view
         (and in front of that camera);
      2. valid depth: the source depth in ``depth_1`` is non-zero;
      3. depth match: the reprojected depth agrees with the sampled depth of *any*
         second view, judged by ``torch.isclose(..., rtol=depth_rtol, atol=depth_atol)``.

    Intrinsics are pixel-space (fx in pixels); poses are camera-to-world (c2w).

    Args:
        depth_1: (b, v1, h, w) source-view depth.
        intrinsics_1: (b, v1, 3, 3) pixel-space intrinsics.
        c2w_1: (b, v1, 4, 4) camera-to-world.
        depth_2: (b, v2, h, w) reference-view depth.
        intrinsics_2: (b, v2, 3, 3) pixel-space intrinsics.
        c2w_2: (b, v2, 4, 4) camera-to-world.
        depth_atol: absolute tolerance for the depth-match test.
        depth_rtol: relative tolerance for the depth-match test.

    Returns:
        (b, v1, h, w) bool mask.
    """
    import einops
    import torch.nn.functional as F

    b, v1, h, w = depth_1.shape
    v2 = depth_2.shape[1]

    # Unproject source depth to world points, then reproject into all v2 views.
    points_3d = _unproject_depth(depth_1[..., None], intrinsics_1, c2w_1)  # (b, v1, h, w, 3)
    camera_points = _world_to_camera(points_3d, c2w_2)  # (b, v1, v2, h, w, 3)
    points_2d = _camera_to_pixel(camera_points, intrinsics_2)  # (b, v1, v2, h, w, 2)
    reproj_depth = camera_points[..., 2]  # (b, v1, v2, h, w)

    px = points_2d[..., 0]
    py = points_2d[..., 1]

    # Condition 1: inside the frustum of any v2 view (and in front of the camera).
    in_frustum = (
        (px > 0) & (px < w) & (py > 0) & (py < h) & (reproj_depth > 0)
    )  # (b, v1, v2, h, w)
    in_frustum = in_frustum.any(dim=2)  # (b, v1, h, w)

    # Condition 2: source depth is valid (non-zero).
    non_zero_depth = depth_1 > 1e-6  # (b, v1, h, w)

    # Condition 3: reprojected depth matches the sampled v2 depth (any v2 view).
    # grid_sample expects coords in [-1, 1] with align_corners=False, where a
    # pixel-center value `c` (here px/py already = pixel_index + 0.5 from the
    # +0.5 grid used in _unproject_depth) maps via `2*c/size - 1`. Adding another
    # +0.5 would double-count the half-pixel offset.
    gx = 2.0 * px / w - 1.0
    gy = 2.0 * py / h - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (b, v1, v2, h, w, 2)
    grid = einops.rearrange(grid, "b v1 v2 h w c -> (b v1 v2) h w c")
    depth_2_in = einops.repeat(
        depth_2, "b v2 h w -> (b v1 v2) 1 h w", v1=v1
    )  # (b*v1*v2, 1, h, w)
    sampled_depth = F.grid_sample(
        depth_2_in.float(),
        grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )  # (b*v1*v2, 1, h, w)
    sampled_depth = einops.rearrange(
        sampled_depth, "(b v1 v2) 1 h w -> b v1 v2 h w", b=b, v1=v1, v2=v2
    )
    matching_depth = torch.isclose(
        reproj_depth, sampled_depth, rtol=depth_rtol, atol=depth_atol
    )  # (b, v1, v2, h, w)
    matching_depth = matching_depth.any(dim=2)  # (b, v1, h, w)

    return in_frustum & non_zero_depth & matching_depth
