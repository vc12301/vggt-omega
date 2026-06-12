import numpy as np
import torch


def closed_form_inverse_se3(se3, R=None, T=None):
    """Invert a batch of 3x4 or 4x4 SE(3) matrices."""
    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must have shape (N, 4, 4) or (N, 3, 4), got {se3.shape}")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_t = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_t, T)
        inverted = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_t = R.transpose(1, 2)
        top_right = -torch.bmm(R_t, T)
        inverted = torch.eye(4, device=R.device, dtype=R.dtype)[None].repeat(len(R), 1, 1)

    inverted[:, :3, :3] = R_t
    inverted[:, :3, 3:] = top_right
    return inverted


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Unproject depth maps to world-space points.

    Args:
        depth_map: (N, H, W, 1) or (N, H, W) depth in camera space.
        extrinsic: (N, 3, 4) camera-from-world matrices (OpenCV convention).
        intrinsic: (N, 3, 3) calibration matrices.

    Returns:
        (N, H, W, 3) world-space points.
    """
    if depth_map.ndim == 4:
        depth = depth_map[..., 0]
    else:
        depth = depth_map
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )
