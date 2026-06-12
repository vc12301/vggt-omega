"""Image preprocessing and camera transforms for GSDPT training data.

VGGT-Omega's Aggregator applies ResNet normalization internally, so images are
loaded as plain [0, 1] RGB (no ImageNet normalization, unlike Depth-Anything-3).
The single [0, 1] image tensor serves as model input, GT, and the images_merger
RGB injection.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# VGGT-Omega uses patch_size 16 (DA3 used 14).
PATCH_SIZE = 16


def round_to_patch(val: int, patch: int = PATCH_SIZE) -> int:
    """Round value down to a multiple of the patch size (min one patch)."""
    return max(patch, (val // patch) * patch)


def load_image_01(path: str, target_h: int, target_w: int) -> torch.Tensor:
    """Load an image as a [0, 1] CHW float tensor resized to (target_h, target_w)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # CHW, [0, 1]


def scale_intrinsics(
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
) -> torch.Tensor:
    """Build a 3x3 pixel-space intrinsic matrix scaled to the target resolution.

    transforms.json stores intrinsics at the *original* resolution; images may be
    pre-downsampled (e.g. ``images_4/``) and are further resized to the training
    resolution, so the combined scale factor is applied here.

    Returns:
        (3, 3) float32 intrinsics tensor.
    """
    sx = target_w / orig_w
    sy = target_h / orig_h

    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = fl_x * sx
    K[1, 1] = fl_y * sy
    K[0, 2] = cx * sx
    K[1, 2] = cy * sy
    return K


def c2w_opengl_to_opencv(c2w: torch.Tensor) -> torch.Tensor:
    """Convert c2w from OpenGL/nerfstudio convention to OpenCV/COLMAP convention.

    DL3DV transforms.json stores c2w in nerfstudio (OpenGL) convention where the
    Y-axis points up and the Z-axis points backward. VGGT-Omega uses the OpenCV
    convention (Y down, Z forward). Conversion flips the Y and Z columns.
    """
    flip = torch.tensor([1, -1, -1, 1], dtype=c2w.dtype, device=c2w.device)
    return c2w * flip[None, :]  # broadcast: flip columns 1, 2


def c2w_to_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """Convert camera-to-world (4x4) to world-to-camera (4x4)."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = torch.eye(4, dtype=c2w.dtype)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c
