# Standard 3DGS PLY layout (as in graphdeco-inria/gaussian-splatting and the
# Depth-Anything-3 exporter): opacity is stored as a logit and scales in log
# space; rotations are unit WXYZ quaternions.

import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def export_gaussians_ply(gaussians, ply_path: str, save_sh_dc_only: bool = True) -> None:
    """Write a Gaussians dataclass (batch size 1) to a standard 3DGS PLY file."""
    if gaussians.means.shape[0] != 1:
        raise ValueError("export_gaussians_ply expects batch_size=1 Gaussians")

    means = gaussians.means[0].detach().float().cpu().numpy()
    scales = gaussians.scales[0].detach().float().cpu().clamp(min=1e-10).log().numpy()
    rotations = gaussians.rotations[0].detach().float().cpu().numpy()
    opacities = inverse_sigmoid(gaussians.opacities[0].detach().float().cpu()).numpy()
    harmonics = gaussians.harmonics[0].detach().float().cpu()
    f_dc = harmonics[..., 0].numpy()
    f_rest = harmonics[..., 1:].flatten(start_dim=1).numpy()

    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    columns = [means, np.zeros_like(means), f_dc]
    if not save_sh_dc_only and f_rest.shape[1] > 0:
        fields += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        columns.append(f_rest)
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    columns += [opacities[:, None], scales, rotations]

    attributes = np.concatenate(columns, axis=1).astype(np.float32)
    vertex = np.empty(attributes.shape[0], dtype=[(field, "f4") for field in fields])
    for i, field in enumerate(fields):
        vertex[field] = attributes[:, i]

    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(ply_path)
