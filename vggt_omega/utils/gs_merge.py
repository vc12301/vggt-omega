"""Opacity-weighted voxel merging of world-space Gaussians.

Compresses the per-pixel Gaussian count by collapsing every Gaussian that falls
in the same spatial voxel into a single Gaussian, using ``opacity`` as the merge
weight (weight == opacity, sharing its gradient). Operating on the post-adapter
``Gaussians`` dataclass keeps the GS head and its checkpoints unchanged.

The grouping (floor + unique) is non-differentiable, but gradients flow through
the weighted averages back to means / scales / harmonics / rotations / opacities.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import torch

from vggt_omega.models.gs_adapter import Gaussians

if TYPE_CHECKING:
    from vggt_omega.training.config import VoxelMergeConfig


def voxel_merge_gaussians(
    gaussians: Gaussians,
    voxel_size: float,
    filter_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Gaussians:
    """Merge Gaussians sharing a voxel into one, weighted by opacity.

    Args:
        gaussians: ``Gaussians`` with batch size 1 (the training/inference
            invariant). Fields: means (1, M, 3), harmonics (1, M, 3, d_sh),
            opacities (1, M), scales (1, M, 3), rotations (1, M, 4) WXYZ.
        voxel_size: Edge length of the cubic voxels in world units.
        filter_mask: Optional bool tensor (M,) — True = keep. Applied before the
            voxel merge (e.g. confidence / edge filtering).
        eps: Floor for the per-voxel weight sum and the quaternion norm.

    Returns:
        A new ``Gaussians`` (batch size 1) with K <= M merged Gaussians. If no
        merging happens (every kept Gaussian is alone in its voxel) the
        (optionally filtered) Gaussians are returned without re-aggregation.
    """
    if gaussians.means.shape[0] != 1:
        raise ValueError("voxel_merge_gaussians expects batch_size=1 Gaussians")

    means = gaussians.means[0]  # (M, 3)
    harmonics = gaussians.harmonics[0]  # (M, 3, d_sh)
    opacities = gaussians.opacities[0]  # (M,)
    scales = gaussians.scales[0]  # (M, 3)
    rotations = gaussians.rotations[0]  # (M, 4), WXYZ

    if filter_mask is not None:
        fm = filter_mask.reshape(-1).to(means.device).bool()
        means, harmonics, opacities = means[fm], harmonics[fm], opacities[fm]
        scales, rotations = scales[fm], rotations[fm]

    num_in = means.shape[0]

    # --- Voxel indices: floor, shift to non-negative, flatten 3D -> 1D ---
    voxel_idx = torch.floor(means / voxel_size).long()  # (M, 3)
    voxel_idx = voxel_idx - voxel_idx.min(dim=0).values
    dims = voxel_idx.max(dim=0).values + 1  # (3,)
    flat = voxel_idx[:, 0] * (dims[1] * dims[2]) + voxel_idx[:, 1] * dims[2] + voxel_idx[:, 2]

    _, inv = torch.unique(flat, return_inverse=True)
    num_out = int(inv.max().item()) + 1 if num_in > 0 else 0

    # Nothing to merge: every kept Gaussian is alone in its voxel.
    if num_out == num_in:
        return Gaussians(
            means=means[None],
            harmonics=harmonics[None],
            opacities=opacities[None],
            scales=scales[None],
            rotations=rotations[None],
        )

    # --- Opacity weights (== weights, shared gradient) ---
    w = opacities.clamp_min(0.0)  # (M,)
    wsum = torch.zeros(num_out, device=means.device, dtype=means.dtype)
    wsum.index_add_(0, inv, w)
    wsum = wsum.clamp_min(eps).unsqueeze(-1)  # (K, 1)

    def _weighted_mean(values: torch.Tensor) -> torch.Tensor:
        """Σ(values · w) / Σw over the leading (M) dim; values is (M, C)."""
        c = values.shape[1]
        acc = torch.zeros(num_out, c, device=values.device, dtype=values.dtype)
        acc.index_add_(0, inv, values * w.unsqueeze(-1))
        return acc / wsum

    merged_means = _weighted_mean(means)  # (K, 3)
    merged_scales = _weighted_mean(scales)  # (K, 3)

    # harmonics (M, 3, d_sh) -> flatten channels for the weighted mean, then reshape.
    d_sh = harmonics.shape[-1]
    merged_sh = _weighted_mean(harmonics.reshape(num_in, 3 * d_sh)).reshape(num_out, 3, d_sh)

    # opacities: Σ(w²) / Σw.
    op_acc = torch.zeros(num_out, device=means.device, dtype=means.dtype)
    op_acc.index_add_(0, inv, w * w)
    merged_opacities = op_acc / wsum.squeeze(-1)  # (K,)

    # rotations: weighted sum then L2-normalize.
    quat_acc = _weighted_mean(rotations) * wsum  # undo the /wsum -> plain weighted sum
    merged_quats = quat_acc / quat_acc.norm(dim=-1, keepdim=True).clamp_min(eps)

    return Gaussians(
        means=merged_means[None],
        harmonics=merged_sh[None],
        opacities=merged_opacities[None],
        scales=merged_scales[None],
        rotations=merged_quats[None],
    )


def get_curriculum_voxel_size(step: int, cfg: "VoxelMergeConfig") -> float:
    """Sample a voxel size for ``step`` under the curriculum schedule.

    The lower bound is fixed at ``voxel_size_min``; the upper bound ramps
    linearly from ``voxel_size_max_start`` to ``voxel_size_max_end`` over
    ``warmup_steps``. The returned size is drawn uniformly from [lo, hi], so with
    ``voxel_size_max_start == voxel_size_max_end == voxel_size_min`` it degenerates
    to the fixed value.
    """
    t = min(step / max(1, cfg.warmup_steps), 1.0)
    lo = cfg.voxel_size_min
    hi = cfg.voxel_size_max_start + t * (cfg.voxel_size_max_end - cfg.voxel_size_max_start)
    if hi <= lo:
        return lo
    return random.uniform(lo, hi)
