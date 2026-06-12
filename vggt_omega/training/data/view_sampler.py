"""View sampling: FPS for context views, uniform sampling for targets, curriculum.

Ported verbatim from Depth-Anything-3. GT camera poses are used ONLY to select
which frames become context/target views; VGGT-Omega never consumes them as
input (it is pose-free).
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np


def geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic distance between two 3x3 rotation matrices."""
    R_rel = R1.T @ R2
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.acos(cos_angle)


def compute_pairwise_distances(
    c2ws: np.ndarray,
    rotation_weight: float = 1.0,
) -> np.ndarray:
    """Compute pairwise 6-DoF distances between cameras.

    Args:
        c2ws: (N, 4, 4) camera-to-world matrices.
        rotation_weight: weight for rotation component.

    Returns:
        (N, N) distance matrix.
    """
    N = len(c2ws)
    translations = c2ws[:, :3, 3]  # (N, 3)
    rotations = c2ws[:, :3, :3]  # (N, 3, 3)

    # Translation distances
    t_diff = translations[:, None] - translations[None, :]  # (N, N, 3)
    t_dist = np.linalg.norm(t_diff, axis=-1)  # (N, N)

    # Rotation distances
    r_dist = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = geodesic_distance(rotations[i], rotations[j])
            r_dist[i, j] = d
            r_dist[j, i] = d

    return t_dist + rotation_weight * r_dist


def farthest_point_sampling(
    c2ws: np.ndarray,
    K: int,
    rotation_weight: float = 1.0,
    seed: int | None = None,
) -> List[int]:
    """Select K views using Farthest Point Sampling in 6-DoF space.

    Args:
        c2ws: (N, 4, 4) camera-to-world matrices.
        K: number of context views.
        rotation_weight: balance between translation and rotation distances.
        seed: optional RNG seed for the first selection.

    Returns:
        List of K selected indices.
    """
    N = len(c2ws)
    K = min(K, N)

    dist_matrix = compute_pairwise_distances(c2ws, rotation_weight)

    rng = random.Random(seed)
    selected = [rng.randint(0, N - 1)]
    min_dists = dist_matrix[selected[0]].copy()  # (N,)

    for _ in range(K - 1):
        # Pick the point that is farthest from all already-selected points.
        # Set already-selected to -inf so they won't be picked again.
        candidates = min_dists.copy()
        for s in selected:
            candidates[s] = -1.0
        next_idx = int(np.argmax(candidates))
        selected.append(next_idx)
        # Update min distances
        np.minimum(min_dists, dist_matrix[next_idx], out=min_dists)

    return selected


def sample_target_views(
    total_frames: int,
    context_indices: List[int],
    num_targets: int,
    seed: int | None = None,
) -> List[int]:
    """Sample target views uniformly from frames not in the context set.

    Args:
        total_frames: total number of frames in the scene.
        context_indices: indices already selected as context.
        num_targets: desired number of target views.
        seed: optional RNG seed.

    Returns:
        List of target view indices.
    """
    context_set = set(context_indices)
    remaining = [i for i in range(total_frames) if i not in context_set]

    if len(remaining) == 0:
        return []

    num_targets = min(num_targets, len(remaining))

    if num_targets >= len(remaining):
        return remaining

    # Uniform sub-sampling (every k-th frame)
    step = max(1, len(remaining) // num_targets)
    candidates = remaining[::step]

    if len(candidates) > num_targets:
        rng = random.Random(seed)
        candidates = rng.sample(candidates, num_targets)

    return sorted(candidates)


def curriculum_sample_views(
    c2ws: np.ndarray,
    num_context: int,
    num_target: int,
    global_step: int,
    min_gap_start: int,
    min_gap_end: int,
    max_gap_start: int,
    max_gap_end: int,
    warmup_steps: int,
    rotation_weight: float = 1.0,
) -> Tuple[List[int], List[int]]:
    """Curriculum-based view sampling with warmup on frame gap.

    Instead of FPS over all frames (which produces large baselines), selects a
    local window [start, start+gap] and samples within it. The gap grows linearly
    from small (easy) to large (hard) over training.

    Args:
        c2ws: (N, 4, 4) camera-to-world matrices for all frames.
        num_context: number of context views (K).
        num_target: number of target views (M).
        global_step: current training step.
        min_gap_start/end: min frame gap range (linearly interpolated).
        max_gap_start/end: max frame gap range (linearly interpolated).
        warmup_steps: steps to reach final gap values.
        rotation_weight: FPS rotation distance weight.

    Returns:
        (context_indices, target_indices): lists of frame indices.
    """
    N = len(c2ws)

    # 1. Compute current gap range (linear warmup)
    t = min(global_step / max(1, warmup_steps), 1.0)
    min_gap = min_gap_start + t * (min_gap_end - min_gap_start)
    max_gap = max_gap_start + t * (max_gap_end - max_gap_start)

    # 2. Random gap within range, clamped to scene length
    gap = random.randint(int(min_gap), int(max_gap))
    gap = min(gap, N - 1)

    # 3. Random start position (ensure end stays in bounds)
    max_start = max(0, N - 1 - gap)
    start = random.randint(0, max_start)
    end = start + gap  # inclusive

    # 4. FPS within [start, end] window for context views
    window_indices = list(range(start, end + 1))
    window_c2ws = c2ws[start : end + 1]

    K = min(num_context, len(window_indices))
    fps_local = farthest_point_sampling(window_c2ws, K, rotation_weight=rotation_weight)
    context_indices = [window_indices[i] for i in fps_local]

    # 5. Random target views within window (excluding context)
    context_set = set(context_indices)
    remaining = [i for i in window_indices if i not in context_set]
    M = min(num_target, len(remaining))
    if M > 0 and len(remaining) >= M:
        target_indices = sorted(random.sample(remaining, M))
    else:
        target_indices = sorted(remaining)

    # Fallback: if no targets available, take last context view as target
    if not target_indices:
        target_indices = [context_indices[-1]]
        context_indices = context_indices[:-1]

    return context_indices, target_indices
