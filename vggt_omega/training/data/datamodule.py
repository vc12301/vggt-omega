"""DataLoader builders for GSDPT training (DL3DV, rendering, or mixed)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Sampler

from vggt_omega.training.config import GSDPTTrainingConfig
from vggt_omega.training.data.dl3dv_dataset import DL3DVSceneDataset
from vggt_omega.training.data.rendering_dataset import RenderingSceneDataset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Scene discovery
# ──────────────────────────────────────────────────────────────────────
def _collect_train_dirs(data_root: str, buckets: List[str]) -> List[Path]:
    """Collect all DL3DV scene directories from training buckets (1K-11K)."""
    root = Path(data_root)
    dirs: List[Path] = []
    for bucket in buckets:
        bucket_dir = root / bucket
        if not bucket_dir.exists():
            logger.warning(f"Bucket directory not found: {bucket_dir}")
            continue
        for scene_dir in sorted(bucket_dir.iterdir()):
            if scene_dir.is_dir():
                dirs.append(scene_dir)
    return dirs


def _collect_val_dirs(val_root: str, scene_subdir: str) -> List[Path]:
    """Collect DL3DV validation scene directories from benchmark_140.

    Each scene is at ``val_root/{scene_hash}/{scene_subdir}/``.
    """
    root = Path(val_root)
    if not root.exists():
        logger.warning(f"Val root not found: {root}")
        return []
    dirs: List[Path] = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        data_dir = scene_dir / scene_subdir
        if (data_dir / "transforms.json").exists():
            dirs.append(data_dir)
        elif (scene_dir / "transforms.json").exists():
            dirs.append(scene_dir)
    return dirs


def _collect_rendering_dirs(
    data_root: str, val_split_ratio: float, camera_dir_name: str = "perspective_camera"
) -> Tuple[List[Path], List[Path]]:
    """Collect rendering_dataset scenes, split per scene_type by name.

    Layout: ``data_root/{scene_type}/{idx}/``. Within each scene_type, the
    ``idx`` dirs are sorted alphabetically and the last ``ceil(ratio * n)`` are
    assigned to validation, the rest to training. Dirs missing
    ``{camera_dir_name}/meta.json`` are skipped.
    """
    root = Path(data_root)
    train_dirs: List[Path] = []
    val_dirs: List[Path] = []
    if not root.exists():
        logger.warning(f"Rendering data root not found: {root}")
        return train_dirs, val_dirs

    for scene_type in sorted(root.iterdir()):
        if not scene_type.is_dir():
            continue
        scenes = [
            d
            for d in sorted(scene_type.iterdir())
            if d.is_dir() and (d / camera_dir_name / "meta.json").exists()
        ]
        if not scenes:
            continue
        n = len(scenes)
        n_val = math.ceil(val_split_ratio * n) if val_split_ratio > 0 else 0
        n_val = min(n_val, n)  # never exceed available
        split = n - n_val
        train_dirs.extend(scenes[:split])
        val_dirs.extend(scenes[split:])
    return train_dirs, val_dirs


# ──────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────
def _collate_fn(batch):
    assert len(batch) == 1
    item = batch[0]
    return {
        "images": item["images"].unsqueeze(0),
        "extrinsics": item["extrinsics"].unsqueeze(0),
        "intrinsics": item["intrinsics"].unsqueeze(0),
        "target_images": item["target_images"].unsqueeze(0),
        "context_images": item["context_images"].unsqueeze(0),
        "num_context": item["num_context"],
        "num_target": item["num_target"],
        "scene_idx": item["scene_idx"],
        "image_h": item["image_h"],
        "image_w": item["image_w"],
    }


# ──────────────────────────────────────────────────────────────────────
# Mixed-dataset weighted sampler
# ──────────────────────────────────────────────────────────────────────
class WeightedConcatSampler(Sampler[int]):
    """Sample indices from a ``ConcatDataset`` by per-dataset probability.

    Each draw picks a sub-dataset according to ``weights`` (normalized), then a
    uniformly random scene within it (with replacement). This decouples sampling
    frequency from the sub-datasets' relative sizes. DDP-aware only in that each
    rank uses a different, deterministic-per-epoch seed so its stream differs;
    the streams are independent with-replacement draws, NOT a disjoint partition
    of the scene set, so two ranks may draw the same scene in a step. This is the
    intended weighted-with-replacement behavior for a large scene pool.
    """

    def __init__(
        self,
        concat: ConcatDataset,
        weights: List[float],
        num_replicas: int = 1,
        rank: int = 0,
        num_samples: Optional[int] = None,
        seed: int = 0,
    ):
        self.cumulative_sizes = list(concat.cumulative_sizes)
        self.sizes = [
            (
                self.cumulative_sizes[0]
                if i == 0
                else self.cumulative_sizes[i] - self.cumulative_sizes[i - 1]
            )
            for i in range(len(self.cumulative_sizes))
        ]
        assert len(weights) == len(
            self.sizes
        ), f"weights ({len(weights)}) must match num sub-datasets ({len(self.sizes)})"
        w = torch.tensor([max(0.0, float(x)) for x in weights], dtype=torch.double)
        if w.sum() <= 0:
            w = torch.ones_like(w)
        self.weights = (w / w.sum()).tolist()
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        total = self.cumulative_sizes[-1]
        self.num_samples = (
            num_samples if num_samples is not None else math.ceil(total / self.num_replicas)
        )
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1_000_003 + self.rank)
        weights = torch.tensor(self.weights, dtype=torch.double)
        for _ in range(self.num_samples):
            d = int(torch.multinomial(weights, 1, generator=g).item())
            local = int(torch.randint(0, self.sizes[d], (1,), generator=g).item())
            offset = 0 if d == 0 else self.cumulative_sizes[d - 1]
            yield offset + local


# ──────────────────────────────────────────────────────────────────────
# Dataset construction
# ──────────────────────────────────────────────────────────────────────
def _resolve_view_sampling(config: GSDPTTrainingConfig, data_cfg):
    """Per-dataset view_sampling override, falling back to the global one."""
    return data_cfg.view_sampling if data_cfg.view_sampling is not None else config.view_sampling


def _resolve_curriculum(config: GSDPTTrainingConfig, data_cfg):
    """Per-dataset curriculum override, falling back to the global one."""
    return data_cfg.curriculum if data_cfg.curriculum is not None else config.curriculum


def _build_dl3dv_datasets(
    config: GSDPTTrainingConfig, step_counter=None
) -> Tuple[Dataset, Dataset]:
    """Construct (train, val) DL3DVSceneDataset pair from config."""
    train_dirs = _collect_train_dirs(config.data.data_root, config.data.train_buckets)
    if config.data.max_train_scenes is not None:
        train_dirs = train_dirs[: config.data.max_train_scenes]
    val_dirs = _collect_val_dirs(config.data.val_root, config.data.val_scene_subdir)
    if config.data.max_val_scenes is not None:
        val_dirs = val_dirs[: config.data.max_val_scenes]
    logger.info(f"[dl3dv] Train scenes: {len(train_dirs)}, Val scenes: {len(val_dirs)}")

    vs = _resolve_view_sampling(config, config.data)
    cur = _resolve_curriculum(config, config.data)
    train_ds = DL3DVSceneDataset(
        scene_dirs=train_dirs,
        resolution_schedules=vs.resolution_schedules,
        image_dir_name=config.data.image_dir_name,
        fps_rotation_weight=vs.fps_rotation_weight,
        min_scene_frames=vs.min_scene_frames,
        is_val=False,
        curriculum_config=cur,
        step_counter=step_counter,
    )
    val_ds = DL3DVSceneDataset(
        scene_dirs=val_dirs,
        resolution_schedules=vs.resolution_schedules,
        image_dir_name=config.data.image_dir_name,
        fps_rotation_weight=vs.fps_rotation_weight,
        min_scene_frames=vs.min_scene_frames,
        is_val=True,
        val_context_gap=config.data.val_context_gap,
    )
    return train_ds, val_ds


def _build_rendering_datasets(
    config: GSDPTTrainingConfig, step_counter=None
) -> Tuple[Dataset, Dataset]:
    """Construct (train, val) RenderingSceneDataset pair from config."""
    rc = config.rendering_data
    train_dirs, val_dirs = _collect_rendering_dirs(
        rc.data_root, rc.val_split_ratio, rc.camera_dir_name
    )
    if rc.max_train_scenes is not None:
        train_dirs = train_dirs[: rc.max_train_scenes]
    if rc.max_val_scenes is not None:
        val_dirs = val_dirs[: rc.max_val_scenes]
    logger.info(f"[rendering] Train scenes: {len(train_dirs)}, Val scenes: {len(val_dirs)}")

    vs = _resolve_view_sampling(config, rc)
    cur = _resolve_curriculum(config, rc)
    common = dict(
        resolution_schedules=vs.resolution_schedules,
        image_dir_name=rc.image_dir_name,
        camera_dir_name=rc.camera_dir_name,
        fps_rotation_weight=vs.fps_rotation_weight,
        min_scene_frames=vs.min_scene_frames,
    )
    train_ds = RenderingSceneDataset(
        scene_dirs=train_dirs,
        is_val=False,
        curriculum_config=cur,
        step_counter=step_counter,
        **common,
    )
    val_ds = RenderingSceneDataset(
        scene_dirs=val_dirs,
        is_val=True,
        val_context_gap=rc.val_context_gap,
        **common,
    )
    return train_ds, val_ds


def _make_train_loader(train_ds, sampler, num_workers, pin_memory) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=1,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def _make_val_loader(val_ds, world_size, rank, num_workers, pin_memory) -> DataLoader:
    use_dist = world_size > 1
    sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if use_dist
        else None
    )
    return DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=min(num_workers, 4),
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
        drop_last=False,
    )


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────
def build_dataloaders(
    config: GSDPTTrainingConfig,
    world_size: int = 1,
    rank: int = 0,
    step_counter=None,
) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """Build the train loader and a per-dataset dict of validation loaders.

    ``config.dataset_mode`` selects:
      - "dl3dv"     : DL3DV only (default)
      - "rendering" : rendering_dataset only
      - "mixed"     : both, train via WeightedConcatSampler(config.mix_ratio);
                      validation kept separate per dataset.

    Returns:
        (train_loader, {dataset_name: val_loader})
    """
    mode = getattr(config, "dataset_mode", "dl3dv")
    use_dist = world_size > 1
    val_loaders: Dict[str, DataLoader] = {}
    dl_cfg = config.data
    rd_cfg = config.rendering_data

    if mode == "dl3dv":
        train_ds, val_ds = _build_dl3dv_datasets(config, step_counter)
        sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            if use_dist
            else None
        )
        train_loader = _make_train_loader(train_ds, sampler, dl_cfg.num_workers, dl_cfg.pin_memory)
        val_loaders["dl3dv"] = _make_val_loader(
            val_ds, world_size, rank, dl_cfg.num_workers, dl_cfg.pin_memory
        )

    elif mode == "rendering":
        train_ds, val_ds = _build_rendering_datasets(config, step_counter)
        sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            if use_dist
            else None
        )
        train_loader = _make_train_loader(train_ds, sampler, rd_cfg.num_workers, rd_cfg.pin_memory)
        val_loaders["rendering"] = _make_val_loader(
            val_ds, world_size, rank, rd_cfg.num_workers, rd_cfg.pin_memory
        )

    elif mode == "mixed":
        dl_train, dl_val = _build_dl3dv_datasets(config, step_counter)
        rd_train, rd_val = _build_rendering_datasets(config, step_counter)
        concat = ConcatDataset([dl_train, rd_train])
        sampler = WeightedConcatSampler(
            concat,
            weights=list(config.mix_ratio),
            num_replicas=world_size,
            rank=rank,
            seed=config.seed,
        )
        # Mixed train loader draws from both sub-datasets; use the DL3DV worker
        # setting for the shared loader, each val loader uses its own.
        train_loader = _make_train_loader(concat, sampler, dl_cfg.num_workers, dl_cfg.pin_memory)
        val_loaders["dl3dv"] = _make_val_loader(
            dl_val, world_size, rank, dl_cfg.num_workers, dl_cfg.pin_memory
        )
        val_loaders["rendering"] = _make_val_loader(
            rd_val, world_size, rank, rd_cfg.num_workers, rd_cfg.pin_memory
        )

    else:
        raise ValueError(f"Unknown dataset_mode: {mode!r}")

    return train_loader, val_loaders
