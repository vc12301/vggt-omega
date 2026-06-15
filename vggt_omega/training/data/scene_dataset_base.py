"""Shared scene-level dataset base for GSDPT training.

Both :class:`DL3DVSceneDataset` and :class:`RenderingSceneDataset` yield one
*scene* per ``__getitem__`` call with an identical output schema. The only
per-dataset difference is how raw camera/image metadata is parsed off disk;
that is isolated in the subclass hook :meth:`_parse_scene`.

Unlike Depth-Anything-3, images are emitted as a single [0, 1] RGB tensor
(VGGT-Omega normalizes internally), so ``images`` doubles as model input, GT, and
the images_merger source — there is no separate ImageNet-normalized stream.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from vggt_omega.training.config import CurriculumConfig, ResolutionSchedule
from vggt_omega.training.data.transforms import (
    load_image_01,
    round_to_patch,
    scale_intrinsics,
)
from vggt_omega.training.data.view_sampler import (
    curriculum_sample_views,
    farthest_point_sampling,
    sample_target_views,
)

logger = logging.getLogger(__name__)


class SceneDatasetBase(Dataset):
    """Base class yielding one scene per ``__getitem__``.

    Subclasses must implement :meth:`_parse_scene` and set ``self.tag`` (used
    only for log messages / cache keying).
    """

    tag: str = "scene"

    def __init__(
        self,
        scene_dirs: List[Path],
        resolution_schedules: List[ResolutionSchedule],
        image_dir_name: str,
        fps_rotation_weight: float = 1.0,
        min_scene_frames: int = 20,
        is_val: bool = False,
        curriculum_config: Optional[CurriculumConfig] = None,
        step_counter=None,  # multiprocessing.Value for global_step
        val_context_gap: Optional[int] = None,
    ):
        super().__init__()
        self.resolution_schedules = resolution_schedules
        self.image_dir_name = image_dir_name
        self.fps_rotation_weight = fps_rotation_weight
        self.min_scene_frames = min_scene_frames
        self.is_val = is_val
        self.curriculum_config = curriculum_config
        self.step_counter = step_counter
        self.val_context_gap = val_context_gap

        self.scenes = self._load_or_build_index(scene_dirs)
        logger.info(f"{type(self).__name__}: {len(self.scenes)} scenes loaded (is_val={is_val})")

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_scene(scene_dir: Path) -> Optional[Dict[str, Any]]:
        """Parse one scene directory into a uniform ``meta`` dict.

        Must return a dict with keys (or ``None`` to skip the scene):
            scene_dir : Path
            orig_w, orig_h : int          original image resolution
            fl_x, fl_y, cx, cy : float    intrinsics at original resolution
            image_names : List[str]       file names under ``image_dir_name``
            c2ws : np.ndarray (N, 4, 4)   OpenCV camera-to-world (for sampling)
            w2cs : np.ndarray (N, 4, 4)   OpenCV world-to-camera (returned)
            num_frames : int
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Cached scene index
    # ------------------------------------------------------------------
    def _load_or_build_index(self, scene_dirs: List[Path]) -> List[Dict[str, Any]]:
        cache_key = self._compute_cache_key(scene_dirs)
        cache_dir = Path.home() / ".cache" / "vggt_omega_gsdpt"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"scene_index_{self.tag}_{cache_key}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    scenes = pickle.load(f)
                logger.info(f"Loaded scene index from cache: {cache_path} ({len(scenes)} scenes)")
                return scenes
            except Exception as e:
                logger.warning(f"Cache load failed ({e}), rebuilding...")

        logger.info(f"Building scene index for {len(scene_dirs)} directories...")
        scenes: List[Dict[str, Any]] = []
        for i, scene_dir in enumerate(scene_dirs):
            try:
                meta = self._parse_scene(scene_dir)
            except Exception as e:
                logger.warning(f"Skipping {scene_dir}: {e}")
                continue
            if meta is None:
                continue
            if meta["num_frames"] < self.min_scene_frames:
                continue
            scenes.append(meta)
            if (i + 1) % 1000 == 0:
                logger.info(f"  indexed {i + 1}/{len(scene_dirs)} directories...")

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(scenes, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved scene index cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache ({e}), continuing without cache")

        return scenes

    def _compute_cache_key(self, scene_dirs: List[Path]) -> str:
        h = hashlib.md5()
        h.update(self.tag.encode())
        h.update(str(self.min_scene_frames).encode())
        h.update(str(len(scene_dirs)).encode())
        for idx in [
            0,
            len(scene_dirs) // 4,
            len(scene_dirs) // 2,
            3 * len(scene_dirs) // 4,
            len(scene_dirs) - 1,
        ]:
            if 0 <= idx < len(scene_dirs):
                h.update(str(scene_dirs[idx]).encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for attempt in range(10):
            try:
                return self._getitem_impl(
                    idx if attempt == 0 else random.randint(0, len(self.scenes) - 1)
                )
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"Scene load failed (attempt {attempt + 1}): {e}")
        raise RuntimeError(f"Failed to load any scene after 10 attempts (last idx={idx})")

    def _getitem_impl(self, idx: int) -> Dict[str, Any]:
        meta = self.scenes[idx]

        # 1. Choose resolution schedule -----------------------------------
        if self.is_val:
            schedule = self.resolution_schedules[0]  # smallest resolution for val
        else:
            weights = [s.weight for s in self.resolution_schedules]
            schedule = random.choices(self.resolution_schedules, weights=weights, k=1)[0]

        target_h = round_to_patch(schedule.resolution[0])
        target_w = round_to_patch(schedule.resolution[1])
        num_context = min(schedule.context_views, meta["num_frames"] - 1)
        num_target = schedule.target_views

        # 2. Select context and target views ------------------------------
        use_curriculum = (
            not self.is_val
            and self.curriculum_config is not None
            and self.curriculum_config.enabled
        )

        if use_curriculum:
            global_step = self.step_counter.value if self.step_counter is not None else 0
            cfg = self.curriculum_config
            context_indices, target_indices = curriculum_sample_views(
                meta["c2ws"],
                num_context=num_context,
                num_target=num_target,
                global_step=global_step,
                min_gap_start=cfg.min_context_gap_start,
                min_gap_end=cfg.min_context_gap_end,
                max_gap_start=cfg.max_context_gap_start,
                max_gap_end=cfg.max_context_gap_end,
                warmup_steps=cfg.warmup_steps,
                rotation_weight=self.fps_rotation_weight,
            )
        else:
            # Validation must be reproducible: derive a per-scene deterministic
            # seed (keyed on scene_dir, so it is stable across val passes, across
            # steps, and across runs) so the same context/target views are picked
            # every time. Training without curriculum stays stochastic (seed=None).
            val_seed = None
            if self.is_val:
                digest = hashlib.md5(str(meta["scene_dir"]).encode()).hexdigest()[:8]
                val_seed = int(digest, 16)

            if self.is_val and self.val_context_gap is not None:
                gap = min(self.val_context_gap, meta["num_frames"])
                ctx_pool_c2ws = meta["c2ws"][:gap]
                context_indices = farthest_point_sampling(
                    ctx_pool_c2ws,
                    K=min(num_context, gap),
                    rotation_weight=self.fps_rotation_weight,
                    seed=val_seed,
                )
            else:
                context_indices = farthest_point_sampling(
                    meta["c2ws"],
                    K=num_context,
                    rotation_weight=self.fps_rotation_weight,
                    seed=val_seed,
                )
            target_indices = sample_target_views(
                meta["num_frames"],
                context_indices,
                num_targets=num_target,
                seed=val_seed,
            )

        if len(target_indices) == 0:
            remaining = [i for i in range(meta["num_frames"]) if i not in set(context_indices)]
            if remaining:
                target_indices = [random.choice(remaining)]
            else:
                target_indices = [context_indices[-1]]
                context_indices = context_indices[:-1]

        num_context = len(context_indices)
        all_indices = context_indices + target_indices

        # 3. Load images and build tensors --------------------------------
        # Single [0, 1] RGB stream: model input, GT, and images_merger source.
        images = []
        extrinsics = []  # w2c (OpenCV)
        img_dir = meta["scene_dir"] / self.image_dir_name

        for i in all_indices:
            img_path = img_dir / meta["image_names"][i]
            images.append(load_image_01(str(img_path), target_h, target_w))
            # w2c is precomputed in OpenCV convention by the subclass parser.
            extrinsics.append(torch.tensor(meta["w2cs"][i], dtype=torch.float32))

        images = torch.stack(images, dim=0)  # (K+M, 3, H, W) [0, 1]
        extrinsics = torch.stack(extrinsics, dim=0)  # (K+M, 4, 4) w2c

        intrinsics = scale_intrinsics(
            meta["fl_x"],
            meta["fl_y"],
            meta["cx"],
            meta["cy"],
            meta["orig_w"],
            meta["orig_h"],
            target_w,
            target_h,
        )
        intrinsics = intrinsics.unsqueeze(0).expand(len(all_indices), -1, -1).clone()

        context_images = images[:num_context].clone()  # (K, 3, H, W) [0, 1]
        target_images = images[num_context:].clone()  # (M, 3, H, W) [0, 1]

        return {
            "images": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "target_images": target_images,
            "context_images": context_images,
            "num_context": num_context,
            "num_target": len(target_indices),
            "scene_idx": idx,
            "image_h": target_h,
            "image_w": target_w,
        }
