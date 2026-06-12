"""DL3DV scene-level dataset for GSDPT training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from vggt_omega.training.data.scene_dataset_base import SceneDatasetBase
from vggt_omega.training.data.transforms import c2w_opengl_to_opencv, c2w_to_w2c

logger = logging.getLogger(__name__)


class DL3DVSceneDataset(SceneDatasetBase):
    """Dataset that yields one DL3DV *scene* per ``__getitem__`` call.

    Each scene is fully self-contained: images, intrinsics, extrinsics, and
    context/target split are all computed on-the-fly. See
    :class:`SceneDatasetBase` for the shared sampling/assembly logic.
    """

    tag = "dl3dv"

    def __init__(self, scene_dirs: List[Path], *args, image_dir_name: str = "images_4", **kwargs):
        super().__init__(scene_dirs, *args, image_dir_name=image_dir_name, **kwargs)

    @staticmethod
    def _parse_scene(scene_dir: Path) -> Optional[Dict[str, Any]]:
        tf_path = scene_dir / "transforms.json"
        if not tf_path.exists():
            return None
        with open(tf_path) as f:
            data = json.load(f)

        orig_w = int(data["w"])
        orig_h = int(data["h"])
        fl_x = float(data["fl_x"])
        fl_y = float(data["fl_y"])
        cx = float(data["cx"])
        cy = float(data["cy"])

        frames = data["frames"]
        image_paths: List[str] = []
        c2ws: List[np.ndarray] = []
        w2cs: List[np.ndarray] = []

        for fr in frames:
            # file_path may say 'images/frame_XXXXX.png' — keep the name only.
            fname = Path(fr["file_path"]).name
            image_paths.append(fname)
            c2w = torch.tensor(
                np.array(fr["transform_matrix"], dtype=np.float64), dtype=torch.float64
            )
            # DL3DV stores c2w in OpenGL/nerfstudio convention -> OpenCV/COLMAP.
            c2w = c2w_opengl_to_opencv(c2w)
            w2c = c2w_to_w2c(c2w)
            c2ws.append(c2w.numpy())
            w2cs.append(w2c.numpy())

        return {
            "scene_dir": scene_dir,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "image_names": image_paths,
            "c2ws": np.stack(c2ws, axis=0),  # (N, 4, 4) OpenCV c2w
            "w2cs": np.stack(w2cs, axis=0),  # (N, 4, 4) OpenCV w2c
            "num_frames": len(frames),
        }
