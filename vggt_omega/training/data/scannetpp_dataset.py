"""ScanNet++ scene-level dataset for GSDPT training.

On-disk layout (one directory per scene under ``data_root``)::

    {data_root}/{scene_id}/
        dslr/resized_undistorted_images/*.JPG          # perspective frames
        dslr/nerfstudio/transforms_undistorted.json    # nerfstudio camera params

The transforms file is the standard nerfstudio format — structurally identical to
DL3DV's ``transforms.json`` (top-level ``fl_x/fl_y/cx/cy/w/h``; per-frame
``file_path`` + 4x4 ``transform_matrix`` c2w in the OpenGL/nerfstudio convention).
Differences from DL3DV: the transforms/image paths are nested under ``dslr/``, and
each frame carries an ``is_bad`` flag (blurry / unusable captures) that is filtered
out here. As with DL3DV, c2w is converted OpenGL -> OpenCV before being stored.
"""

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


class ScanNetppSceneDataset(SceneDatasetBase):
    """Dataset yielding one ScanNet++ *scene* per ``__getitem__`` call.

    Output schema is identical to :class:`DL3DVSceneDataset`; only the on-disk
    parsing differs (nested ``dslr/`` paths and ``is_bad`` frame filtering).
    """

    tag = "scannetpp"

    def __init__(
        self,
        scene_dirs: List[Path],
        *args,
        image_dir_name: str = "dslr/resized_undistorted_images",
        transforms_subpath: str = "dslr/nerfstudio/transforms_undistorted.json",
        **kwargs,
    ):
        self.transforms_subpath = transforms_subpath
        super().__init__(scene_dirs, *args, image_dir_name=image_dir_name, **kwargs)

    def _parse_scene(self, scene_dir: Path) -> Optional[Dict[str, Any]]:  # type: ignore[override]
        tf_path = scene_dir / self.transforms_subpath
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
            # Skip frames flagged as bad (blurry / unusable captures).
            if fr.get("is_bad"):
                continue
            # file_path is already a bare filename (e.g. 'DSC00850.JPG').
            fname = Path(fr["file_path"]).name
            image_paths.append(fname)
            c2w = torch.tensor(
                np.array(fr["transform_matrix"], dtype=np.float64), dtype=torch.float64
            )
            # ScanNet++ stores c2w in OpenGL/nerfstudio convention -> OpenCV/COLMAP.
            c2w = c2w_opengl_to_opencv(c2w)
            w2c = c2w_to_w2c(c2w)
            c2ws.append(c2w.numpy())
            w2cs.append(w2c.numpy())

        if not image_paths:
            return None

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
            "num_frames": len(image_paths),
        }
