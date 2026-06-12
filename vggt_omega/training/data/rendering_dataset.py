"""Synthetic ``rendering_dataset`` (pano-perspective renders) for GSDPT training.

On-disk layout::

    {data_root}/{scene_type}/{idx}/
        perspective_frames/frame_XXXXX.png        # PNG frames
        perspective_camera/meta.json              # frames_processed, resolution
        perspective_camera/Camera{N}.json         # per-frame intrinsics + w2c
        perspective_camera/extrinsic_matrices.json

Camera JSON stores ``intrinsics`` and ``extrinsic_world2cam`` (4x4), already in
the OpenCV camera convention (x=right, y=down, z=forward) — matching VGGT-Omega,
so **no convention flip is applied** (unlike DL3DV).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vggt_omega.training.data.scene_dataset_base import SceneDatasetBase

logger = logging.getLogger(__name__)


class RenderingSceneDataset(SceneDatasetBase):
    """Dataset yielding one rendering_dataset *scene* per ``__getitem__`` call.

    Output schema is identical to :class:`DL3DVSceneDataset`. The only difference
    is on-disk parsing (see module docstring).
    """

    tag = "rendering"

    def __init__(
        self,
        scene_dirs: List[Path],
        *args,
        image_dir_name: str = "perspective_frames",
        camera_dir_name: str = "perspective_camera",
        **kwargs,
    ):
        self.camera_dir_name = camera_dir_name
        super().__init__(scene_dirs, *args, image_dir_name=image_dir_name, **kwargs)

    def _parse_scene(self, scene_dir: Path) -> Optional[Dict[str, Any]]:  # type: ignore[override]
        cam_dir = scene_dir / self.camera_dir_name
        meta_path = cam_dir / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            meta = json.load(f)

        num_frames = int(meta["frames_processed"])
        res = meta["output_resolution"]
        orig_w = int(res["width"])
        orig_h = int(res["height"])

        # Intrinsics: identical across frames in a scene — read from Camera0.json.
        cam0_path = cam_dir / "Camera0.json"
        with open(cam0_path) as f:
            cam0 = json.load(f)
        intr = cam0["intrinsics"]
        fl_x = float(intr["fx"])
        fl_y = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])

        # Prefer the combined extrinsics file (one read); fall back per-frame.
        w2cs: List[np.ndarray] = []
        combined = cam_dir / "extrinsic_matrices.json"
        if combined.exists():
            with open(combined) as f:
                mats = json.load(f)
            if len(mats) < num_frames:
                num_frames = len(mats)
            for i in range(num_frames):
                w2cs.append(np.array(mats[i], dtype=np.float64))
        else:
            for i in range(num_frames):
                cp = cam_dir / f"Camera{i}.json"
                with open(cp) as f:
                    c = json.load(f)
                w2cs.append(np.array(c["extrinsic_world2cam"], dtype=np.float64))

        w2cs_arr = np.stack(w2cs, axis=0)  # (N, 4, 4) OpenCV w2c
        # c2w = inverse(w2c); needed by the view samplers for camera centers.
        c2ws_arr = np.linalg.inv(w2cs_arr)  # (N, 4, 4) OpenCV c2w

        image_names = [f"frame_{i:05d}.png" for i in range(num_frames)]

        return {
            "scene_dir": scene_dir,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "image_names": image_names,
            "c2ws": c2ws_arr,
            "w2cs": w2cs_arr,
            "num_frames": num_frames,
        }
