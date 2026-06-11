"""Video inference pipeline for VGGT-Omega: point cloud (PLY) + depth map outputs.

Extracts frames from a video at a target FPS, then runs the shared pipeline
from inference_pipeline.py. Outputs:

    {output_dir}/
        input_images/{i:06d}.png        extracted frames
        depth/{stem}.npy + {stem}.png   raw + colorized depth per frame
        pcd/pointcloud_{video_stem}.ply RGB point cloud (binary PLY)
        cameras.npz, depth_conf.npy

Usage:
    python inference_pipeline_video.py --video examples/snow_lift.mp4 --fps 1.0
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from inference_pipeline import add_common_args, run_pipeline


def _get_video_rotation(video_path: str) -> int:
    """Read rotation metadata; return clockwise rotation to apply (0/90/180/270).

    Phone-shot portrait videos often store frames in landscape orientation and
    mark the rotation in metadata, which OpenCV does not apply automatically.
    """
    try:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
                cap.release()
                if rot in (90, 180, 270):
                    return rot
            else:
                cap.release()
    except Exception:
        pass

    # Fall back to ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe", "-loglevel", "error", "-select_streams", "v:0",
                "-show_entries", "stream=side_data_list:stream_tags=rotate",
                "-of", "json", video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout or "{}")
        for stream in data.get("streams", []):
            tags = stream.get("tags") or {}
            if "rotate" in tags:
                return int(tags["rotate"]) % 360
            for sd in stream.get("side_data_list") or []:
                if "rotation" in sd:
                    # ffprobe reports the transform already applied to the
                    # stored frames; negate to get the display rotation.
                    return (-int(sd["rotation"])) % 360
    except Exception:
        pass

    return 0


def _rotate_frame(frame: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def extract_frames(video_path: str, frames_dir: str, target_fps: float) -> list:
    os.makedirs(frames_dir, exist_ok=True)
    rotation = _get_video_rotation(video_path)
    if rotation:
        print(f"Detected video rotation metadata: {rotation}° (will apply to frames)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    # Disable OpenCV's auto-rotation to avoid applying the rotation twice
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        except Exception:
            pass
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / target_fps))
    actual_fps = video_fps / frame_interval
    print(
        f"Video FPS={video_fps:.2f}, total={total_frames}, "
        f"sampling every {frame_interval} frame(s) → actual fps={actual_fps:.2f}"
    )

    paths, frame_count, saved = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_count % frame_interval == 0:
            if rotation:
                frame = _rotate_frame(frame, rotation)
            p = os.path.join(frames_dir, f"{saved:06d}.png")
            cv2.imwrite(p, frame)
            paths.append(p)
            saved += 1
        frame_count += 1
    cap.release()
    if not paths:
        raise RuntimeError("No frames extracted from video")
    print(f"✅ Extracted {saved} frames → {frames_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Target frame sampling rate")
    parser.add_argument("--frames_dir", default=None,
                        help="Directory for extracted frames (default: {output_dir}/input_images)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: inference_pipeline_video_outputs/{video_stem})")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames used for inference")
    add_common_args(parser)
    args = parser.parse_args()

    video_stem = Path(args.video).stem
    output_dir = args.output_dir or os.path.join("inference_pipeline_video_outputs", video_stem)
    frames_dir = args.frames_dir or os.path.join(output_dir, "input_images")

    frame_paths = extract_frames(args.video, frames_dir, args.fps)
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
        print(f"Using first {len(frame_paths)} frames")

    run_pipeline(args, frame_paths, output_dir, ply_suffix=f"_{video_stem}")


if __name__ == "__main__":
    main()
