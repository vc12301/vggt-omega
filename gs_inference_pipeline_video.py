"""Video GS reconstruction pipeline for VGGT-Omega.

Samples frames from a video at a target FPS, then runs the shared GS pipeline
from test_gs_inference.py (the same default model load). Outputs:

    {output_dir}/
        input_images/{i:06d}.png        extracted frames
        depth/{stem}.npy + {stem}.png   raw + colorized depth per frame
        depth_conf.npy, cameras.npz
        pcd/pointcloud_{video_stem}.ply RGB point cloud (binary PLY)
        gaussians.ply                   standard 3DGS PLY
        compare/{i:03d}.png             input view vs. re-rendered view (unless --no_compare)
        vggt-o-{video_stem}-render.mp4  interpolated camera-path rendering (unless --no_render_video)

Usage:
    python gs_inference_pipeline_video.py --video examples/snow_lift.mp4 --fps 1.0
    python gs_inference_pipeline_video.py --video clip.mp4 \
        --gsdpt_checkpoint checkpoints/gsdpt-24000-psnr20.17.pt
"""

import argparse
import os
from pathlib import Path

from inference_pipeline import add_common_args
from inference_pipeline_video import extract_frames
from test_gs_inference import add_gs_args, run_gs_pipeline


def main() -> None:

    scene="1m-1m40s_compressed"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=f"/data-nas/experiments/yemu/workspace/da3/test_data/jingzhuang/{scene}.mp4", help="Input video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Target frame sampling rate")
    parser.add_argument("--frames_dir", default=None,
                        help="Directory for extracted frames (default: {output_dir}/input_images)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: gs_inference_pipeline_video_outputs/{video_stem})")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames used for inference")
    add_gs_args(parser)
    add_common_args(parser)
    parser.set_defaults(conf_percentile=0.0)  # keep all gaussians/points unless asked to filter
    args = parser.parse_args()

    video_stem = Path(args.video).stem
    output_dir = args.output_dir or os.path.join("gs_inference_pipeline_video_outputs", video_stem)
    frames_dir = args.frames_dir or os.path.join(output_dir, "input_images")

    frame_paths = extract_frames(args.video, frames_dir, args.fps)
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
        print(f"Using first {len(frame_paths)} frames")

    run_gs_pipeline(
        args,
        frame_paths,
        output_dir,
        scene_name=args.scene or video_stem,
        ply_suffix=f"_{video_stem}",
    )


if __name__ == "__main__":
    main()
