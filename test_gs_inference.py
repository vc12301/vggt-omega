"""Test pipeline for the VGGT-Omega 3DGS branch: gaussian PLY + rendered video.

Runs VGGT-Omega with the GSDPT head enabled (untrained: released checkpoints
carry no GS weights, so the head keeps its initialization — Gaussians at the
unprojected depth points, color = pixel RGB, opacity 0.12, ~0.25 px footprint),
then writes:

    {output_dir}/
        gaussians.ply       standard 3DGS PLY (opacity logit, log scales, WXYZ rot)
        compare/{i}.png     input view vs. re-rendered view, side by side
        render.mp4          interpolated camera-path rendering

Usage:
    python test_gs_inference.py --image_dir path/to/images
"""

import argparse
import os

import cv2
import imageio
import numpy as np
import torch

from inference_pipeline import add_common_args, collect_images, depth_edge, load_model
from vggt_omega.models.gs_adapter import Gaussians
from vggt_omega.utils.gs_ply import export_gaussians_ply
from vggt_omega.utils.gs_renderer import interpolate_camera_path, render_gaussians
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


DEFAULT_IMAGE_DIR = "/data-nas/experiments/yemu/workspace/HY-World-2.0/inference_output/_frames/scene"
# DEFAULT_IMAGE_DIR = "demo_outputs/input_images_20260610_214602_134139/images"

def print_sanity_stats(predictions: dict, model) -> None:
    """Report the raw head outputs; with an untrained head these equal the init biases."""
    layout = model.gs_head.channel_layout
    raw_gs = predictions["raw_gs"][0]
    opacity = predictions["gs_opacity"][0]

    scale_raw = raw_gs[..., layout["scales"]]
    smin, smax = model.gs_adapter.gaussian_scale_min, model.gs_adapter.gaussian_scale_max
    scale_act = smin + (smax - smin) * scale_raw.sigmoid()

    target_scale = model.gs_head.init_pixel_size / (2.0 * model.gs_adapter.scale_multiplier)
    print("GS head sanity stats (untrained head should match the init values):")
    print(f"  opacity        mean={opacity.mean():.4f} (init: {model.gs_head.init_opacity})")
    print(f"  |xy_offset|    max={raw_gs[..., layout['xy_offset']].abs().max():.6f} (init: 0)")
    print(f"  |depth_offset| max={raw_gs[..., layout['depth_offset']].abs().max():.6f} (init: 0)")
    print(f"  |residual_sh|  max={raw_gs[..., layout['sh']].abs().max():.6f} (init: 0)")
    print(f"  quat (xyzw)    mean={raw_gs[..., layout['quaternion']].mean(dim=(0, 1, 2)).tolist()} (init: [0,0,0,1])")
    print(f"  scale (act)    mean={scale_act.mean():.4f} (init target: {target_scale:.2f})")


def select_gaussians(gaussians: Gaussians, mask: torch.Tensor) -> Gaussians:
    """Keep only the gaussians where the per-gaussian mask (M,) is True."""
    return Gaussians(
        means=gaussians.means[:, mask],
        harmonics=gaussians.harmonics[:, mask],
        opacities=gaussians.opacities[:, mask],
        scales=gaussians.scales[:, mask],
        rotations=gaussians.rotations[:, mask],
    )


def conf_keep_mask(depth_conf: torch.Tensor, percentile: float) -> torch.Tensor:
    """Boolean (M,) mask keeping gaussians at or above the given confidence percentile."""
    conf = depth_conf.reshape(-1).float()
    threshold = torch.quantile(conf, percentile / 100.0)
    return conf >= threshold


def edge_keep_mask(depth: torch.Tensor, rtol: float, kernel_size: int) -> torch.Tensor:
    """Boolean (M,) mask dropping gaussians on depth discontinuities (flying edges).

    Reuses inference_pipeline.depth_edge, which flags pixels whose local depth
    range jumps by more than rtol (relative).
    """
    depth_np = depth[..., 0].float().cpu().numpy()  # (N, H, W)
    edges = depth_edge(depth_np, rtol=rtol, kernel_size=kernel_size)
    return torch.from_numpy(~edges.reshape(-1)).to(depth.device)


def to_uint8_image(image_chw: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] -> (H, W, 3) uint8 RGB."""
    return (image_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def save_view_comparisons(images: torch.Tensor, renders: torch.Tensor, compare_dir: str) -> None:
    os.makedirs(compare_dir, exist_ok=True)
    for i in range(images.shape[0]):
        side_by_side = np.hstack([to_uint8_image(images[i]), to_uint8_image(renders[i])])
        cv2.imwrite(os.path.join(compare_dir, f"{i:03d}.png"), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved {images.shape[0]} input-vs-render comparisons → {compare_dir}")


def save_video(frames: torch.Tensor, video_path: str, fps: int) -> None:
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", pixelformat="yuv420p")
    for i in range(frames.shape[0]):
        writer.append_data(to_uint8_image(frames[i]))
    writer.close()
    print(f"✅ Saved rendered video ({frames.shape[0]} frames @ {fps} fps) → {video_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Directory containing input images")
    parser.add_argument("--pattern", default="*", help="Glob pattern for input images")
    parser.add_argument("--stride", type=int, default=10, help="Take every N-th image")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of input images")
    parser.add_argument("--output_dir", default="gs_test_outputs", help="Output directory")
    parser.add_argument("--sh_degree", type=int, default=2, help="SH degree of the GS head (0-2)")
    parser.add_argument("--steps_per_pair", type=int, default=8, help="Interpolated frames per camera pair")
    parser.add_argument("--fps", type=int, default=24, help="Output video frame rate")
    add_common_args(parser)
    parser.set_defaults(conf_percentile=0.0)  # keep all gaussians unless asked to filter
    args = parser.parse_args()

    image_paths = collect_images(args.image_dir, args.pattern, args.stride, args.max_frames)
    print(f"Found {len(image_paths)} images in {args.image_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint} (enable_gs=True, sh_degree={args.sh_degree})")
    model = load_model(args.checkpoint, device=args.device, enable_gs=True, gs_sh_degree=args.sh_degree)

    images = load_and_preprocess_images(image_paths, image_resolution=args.resolution).to(args.device)
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    print_sanity_stats(predictions, model)

    height, width = images.shape[-2:]
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], (height, width))
    extrinsic, intrinsic = extrinsic[0], intrinsic[0]
    gaussians = predictions["gaussians"]
    print(f"Predicted {gaussians.means.shape[1]} gaussians from {predictions['images'].shape[1]} frames")

    num_total = gaussians.means.shape[1]
    keep = torch.ones(num_total, dtype=torch.bool, device=gaussians.means.device)
    if args.conf_percentile > 0:
        keep &= conf_keep_mask(predictions["depth_conf"][0], args.conf_percentile)
    if args.edge_filter:
        keep &= edge_keep_mask(predictions["depth"][0], args.edge_rtol, args.edge_kernel_size)
    if not bool(keep.all()):
        print(f"Filtered gaussians: kept {int(keep.sum())}/{num_total} "
              f"(conf_percentile={args.conf_percentile:g}, edge_filter={args.edge_filter})")
        gaussians = select_gaussians(gaussians, keep)

    ply_path = os.path.join(args.output_dir, "gaussians.ply")
    export_gaussians_ply(gaussians, ply_path, save_sh_dc_only=(args.sh_degree == 0))
    print(f"✅ Saved gaussians ({gaussians.means.shape[1]} points) → {ply_path}")

    with torch.no_grad():
        renders, _ = render_gaussians(gaussians, extrinsic, intrinsic, height, width)
        save_view_comparisons(predictions["images"][0], renders, os.path.join(args.output_dir, "compare"))

        path_extrinsic, path_intrinsic = interpolate_camera_path(extrinsic, intrinsic, args.steps_per_pair)
        print(f"Rendering interpolated path: {path_extrinsic.shape[0]} frames")
        frames, _ = render_gaussians(gaussians, path_extrinsic, path_intrinsic, height, width)
        save_video(frames, os.path.join(args.output_dir, "render.mp4"), args.fps)


if __name__ == "__main__":
    main()
