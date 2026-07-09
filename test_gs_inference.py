"""Test pipeline for the VGGT-Omega 3DGS branch: gaussian PLY + rendered video.

Runs VGGT-Omega with the GSDPT head enabled, then writes:

    {output_dir}/
        gaussians.ply       standard 3DGS PLY (opacity logit, log scales, WXYZ rot)
        depth/{stem}.npy + {stem}.png   raw + colorized depth per frame
        depth_conf.npy, cameras.npz
        pcd/pointcloud.ply  RGB point cloud (binary PLY)
        compare/{i}.png     input view vs. re-rendered view, side by side (unless --no_compare)
        vggt-o-{scene}-render.mp4   interpolated camera-path rendering (unless --no_render_video)

By default the head is untrained (released checkpoints carry no GS weights, so
the head keeps its initialization — Gaussians at the unprojected depth points,
color = pixel RGB, opacity 0.12, ~0.25 px footprint). Pass --gsdpt_checkpoint to
load a trained GSDPT head (a training checkpoint produced by train_gs.py).

Usage:
    python test_gs_inference.py --image_dir path/to/images
    python test_gs_inference.py --image_dir path/to/images \
        --gsdpt_checkpoint checkpoints/gsdpt_training_stage_1/gsdpt-24500-psnr18.01.pt
"""

import argparse
import os

import cv2
import imageio
import numpy as np
import torch

from inference_pipeline import (
    add_common_args,
    build_point_cloud,
    collect_images,
    depth_edge,
    load_model,
    save_depth_outputs,
    save_point_cloud_ply,
)
from vggt_omega.models.gs_adapter import Gaussians
from vggt_omega.utils.gs_ply import export_gaussians_ply
from vggt_omega.utils.gs_renderer import interpolate_camera_path, render_gaussians
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

scene = "scene"
DEFAULT_IMAGE_DIR = f"/root/to/images/{scene}"

def load_gsdpt_checkpoint(model, checkpoint_path: str, device: str) -> int:
    """Load a trained GSDPT head into ``model.gs_head`` in place.

    Accepts either a training checkpoint dict (as saved by train_gs.py, with the
    head ``state_dict`` under the ``"gs_head"`` key) or a bare head ``state_dict``.
    Returns the training step if present in the checkpoint, else -1.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "gs_head" in ckpt:
        state_dict = ckpt["gs_head"]
        step = ckpt.get("step", -1)
    else:
        state_dict = ckpt
        step = -1

    # The training checkpoint stores its full config; if its SH degree differs
    # from the model we just built, the channel layout (and final conv shape)
    # won't match — fail early with an actionable message instead of a cryptic
    # size-mismatch error from load_state_dict.
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    ckpt_sh = getattr(getattr(ckpt_cfg, "model", None), "gs_sh_degree", None)
    if ckpt_sh is not None and ckpt_sh != model.gs_head.sh_degree:
        raise ValueError(
            f"Checkpoint was trained with sh_degree={ckpt_sh} but the model was built "
            f"with sh_degree={model.gs_head.sh_degree}. Re-run with --sh_degree {ckpt_sh}."
        )

    # Training may save a torch.compile-wrapped head (keys prefixed with
    # ``_orig_mod.``); strip the prefix so it loads onto a plain GSDPTHead.
    state_dict = {k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}

    missing, unexpected = model.gs_head.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "GSDPT checkpoint does not match the model head "
            f"(missing={list(missing)}, unexpected={list(unexpected)}). "
            "Check that --sh_degree matches the trained head."
        )
    return step


def print_sanity_stats(predictions: dict, model, trained: bool = False) -> None:
    """Report the raw head outputs.

    With an untrained head these equal the init biases; with a trained head they
    are printed for reference only (no longer expected to match the init values).
    """
    layout = model.gs_head.channel_layout
    raw_gs = predictions["raw_gs"][0]
    opacity = predictions["gs_opacity"][0]

    scale_raw = raw_gs[..., layout["scales"]]
    smin, smax = model.gs_adapter.gaussian_scale_min, model.gs_adapter.gaussian_scale_max
    scale_act = smin + (smax - smin) * scale_raw.sigmoid()

    target_scale = model.gs_head.init_pixel_size / (2.0 * model.gs_adapter.scale_multiplier)
    header = "GS head sanity stats" + (
        " (trained head; init values shown for reference):"
        if trained
        else " (untrained head should match the init values):"
    )
    print(header)
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


def add_gs_args(parser: argparse.ArgumentParser) -> None:
    """GS / render / video CLI args shared by the image-dir and video scripts."""
    parser.add_argument("--sh_degree", type=int, default=0, help="SH degree of the GS head (0-2)")
    parser.add_argument(
        "--gsdpt_checkpoint",
        default="/root/vggt_omega_gsdpt_checkpoints/gsdpt-vggt-omega.pt",
        help="Path to a trained GSDPT checkpoint (from train_gs.py) to load onto the GS head",
    )
    parser.add_argument("--steps_per_pair", type=int, default=8, help="Interpolated frames per camera pair")
    parser.add_argument("--render_fps", type=int, default=24, help="Output render video frame rate")
    parser.add_argument("--render_chunk_size", type=int, default=8, help="Views rendered per gsplat forward pass")
    parser.add_argument("--render_mode", default="RGB+ED", help="gsplat render mode (e.g. RGB, RGB+D)")
    parser.add_argument("--scene", default=None, help="Scene name for the render video (default: input dir/video stem)")
    parser.add_argument("--no_compare", action="store_true", help="Skip the input-vs-render comparison images")
    parser.add_argument("--no_render_video", action="store_true", help="Skip the interpolated render video")


def run_gs_pipeline(args, image_paths: list, output_dir: str, scene_name: str, ply_suffix: str = "") -> None:
    """Full GS reconstruction flow shared by the image-dir and video entry points.

    Runs one GS-enabled forward pass and writes depth maps, a point cloud, a 3DGS
    PLY, optional input-vs-render comparisons, and an optional render video. The
    confidence/edge filters (--conf_percentile, --edge_*) apply to both the
    gaussians and the point cloud; --max_points/--seed downsample the point cloud.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint} (enable_gs=True, sh_degree={args.sh_degree})")
    model = load_model(args.checkpoint, device=args.device, enable_gs=True, gs_sh_degree=args.sh_degree)

    trained = False
    if args.gsdpt_checkpoint is not None:
        step = load_gsdpt_checkpoint(model, args.gsdpt_checkpoint, args.device)
        step_str = f" (step {step})" if step >= 0 else ""
        print(f"✅ Loaded trained GSDPT head from {args.gsdpt_checkpoint}{step_str}")
        trained = True

    images = load_and_preprocess_images(image_paths, image_resolution=args.resolution).to(args.device)
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    print_sanity_stats(predictions, model, trained=trained)

    height, width = images.shape[-2:]
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], (height, width))
    extrinsic, intrinsic = extrinsic[0], intrinsic[0]
    gaussians = predictions["gaussians"]
    print(f"Predicted {gaussians.means.shape[1]} gaussians from {predictions['images'].shape[1]} frames")

    # Depth maps + point cloud reuse the numpy helpers from inference_pipeline.
    # Assemble a preds_np dict shaped exactly like run_inference's output (batch
    # dim squeezed) — run_inference itself can't be reused as it drops Gaussians.
    def to_np(t: torch.Tensor) -> np.ndarray:
        return t[0].detach().float().cpu().numpy()

    preds_np = {
        "depth": to_np(predictions["depth"]),  # (N, H, W, 1)
        "depth_conf": to_np(predictions["depth_conf"]),  # (N, H, W)
        "images": to_np(predictions["images"]),  # (N, 3, H, W)
        "extrinsic": extrinsic.detach().float().cpu().numpy(),  # (N, 3, 4)
        "intrinsic": intrinsic.detach().float().cpu().numpy(),  # (N, 3, 3)
    }
    save_depth_outputs(preds_np, image_paths, output_dir)
    points, colors = build_point_cloud(
        preds_np,
        conf_percentile=args.conf_percentile,
        edge_filter=args.edge_filter,
        edge_rtol=args.edge_rtol,
        edge_kernel_size=args.edge_kernel_size,
        max_points=args.max_points,
        seed=args.seed,
    )
    save_point_cloud_ply(points, colors, os.path.join(output_dir, "pcd", f"pointcloud{ply_suffix}.ply"))

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

    ply_path = os.path.join(output_dir, "gaussians.ply")
    export_gaussians_ply(gaussians, ply_path, save_sh_dc_only=(args.sh_degree == 0))
    print(f"✅ Saved gaussians ({gaussians.means.shape[1]} points) → {ply_path}")

    if args.no_compare and args.no_render_video:
        return

    with torch.no_grad():
        if not args.no_compare:
            renders, _ = render_gaussians(
                gaussians, extrinsic, intrinsic, height, width,
                chunk_size=args.render_chunk_size, render_mode=args.render_mode,
            )
            save_view_comparisons(predictions["images"][0], renders, os.path.join(output_dir, "compare"))

        if not args.no_render_video:
            path_extrinsic, path_intrinsic = interpolate_camera_path(extrinsic, intrinsic, args.steps_per_pair)
            print(f"Rendering interpolated path: {path_extrinsic.shape[0]} frames")
            frames, _ = render_gaussians(
                gaussians, path_extrinsic, path_intrinsic, height, width,
                chunk_size=args.render_chunk_size, render_mode=args.render_mode,
            )
            video_path = os.path.join(output_dir, f"vggt-o-{scene_name}-render.mp4")
            save_video(frames, video_path, args.render_fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Directory containing input images")
    parser.add_argument("--pattern", default="*", help="Glob pattern for input images")
    parser.add_argument("--stride", type=int, default=1, help="Take every N-th image")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of input images")
    parser.add_argument("--output_dir", default="gs_test_outputs", help="Output directory")
    add_gs_args(parser)
    add_common_args(parser)
    parser.set_defaults(conf_percentile=0.0)  # keep all gaussians/points unless asked to filter
    args = parser.parse_args()

    image_paths = collect_images(args.image_dir, args.pattern, args.stride, args.max_frames)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    scene_name = args.scene or os.path.basename(os.path.normpath(args.image_dir))
    run_gs_pipeline(args, image_paths, args.output_dir, scene_name=scene_name)


if __name__ == "__main__":
    main()
