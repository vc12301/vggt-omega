"""Inference pipeline for VGGT-Omega: point cloud (PLY) + depth map outputs.

Takes a directory of images, runs VGGT-Omega to predict per-frame depth and
camera poses, unprojects depth to a world-space point cloud, and writes:

    {output_dir}/
        depth/{stem}.npy        raw float32 depth per frame
        depth/{stem}.png        colorized depth visualization
        pcd/pointcloud.ply      RGB point cloud (binary PLY)
        cameras.npz             extrinsic (N,3,4) + intrinsic (N,3,3)
        depth_conf.npy          per-frame confidence maps

Usage:
    python inference_pipeline.py --image_dir path/to/images
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.geometry import unproject_depth_map_to_point_map
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

DEFAULT_IMAGE_DIR = "demo_outputs/input_images_20260610_214602_134139/images"
DEFAULT_IMAGE_DIR = "/data-nas/experiments/yemu/workspace/HY-World-2.0/inference_output/_frames/bedroom_w_exterior"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    """Mark pixels whose local depth range jumps by more than rtol (relative)."""
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    enable_gs: bool = False,
    gs_sh_degree: int = 0,
) -> VGGTOmega:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    enable_alignment = any(k.startswith("text_alignment_head.") for k in state_dict)
    model = VGGTOmega(enable_alignment=enable_alignment, enable_gs=enable_gs, gs_sh_degree=gs_sh_degree).eval()

    if enable_gs:
        # Released checkpoints have no GS weights yet; keep the gs_head initialization.
        result = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in result.missing_keys if not k.startswith(("gs_head.", "gs_adapter."))]
        if unexpected or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch beyond GS modules: missing={unexpected}, unexpected={result.unexpected_keys}"
            )
        if result.missing_keys:
            print(f"GS head not in checkpoint; using initialized weights ({len(result.missing_keys)} tensors)")
    else:
        model.load_state_dict(state_dict)
    return model.to(device)


def run_inference(model: VGGTOmega, image_paths: list, resolution: int, device: str = "cuda") -> dict:
    images = load_and_preprocess_images(image_paths, image_resolution=resolution).to(device)
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    torch.cuda.empty_cache()
    return predictions_np


def build_point_cloud(
    preds: dict,
    conf_percentile: float = 20.0,
    edge_filter: bool = False,
    edge_rtol: float = 0.03,
    edge_kernel_size: int = 3,
    max_points: int = 1_000_000,
    seed: int = 42,
) -> tuple:
    depth = preds["depth"]  # (N, H, W, 1)
    conf = preds["depth_conf"].copy()  # (N, H, W)

    world_points = unproject_depth_map_to_point_map(depth, preds["extrinsic"], preds["intrinsic"])

    if edge_filter:
        edges = depth_edge(depth[..., 0], rtol=edge_rtol, kernel_size=edge_kernel_size)
        conf[edges] = 0.0
        print(f"Edge filter: marked {edges.sum()} pixels as depth edges")

    points = world_points.reshape(-1, 3)
    colors = (np.transpose(preds["images"], (0, 2, 3, 1)).reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)
    depth_flat = depth[..., 0].reshape(-1)

    mask = np.isfinite(points).all(axis=-1) & np.isfinite(conf_flat) & (depth_flat > 0)
    if conf_percentile > 0 and mask.any():
        threshold = np.percentile(conf_flat[mask], conf_percentile)
        mask &= conf_flat >= threshold
    mask &= conf_flat > 1e-5
    print(f"Confidence filter: kept {mask.sum()}/{mask.size} points")

    points, colors = points[mask], colors[mask]

    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points, colors = points[indices], colors[indices]
        print(f"Downsampled to {max_points} points")

    return points, colors


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, ply_path: str) -> None:
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex = np.empty(len(points), dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(ply_path)
    print(f"✅ Saved point cloud ({len(points)} points) → {ply_path}")


def colorize_depth(depth_hw: np.ndarray) -> np.ndarray:
    """Colorize a (H, W) depth map; near is warm, far is cold. Returns BGR uint8."""
    valid = np.isfinite(depth_hw) & (depth_hw > 0)
    if valid.any():
        lo, hi = np.percentile(depth_hw[valid], [2, 98])
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((depth_hw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    norm_u8 = (255 - norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def save_depth_outputs(preds: dict, image_paths: list, output_dir: str) -> None:
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    depth = preds["depth"][..., 0]  # (N, H, W)
    for i in range(depth.shape[0]):
        stem = os.path.splitext(os.path.basename(image_paths[i]))[0] if i < len(image_paths) else f"{i:06d}"
        np.save(os.path.join(depth_dir, f"{stem}.npy"), depth[i].astype(np.float32))
        cv2.imwrite(os.path.join(depth_dir, f"{stem}.png"), colorize_depth(depth[i]))
    print(f"✅ Saved {depth.shape[0]} depth maps (.npy + .png) → {depth_dir}")

    np.save(os.path.join(output_dir, "depth_conf.npy"), preds["depth_conf"].astype(np.float32))
    np.savez(
        os.path.join(output_dir, "cameras.npz"),
        extrinsic=preds["extrinsic"].astype(np.float32),
        intrinsic=preds["intrinsic"].astype(np.float32),
    )
    print(f"✅ Saved cameras.npz and depth_conf.npy → {output_dir}")


def run_pipeline(args, image_paths: list, output_dir: str, ply_suffix: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device=args.device)

    print(f"Running inference on {len(image_paths)} images (resolution={args.resolution})")
    preds = run_inference(model, image_paths, args.resolution, device=args.device)

    save_depth_outputs(preds, image_paths, output_dir)

    points, colors = build_point_cloud(
        preds,
        conf_percentile=args.conf_percentile,
        edge_filter=args.edge_filter,
        edge_rtol=args.edge_rtol,
        edge_kernel_size=args.edge_kernel_size,
        max_points=args.max_points,
        seed=args.seed,
    )
    ply_path = os.path.join(output_dir, "pcd", f"pointcloud{ply_suffix}.ply")
    save_point_cloud_ply(points, colors, ply_path)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default="ckpts/VGGT-Omega/vggt_omega_1b_512.pt", help="Model checkpoint path")
    parser.add_argument("--resolution", type=int, default=512, help="Inference image resolution")
    parser.add_argument("--conf_percentile", type=float, default=20.0,
                        help="Drop points below this confidence percentile (0 disables)")
    parser.add_argument("--edge_filter", action="store_true",
                        help="Filter out points on depth discontinuities (flying points)")
    parser.add_argument("--edge_rtol", type=float, default=0.03, help="Relative depth jump threshold for edge filter")
    parser.add_argument("--edge_kernel_size", type=int, default=3, help="Neighborhood size for edge filter")
    parser.add_argument("--max_points", type=int, default=0,
                        help="Randomly downsample point cloud to at most this many points (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for point downsampling")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")


def collect_images(image_dir: str, pattern: str, stride: int, max_frames) -> list:
    paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if pattern == "*":
        paths = [p for p in paths if p.lower().endswith(IMAGE_EXTENSIONS)]
    paths = paths[::stride]
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"No images found in {image_dir} with pattern {pattern!r}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Directory containing input images")
    parser.add_argument("--pattern", default="*", help="Glob pattern for input images")
    parser.add_argument("--stride", type=int, default=1, help="Take every N-th image")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of input images")
    parser.add_argument("--output_dir", default="inference_pipeline_outputs", help="Output directory")
    add_common_args(parser)
    args = parser.parse_args()

    image_paths = collect_images(args.image_dir, args.pattern, args.stride, args.max_frames)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    run_pipeline(args, image_paths, args.output_dir)


if __name__ == "__main__":
    main()
