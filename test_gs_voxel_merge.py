"""Inference-only sweep of opacity-weighted voxel merging across voxel sizes.

Runs VGGT-Omega with the GSDPT head, then for each requested voxel_size merges
the predicted Gaussians (utils/gs_merge.voxel_merge_gaussians) and reports the
resulting Gaussian count + compression ratio. For each size it also re-renders the
merged Gaussians back to the input views and along an interpolated camera path, so
the count/ratio can be weighed against visual quality.

Layout:
    {output_dir}/
        summary.txt              voxel_size, kept count, original count, ratio
        v{voxel_size}/
            gaussians.ply        merged 3DGS PLY
            compare/{i}.png       input view vs. re-rendered merged view
            render.mp4            interpolated camera-path rendering (merged)

Usage:
    python test_gs_voxel_merge.py --image_dir path/to/images \\
        --voxel_sizes 0.002 0.004 0.008
"""

import argparse
import os

import torch

from inference_pipeline import add_common_args, collect_images, load_model
from test_gs_inference import (
    conf_keep_mask,
    edge_keep_mask,
    save_video,
    save_view_comparisons,
)
from vggt_omega.utils.gs_merge import voxel_merge_gaussians
from vggt_omega.utils.gs_ply import export_gaussians_ply
from vggt_omega.utils.gs_renderer import interpolate_camera_path, render_gaussians
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


DEFAULT_IMAGE_DIR = (
    "/seaweed/training-mri/dataset/open_source/DL3DV-10K/benchmark_140/"
    "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7/nerfstudio/images_4"
)
DEFAULT_CHECKPOINT = "vggt_omega_ckpts/vggt_omega_1b_512.pt"
DEFAULT_GSDPT_CHECKPOINT = "checkpoints/gsdpt_training_stage_3/gsdpt-24000-psnr20.17.pt"


def load_gsdpt_head(model, gsdpt_checkpoint: str) -> None:
    """Load a trained GSDPT head state_dict into ``model.gs_head`` (in place).

    Mirrors training/model_wrapper.GSModel: the checkpoint is either the raw head
    state_dict or a training dict with a ``gs_head`` entry.
    """
    ckpt = torch.load(gsdpt_checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["gs_head"] if isinstance(ckpt, dict) and "gs_head" in ckpt else ckpt
    model.gs_head.load_state_dict(sd)
    print(f"Loaded trained GSDPT head from {gsdpt_checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Directory containing input images")
    parser.add_argument("--pattern", default="*", help="Glob pattern for input images")
    parser.add_argument("--stride", type=int, default=10, help="Take every N-th image")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of input images")
    parser.add_argument("--output_dir", default="gs_voxel_merge_outputs", help="Output directory")
    parser.add_argument(
        "--gsdpt_checkpoint", default=DEFAULT_GSDPT_CHECKPOINT,
        help="Trained GSDPT head checkpoint to load on top of the VGGT model (None/'' to skip)",
    )
    parser.add_argument("--sh_degree", type=int, default=0, help="SH degree of the GS head (0-2)")
    parser.add_argument(
        "--voxel_sizes", type=float, nargs="+", default=[0.001, 0.002, 0.004, 0.008, 0.016],
        help="Voxel sizes to sweep (world units)",
    )
    parser.add_argument("--steps_per_pair", type=int, default=8, help="Interpolated frames per camera pair")
    parser.add_argument("--fps", type=int, default=24, help="Output video frame rate")
    parser.add_argument("--no_render", action="store_true", help="Only report counts; skip render/PLY")
    add_common_args(parser)
    parser.set_defaults(
        conf_percentile=0.0,  # keep all gaussians unless asked to filter
        checkpoint=DEFAULT_CHECKPOINT,
    )
    args = parser.parse_args()

    image_paths = collect_images(args.image_dir, args.pattern, args.stride, args.max_frames)
    print(f"Found {len(image_paths)} images in {args.image_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint} (enable_gs=True, sh_degree={args.sh_degree})")
    model = load_model(args.checkpoint, device=args.device, enable_gs=True, gs_sh_degree=args.sh_degree)
    if args.gsdpt_checkpoint:
        load_gsdpt_head(model, args.gsdpt_checkpoint)

    images = load_and_preprocess_images(image_paths, image_resolution=args.resolution).to(args.device)
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    height, width = images.shape[-2:]
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], (height, width))
    extrinsic, intrinsic = extrinsic[0], intrinsic[0]
    gaussians = predictions["gaussians"]
    num_total = gaussians.means.shape[1]
    print(f"Predicted {num_total} gaussians from {predictions['images'].shape[1]} frames")

    # Optional pre-merge filtering (confidence / depth-edge), mirroring test_gs_inference.
    keep = torch.ones(num_total, dtype=torch.bool, device=gaussians.means.device)
    if args.conf_percentile > 0:
        keep &= conf_keep_mask(predictions["depth_conf"][0], args.conf_percentile)
    if args.edge_filter:
        keep &= edge_keep_mask(predictions["depth"][0], args.edge_rtol, args.edge_kernel_size)
    filter_mask = None if bool(keep.all()) else keep
    base_count = int(keep.sum()) if filter_mask is not None else num_total
    if filter_mask is not None:
        print(f"Pre-merge filter: kept {base_count}/{num_total} gaussians")

    summary_lines = ["voxel_size\tmerged\toriginal\tratio"]
    for vs in args.voxel_sizes:
        with torch.inference_mode():
            merged = voxel_merge_gaussians(gaussians, vs, filter_mask=filter_mask)
        merged_count = merged.means.shape[1]
        ratio = base_count / max(1, merged_count)
        line = f"{vs:.6g}\t{merged_count}\t{base_count}\t{ratio:.3f}x"
        summary_lines.append(line)
        print(f"voxel_size={vs:<8g} merged={merged_count:<8d} original={base_count:<8d} ratio={ratio:.3f}x")

        if args.no_render:
            continue

        vs_dir = os.path.join(args.output_dir, f"v{vs:g}")
        os.makedirs(vs_dir, exist_ok=True)

        ply_path = os.path.join(vs_dir, "gaussians.ply")
        export_gaussians_ply(merged, ply_path, save_sh_dc_only=(args.sh_degree == 0))

        with torch.no_grad():
            renders, _ = render_gaussians(merged, extrinsic, intrinsic, height, width)
            save_view_comparisons(predictions["images"][0], renders, os.path.join(vs_dir, "compare"))

            path_extrinsic, path_intrinsic = interpolate_camera_path(
                extrinsic, intrinsic, args.steps_per_pair
            )
            frames, _ = render_gaussians(merged, path_extrinsic, path_intrinsic, height, width)
            save_video(frames, os.path.join(vs_dir, "render.mp4"), args.fps)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"✅ Saved compression summary → {summary_path}")


if __name__ == "__main__":
    main()
