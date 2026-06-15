# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VGGT-Omega: feed-forward camera and depth reconstruction from multi-view images or video (Meta/Oxford VGG research release). Given N frames, the model predicts per-frame depth maps, depth confidence, and camera poses in a single forward pass. An optional, untrained 3DGS branch (`enable_gs=True`) additionally predicts per-pixel Gaussian parameters.

## Setup and commands

```bash
pip install -r requirements.txt && pip install -e .   # core install (numpy<2 required)
pip install -r requirements_demo.txt                  # extras for demo_gradio.py / visual_util.py only

# Quick smoke test of model loading + forward pass
python test_inference.py

# Headless inference: point cloud PLY + depth maps from an image directory
python inference_pipeline.py --image_dir path/to/images

# Same, from a video (extracts frames at --fps first)
python inference_pipeline_video.py --video examples/snow_lift.mp4 --fps 1.0

# 3DGS branch test: gaussian PLY + rendered comparisons/video (needs requirements_gs.txt)
python test_gs_inference.py --image_dir path/to/images

# Interactive Gradio demo (GLB visualization)
python demo_gradio.py --checkpoint ckpts/VGGT-Omega/vggt_omega_1b_512.pt --image-resolution 512
```

There is no test suite or linter configured. Local checkpoints live in `ckpts/VGGT-Omega/`: `vggt_omega_1b_512.pt` (default, 512 res) and `vggt_omega_1b_256_text.pt` (256 res, text alignment).

## Architecture

The model (`vggt_omega/models/vggt_omega.py`) is `VGGTOmega = Aggregator + heads`:

- **Aggregator** (`vggt_omega/models/aggregator.py`): alternating-attention transformer over all frames jointly. Tokens per frame = 1 camera token + 16 register tokens + patch tokens (patch_size 16). Applies ResNet-style normalization internally — inputs are plain [0,1] RGB.
- **CameraHead** → `pose_enc` (B,N,9): translation [3] + quaternion XYZW [4] + vertical/horizontal FoV [2].
- **DenseHead** → `depth` (B,N,H,W,1) and `depth_conf` (B,N,H,W). **`depth_conf` is unbounded positive (≈1 to ~20+), not [0,1]** — filter by percentile (see `build_point_cloud` in inference_pipeline.py or `predictions_to_glb` in visual_util.py), never by an absolute threshold like 0.9.
- **TextAlignmentHead**: only present with `VGGTOmega(enable_alignment=True)`; required to strict-load the `_text` checkpoint. `inference_pipeline.load_model` auto-detects this from state_dict keys.
- **GSDPTHead + GaussianAdapter** (`heads/gsdpt_head.py`, `models/gs_adapter.py`): only with `VGGTOmega(enable_gs=True, gs_sh_degree=0..2)`; requires camera+depth heads. The head mirrors DenseHead's DPT trunk but outputs at full resolution with an `images_merger` RGB injection (DA3-style, no pixel_shuffle). Raw channels: `[xy_offset(2), scales(3), quat XYZW(4), residual SH(3·(deg+1)²), depth_offset(1)] + opacity`; only opacity is activated in the head (sigmoid), everything else in the adapter. SH is **residual**: the predicted DC is added to `(rgb−0.5)/0.2820948`. Predicted scale is scene-scale (depth/FoV-independent); the adapter multiplies by `depth · 0.1·(1/fx+1/fy)` so the screen projection matches. Final-conv init (near-zero weights — a tiny Gaussian perturbation with std `init_weight_std`=1e-4 to break per-pixel symmetry for training — plus biases in logit space) makes the untrained head render a plausible scene: ≈zero offsets, opacity ≈0.12, identity quaternion, ~0.25 px footprint (configurable via `init_opacity` / `init_pixel_size` / `init_weight_std`). `Gaussians` outputs are world-space with **WXYZ** quaternions (gsplat/3DGS convention; everything else in the repo is XYZW). Rendering (`utils/gs_renderer.py`, gsplat) and PLY export (`utils/gs_ply.py`, opacity logit / log scales) lazily import their extras; `e3nn` is needed only for `sh_degree > 0`. Released checkpoints have no GS weights — `load_model(..., enable_gs=True)` loads with `strict=False` and keeps the initialization.

Standard inference flow (see `test_inference.py` or `inference_pipeline.run_inference`):

1. `load_and_preprocess_images(paths, image_resolution=512)` (`vggt_omega/utils/load_fn.py`) → (N,3,H,W) in [0,1]. `mode="balanced"` (default) targets a token budget ≈ resolution²; `mode="max_size"` caps the longest side. Aspect ratios are clamped to [0.5, 2.0] by center-cropping; mixed-size batches are padded.
2. `model(images)` under `torch.inference_mode()`. **Do not wrap in an external autocast** — `forward` manages it internally (bf16/fp16 for the aggregator, fp32 for heads).
3. `encoding_to_camera(pose_enc, images.shape[-2:])` (`vggt_omega/utils/pose_enc.py`) → extrinsic (B,N,3,4) camera-from-world (OpenCV convention) + intrinsic (B,N,3,3).
4. `unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)` (`vggt_omega/utils/geometry.py`) → (N,H,W,3) world points.

## Import boundaries

- `demo_gradio.py` imports `gradio` and `visual_util.py` imports trimesh/matplotlib/scipy/onnxruntime (demo extras). Headless scripts must not import from either — shared logic belongs in `vggt_omega/utils/` (e.g., `unproject_depth_map_to_point_map` lives in `geometry.py` for this reason) or in `inference_pipeline.py`, which `inference_pipeline_video.py` reuses.
- The core package (`vggt_omega/`) depends only on torch/numpy/PIL/einops/cv2.
