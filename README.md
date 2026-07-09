<div align="center">
<h1>Feed-Forward 3DGS with VGGT-&Omega;</h1>

<a href="https://huggingface.co/vc12301/VGGT-Omega-GSDPT/tree/main"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GSDPT_Checkpoint-blue'></a>

**English** | [中文](./README_zh.md)
</div>

## Overview

This project builds a **feed-forward 3D Gaussian Splatting (3DGS)** framework on top of the
[VGGT-Omega](http://vggt-omega.github.io/) reconstruction model. Given a set of images or a video,
a trained **GSDPT head** predicts per-pixel 3D Gaussians in a single forward pass — no per-scene
optimization. From those Gaussians the pipeline exports a standard 3DGS `.ply`, renders novel views
along an interpolated camera path, and also produces the underlying depth maps, cameras, and point
cloud.

The framework consists of two components loaded together at runtime:

- **VGGT-Omega backbone** — feed-forward camera + depth reconstruction (frozen during 3DGS training).
- **GSDPT head** — the feed-forward 3DGS head trained in this repo, which turns backbone features
  into per-pixel Gaussians.

Both **inference** (`gs_inference_pipeline_video.py`) and **training** (`train_gs.py`) are supported.

## Pretrained Models

You need two checkpoints — the VGGT-Omega backbone and the trained GSDPT head.

| Component | Model | Resolution | Download |
| :--- | :--- | :--- | :--- |
| GSDPT head | `gsdpt-vggt-omega.pt` | 512 | [🤗 vc12301/VGGT-Omega-GSDPT](https://huggingface.co/vc12301/VGGT-Omega-GSDPT/tree/main) |
| Backbone | `vggt_omega_1b_512.pt` | 512 | [🤗 facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt) |

> The backbone lives in the official VGGT-Omega Hugging Face repo and requires access approval.
> Requests are reviewed by an automated process based on the information provided.

## Installation

```bash
conda create -n vggt-omega python=3.10
conda activate vggt-omega
cd /path/to/Feedforward-3DGS-with-VGGT-Omega

# Choose the torch build that matches your CUDA version:
# https://pytorch.org/get-started/previous-versions/
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt        # core model + inference
pip install -r requirements_gs.txt     # 3DGS rendering/export (gsplat, e3nn, imageio)
pip install -r requirements_train.txt  # training extras (lpips, wandb, pyyaml, tqdm)
pip install -e .

# System libs needed by OpenCV / rendering
apt-get install -y libgl1-mesa-glx libglib2.0-0
```

For inference only, `requirements.txt` + `requirements_gs.txt` are sufficient;
`requirements_train.txt` is only needed for training.

## Inference

The main entry point is `gs_inference_pipeline_video.py`, which samples frames from a video, runs
one feed-forward pass, and reconstructs the scene. Point it at your downloaded backbone and GSDPT
checkpoints:

```bash
python gs_inference_pipeline_video.py \
    --checkpoint   /PATH/TO/vggt_omega_1b_512.pt \
    --gsdpt_checkpoint /PATH/TO/gsdpt-vggt-omega.pt \
    --video examples/school.mp4 \
    --fps 1.0 \
    --max_frames 45
```

### Outputs

Results are written to `gs_inference_pipeline_video_outputs/{video_stem}/` by default
(override with `--output_dir`):

| Path | Description |
| :--- | :--- |
| `gaussians.ply` | Standard 3DGS point cloud — open in any 3DGS viewer |
| `vggt-o-{scene}-render.mp4` | Novel-view rendering along an interpolated camera path |
| `compare/{i:03d}.png` | Side-by-side input view vs. re-rendered view |
| `pcd/pointcloud_{stem}.ply` | RGB point cloud (depth unprojection) |
| `depth/{stem}.npy` + `.png` | Raw float32 depth and colorized depth per frame |
| `depth_conf.npy`, `cameras.npz` | Depth confidence and predicted `extrinsic`/`intrinsic` |
| `input_images/{i:06d}.png` | Extracted input frames |

### Key options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--video` | `examples/school.mp4` | Input video file |
| `--fps` | `1.0` | Frame sampling rate |
| `--max_frames` | `45` | Cap on frames used for inference (more frames need more GPU memory) |
| `--checkpoint` | — | VGGT-Omega backbone checkpoint |
| `--gsdpt_checkpoint` | — | Trained GSDPT head checkpoint |
| `--sh_degree` | `0` | SH degree of the GS head — must match the trained head |
| `--resolution` | `512` | Inference image resolution |
| `--no_render_video` | off | Skip the interpolated render video |
| `--no_compare` | off | Skip the input-vs-render comparison images |
| `--conf_percentile` | `0.0` | Drop low-confidence points/Gaussians below this percentile |
| `--edge_filter` | off | Remove flying points on depth discontinuities |

> The `--checkpoint` and `--gsdpt_checkpoint` defaults are hardcoded to the authors' local paths.
> Pass your own paths on the command line (as above), or edit the defaults in
> `test_gs_inference.py` / `inference_pipeline.py`.

### Reconstruct from an image directory

To run on a folder of images instead of a video, use `test_gs_inference.py` (same GSDPT/backbone
loading, `--image_dir` instead of `--video`):

```bash
python test_gs_inference.py \
    --checkpoint   /PATH/TO/vggt_omega_1b_512.pt \
    --gsdpt_checkpoint /PATH/TO/gsdpt-vggt-omega.pt \
    --image_dir /PATH/TO/images
```

## Training

Training optimizes **only the GSDPT head**; the VGGT-Omega backbone, camera head, and depth head
stay frozen. Since VGGT-Omega is pose-free, ground-truth cameras are used only for view sampling —
the model's own predicted cameras and depth drive Gaussian unprojection and novel-view rendering.

### Datasets

Training draws from up to three datasets, selected via `dataset_mode` / `mix_datasets` in the config:

- **DL3DV** — real multi-view scenes
- **rendering** — synthetic renders (closed-sourced)
- **ScanNet++** — real indoor scenes

Each dataset root and its layout keys (`data_root`, `val_root`, `image_dir_name`, etc.) are set in
the config. All shipped configs point at the authors' internal absolute paths, so **you must repoint
every dataset root you use**.

### Commands

Single-GPU:

```bash
python train_gs.py --config configs/gsdpt_training_stage_3.yaml
```

Single-node multi-GPU (8 GPUs, DDP via torchrun):

```bash
torchrun --nproc_per_node=8 train_gs.py \
    --config configs/gsdpt_training_stage_3.yaml
```

CLI overrides use dot notation on top of the YAML, e.g.
`optimizer.lr=5e-5 training.max_iterations=50000`. (List-valued keys such as `mix_ratio` can only be
set in YAML.)

### Stages

The configs form a progressive curriculum, chained through checkpoint paths — reproduce them in
order, editing the resume path at each step:

| Config | Data | Notes |
| :--- | :--- | :--- |
| `gsdpt_training_stage_1.yaml` | DL3DV | Base training of the GS head, core losses |
| `gsdpt_training_stage_2.yaml` | DL3DV + rendering | Refinement with synthetic renders, resumes stage 1 |
| `gsdpt_training_stage_3.yaml` | DL3DV + rendering + ScanNet++ | Longest mixed run, full feature set |

### Before you train — edit the config

In your chosen `configs/gsdpt_training_stage_*.yaml`, update at least:

- `model.checkpoint_path` — path to the VGGT-Omega backbone (`vggt_omega_1b_512.pt`)
- `model.resume_gsdpt_checkpoint` / `model.resume_full_checkpoint` — set to `null` for a fresh run,
  or point at a prior stage's checkpoint to continue
- `data.data_root` / `data.val_root`, `rendering_data.data_root`, `scannetpp_data.data_root` —
  your dataset roots (only the ones your `dataset_mode` uses)
- `checkpoint.save_dir` — where checkpoints are written
- `wandb.offline: true` — set this to disable Weights & Biases logging

> **Notes**
> - `training.batch_size` must be `1`. Scale the effective batch with
>   `gradient_accumulation_steps` and/or more DDP ranks.
> - Resolutions in `view_sampling.resolution_schedules` must be multiples of 16 (the patch size).
> - `train_gs.py` hardcodes a WandB base URL and a SOCKS5 proxy in its environment (top of the
>   file). On a normal machine these will break outbound networking — **remove or override those
>   lines**, or set `wandb.offline: true`.
> - A CUDA GPU is required.

## License

See the [LICENSE](./LICENSE) file for details about the license under which this code is made
available.

## Citation

This project builds on VGGT-Omega:

```bibtex
@misc{wang2026vggtomega,
      title={VGGT-$\Omega$},
      author={Jianyuan Wang and Minghao Chen and Shangzhan Zhang and Nikita Karaev and Johannes Schönberger and Patrick Labatut and Piotr Bojanowski and David Novotny and Andrea Vedaldi and Christian Rupprecht},
      year={2026},
      eprint={2605.15195},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.15195},
}
```
