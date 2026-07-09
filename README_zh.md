<div align="center">
<h1>基于 VGGT-&Omega; 的前馈 3DGS</h1>

<a href="https://huggingface.co/vc12301/VGGT-Omega-GSDPT/tree/main"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GSDPT_Checkpoint-blue'></a>

[English](./README.md) | **中文**
</div>

## 项目简介

本项目在三维重建模型 [VGGT-Omega](http://vggt-omega.github.io/) 之上，搭建了一套**前馈式3D 高斯泼溅（3DGS）**框架。给定一组图像或一段视频，训练好的 **GSDPT 预测头**在单次前向推理中直接预测逐像素的 3D 高斯，无需逐场景优化。据此，推理流程会导出标准 3DGS `.ply` 文件、沿插值相机路径渲染新视角，并同时输出对应的深度图、相机参数与点云。

框架在运行时由两部分组合加载：

- **VGGT-Omega 主干网络** —— 前馈式相机与深度重建（3DGS 训练时冻结）。
- **GSDPT 预测头** —— 本仓库训练的前馈 3DGS 预测头，将主干特征转换为逐像素高斯。

同时支持**推理**（`gs_inference_pipeline_video.py`）与**训练**（`train_gs.py`）。

## 预训练模型

你需要两个 checkpoint —— VGGT-Omega 主干网络与训练好的 GSDPT 预测头。

| 组件 | 模型 | 分辨率 | 下载 |
| :--- | :--- | :--- | :--- |
| GSDPT 预测头 | `gsdpt-vggt-omega.pt` | 512 | [🤗 vc12301/VGGT-Omega-GSDPT](https://huggingface.co/vc12301/VGGT-Omega-GSDPT/tree/main) |
| 主干网络 | `vggt_omega_1b_512.pt` | 512 | [🤗 facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt) |

> 主干网络位于 VGGT-Omega 官方 Hugging Face 仓库，需要申请访问权限。申请由自动化流程根据你填写的信息进行审核。

## 安装

```bash
conda create -n vggt-omega python=3.10
conda activate vggt-omega
cd /path/to/Feedforward-3DGS-with-VGGT-Omega

# 根据你的 CUDA 版本选择合适的 torch 版本：
# https://pytorch.org/get-started/previous-versions/
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt        # 核心模型 + 推理
pip install -r requirements_gs.txt     # 3DGS 渲染/导出（gsplat、e3nn、imageio）
pip install -r requirements_train.txt  # 训练额外依赖（lpips、wandb、pyyaml、tqdm）
pip install -e .

# OpenCV / 渲染所需的系统库
apt-get install -y libgl1-mesa-glx libglib2.0-0
```

若仅做推理，安装 `requirements.txt` + `requirements_gs.txt` 即可；`requirements_train.txt`只在训练时需要。

## 推理

主入口为 `gs_inference_pipeline_video.py`：它从视频中采样帧，执行一次前馈推理并重建场景。将其指向你下载的主干与 GSDPT checkpoint：

```bash
python gs_inference_pipeline_video.py \
    --checkpoint   /PATH/TO/vggt_omega_1b_512.pt \
    --gsdpt_checkpoint /PATH/TO/gsdpt-vggt-omega.pt \
    --video examples/school.mp4 \
    --fps 1.0 \
    --max_frames 45
```

### 输出结果

默认写入 `gs_inference_pipeline_video_outputs/{video_stem}/`（可用 `--output_dir` 覆盖）：

| 路径 | 说明 |
| :--- | :--- |
| `gaussians.ply` | 标准 3DGS 点云 —— 可用任意 3DGS 查看器打开 |
| `vggt-o-{scene}-render.mp4` | 沿插值相机路径的新视角渲染视频 |
| `compare/{i:03d}.png` | 输入视角与重渲染视角的并排对比 |
| `pcd/pointcloud_{stem}.ply` | RGB 点云（深度反投影） |
| `depth/{stem}.npy` + `.png` | 每帧的原始 float32 深度与着色深度 |
| `depth_conf.npy`、`cameras.npz` | 深度置信度与预测的 `extrinsic`/`intrinsic` |
| `input_images/{i:06d}.png` | 抽取出的输入帧 |

### 主要参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--video` | `examples/school.mp4` | 输入视频文件 |
| `--fps` | `1.0` | 帧采样率 |
| `--max_frames` | `45` | 推理使用的帧数上限（帧数越多显存占用越大） |
| `--checkpoint` | — | VGGT-Omega 主干 checkpoint |
| `--gsdpt_checkpoint` | — | 训练好的 GSDPT 预测头 checkpoint |
| `--sh_degree` | `0` | GS 头的 SH 阶数 —— 必须与训练所用的头一致 |
| `--resolution` | `512` | 推理图像分辨率 |
| `--no_render_video` | 关 | 跳过插值渲染视频 |
| `--no_compare` | 关 | 跳过输入-渲染对比图 |
| `--conf_percentile` | `0.0` | 按置信度百分位丢弃低置信度的点/高斯 |
| `--edge_filter` | 关 | 剔除深度不连续处的“飞点” |

> `--checkpoint` 与 `--gsdpt_checkpoint` 的默认值是作者本地的硬编码路径。请在命令行中传入你自己的路径（如上所示），或修改 `test_gs_inference.py` / `inference_pipeline.py` 中的默认值。

### 从图像目录重建

若要在一个图像文件夹（而非视频）上运行，请使用 `test_gs_inference.py`（GSDPT/主干的加载方式相同，用 `--image_dir` 代替 `--video`）：

```bash
python test_gs_inference.py \
    --checkpoint   /PATH/TO/vggt_omega_1b_512.pt \
    --gsdpt_checkpoint /PATH/TO/gsdpt-vggt-omega.pt \
    --image_dir /PATH/TO/images
```

## 训练

训练**只优化 GSDPT 预测头**；VGGT-Omega 主干、相机头与深度头保持冻结。由于 VGGT-Omega 是无位姿（pose-free）的，真值相机仅用于视角采样 —— 高斯反投影与新视角渲染由模型自身预测的相机与深度驱动。

### 数据集

训练最多使用三个数据集，通过配置中的 `dataset_mode` / `mix_datasets` 选择：

- **DL3DV** —— 真实多视角场景
- **rendering** —— 合成渲染数据（闭源数据）
- **ScanNet++** —— 真实室内场景

每个数据集的根目录及布局字段（`data_root`、`val_root`、`image_dir_name` 等）都在配置文件中设置。所有随仓库提供的配置都指向作者内部的绝对路径，因此**你必须重新设置所用到的每一个数据集根目录**。

### 运行命令

单卡运行：

```bash
python train_gs.py --config configs/gsdpt_training_stage_3.yaml
```

单节点多卡（8 卡，通过 torchrun 进行 DDP）：

```bash
torchrun --nproc_per_node=8 train_gs.py \
    --config configs/gsdpt_training_stage_3.yaml
```

命令行覆盖使用点号表示法，叠加在 YAML 之上，例如
`optimizer.lr=5e-5 training.max_iterations=50000`。（`mix_ratio` 等列表型字段只能在 YAML 中设置。）

### 训练阶段

各配置构成一条渐进式课程，通过 checkpoint 路径串联 —— 请按顺序复现，并在每一步修改续训路径：

| 配置 | 数据 | 说明 |
| :--- | :--- | :--- |
| `gsdpt_training_stage_1.yaml` | DL3DV | GS 头的基础训练，核心损失 |
| `gsdpt_training_stage_2.yaml` | DL3DV + rendering | 引入合成渲染数据精调，续训自阶段 1 |
| `gsdpt_training_stage_3.yaml` | DL3DV + rendering + ScanNet++ | 最长的混合训练，完整功能集 |

### 训练前 —— 修改配置

在所选的 `configs/gsdpt_training_stage_*.yaml` 中，至少需要更新：

- `model.checkpoint_path` —— VGGT-Omega 主干路径（`vggt_omega_1b_512.pt`）
- `model.resume_gsdpt_checkpoint` / `model.resume_full_checkpoint` —— 全新训练设为 `null`，
  或指向上一阶段的 checkpoint 以续训
- `data.data_root` / `data.val_root`、`rendering_data.data_root`、`scannetpp_data.data_root`
  —— 你的数据集根目录（仅需配置 `dataset_mode` 用到的那些）
- `checkpoint.save_dir` —— checkpoint 的写出目录
- `wandb.offline: true` —— 设为 true 可关闭 Weights & Biases 日志

> **注意事项**
> - `training.batch_size` 必须为 `1`。请通过 `gradient_accumulation_steps` 和/或增加 DDP 卡数来
>   放大有效 batch。
> - `view_sampling.resolution_schedules` 中的分辨率必须是 16（patch 尺寸）的整数倍。
> - `train_gs.py` 在文件顶部硬编码了 WandB base URL 与 SOCKS5 代理。在普通机器上这会导致外网访问
>   失败 —— 请**删除或覆盖这几行**，或设置 `wandb.offline: true`。
> - 需要 CUDA GPU。

## 许可证

本代码的许可条款详见 [LICENSE](./LICENSE) 文件。

## 引用

本项目基于 VGGT-Omega：

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
