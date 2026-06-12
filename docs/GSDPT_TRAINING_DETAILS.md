# VGGT-Omega GSDPT 训练框架技术说明

本文档详细说明 VGGT-Omega 中 GSDPT（3D Gaussian Splatting 预测头）**训练框架**的全部技术细节，
包括训练原理、数据管线、损失函数、训练循环、配置参数含义与使用方法。该框架移植自
Depth-Anything-3（da3）的 `gs_training` 模块，并针对 VGGT-Omega 的 **pose-free（无位姿输入）**
架构做了适配。

> 配套阅读：推理侧细节见 [`GSDPT_TECHNICAL_DETAILS.md`](GSDPT_TECHNICAL_DETAILS.md)（GSDPTHead、
> GaussianAdapter、渲染、PLY 导出、通道布局）。本文聚焦**训练**：如何监督这颗未训练的头。

---

## 0. 背景与设计动机

VGGT-Omega 的公开 checkpoint **不含 GS 权重**，`gs_head` 仅靠初始化渲染出一个"占位"场景
（高斯位于反投影深度点、颜色取像素 RGB、不透明度 0.12、约 0.25 px 足迹）。要让它学会预测高质量
的逐像素高斯，需要一套训练框架——这正是本框架提供的能力。

核心思路：**冻结除 `gs_head` 外的一切**（aggregator、camera_head、dense_head、GaussianAdapter），
用这些冻结模块产生的**自洽相机位姿与深度**驱动高斯的反投影与新视角渲染，再用渲染图与真实图像
之间的 photometric / 感知损失监督 `gs_head`。只有 `gs_head`（约 3275 万参数）参与梯度更新。

### 0.1 模块清单

| 文件 | 职责 |
|---|---|
| `vggt_omega/training/config.py` | 全部训练配置（dataclass）+ YAML/CLI 加载 |
| `vggt_omega/training/losses.py` | 所有损失函数（photometric/LPIPS/depth/SSIM/Sobel/正则等）|
| `vggt_omega/training/utils.py` | 余弦退火+warmup 调度器、GSDPT 输出层初始化 |
| `vggt_omega/training/frustum_mask.py` | 目标视角"可见有效像素" mask（含 torch 几何 helper）|
| `vggt_omega/training/model_wrapper.py` | `GSModel`：包裹 VGGTOmega，定义 forward + compute_loss |
| `vggt_omega/training/train.py` | 训练主循环（DDP/AMP/梯度累积/检查点/wandb/验证）|
| `vggt_omega/training/data/transforms.py` | 图像加载（[0,1]）、内参缩放、坐标系转换 |
| `vggt_omega/training/data/view_sampler.py` | FPS 上下文采样、目标采样、curriculum 采样 |
| `vggt_omega/training/data/scene_dataset_base.py` | 场景数据集基类（采样+张量组装+缓存索引）|
| `vggt_omega/training/data/dl3dv_dataset.py` | DL3DV 数据集解析 |
| `vggt_omega/training/data/rendering_dataset.py` | 合成 rendering_dataset 解析 |
| `vggt_omega/training/data/datamodule.py` | dataloader 构建（dl3dv/rendering/mixed）|
| `train_gs.py` | 仓库根训练入口 |
| `configs/gsdpt_training.yaml` | 默认配置（DL3DV + 核心损失）|
| `configs/gsdpt_post_training.yaml` | 全功能配置（mixed + context 监督 + mask 等）|
| `requirements_train.txt` | 训练额外依赖（lpips/wandb/pyyaml/tqdm）|

---

## 1. 与 DA3 的关键差异（pose-free 适配）

这是理解整个框架最重要的一节。DA3 与 VGGT-Omega 的 backbone 接口不同，**不能照搬** DA3 的
model_wrapper，需重写适配层。

### 1.1 VGGT-Omega 是 pose-free —— 最核心的差异

| | DA3 | VGGT-Omega |
|---|---|---|
| 相机条件 | 有 `cam_enc`，把 GT 相机编码成 token 喂 backbone | **无**，aggregator 只吃图像 |
| pose-free 控制 | `camera_dropout_prob`（概率丢弃 GT 相机）| 永远 pose-free |
| 外参归一化 | `normalize_extrinsics`（喂模型前归一化）| **不需要** |
| 相机来源 | 训练时可用 GT 相机 | 用模型**自己预测**的 `pose_enc` |

**结论**：VGGT-Omega 训练等价于 DA3 `camera_dropout_prob=1.0`。因此：
- 数据集里的 GT 相机**只用于 view sampling**（选哪些帧当 context/target），**绝不喂给模型**。
- 高斯反投影、目标视角渲染，全部用模型预测的 extrinsic/intrinsic（`encoding_to_camera(pose_enc)`），
  保证几何自洽。
- 深度"teacher"也来自**同一模型的冻结 dense_head 输出**（detach），与 DA3 一致。

### 1.2 输入归一化

DA3 喂 ImageNet 归一化图、另存 [0,1] 原图。VGGT-Omega 的 Aggregator **内部做 ResNet 归一化**，
输入是 plain [0,1] RGB。所以数据集**只输出一种 [0,1] 图**，同时作模型输入、GT、images_merger 注入。

### 1.3 GS 头 / adapter / 渲染器接口

| | DA3 | VGGT-Omega |
|---|---|---|
| opacity | `map_pdf_to_opacity(raw_gs_conf)` | 头内已 **sigmoid**，直接是 [0,1] |
| adapter 内参 | normalized（fx/W）| **pixel-space**（fx 像素）|
| 渲染器内参 | normalized | **pixel-space** |
| patch_size | 14 | **16**（分辨率必须是 16 的倍数）|

**结论**：opacity 不再过 `map_pdf_to_opacity`；渲染/adapter 都用 pixel-space 内参（省掉归一化步骤）；
配置里的分辨率全部取 16 的倍数（如 272/480、384/672、512/896、544/960）。

### 1.4 跳过 slim 变体

DA3 的 slim GS 变体（38 通道↔精简通道裁剪、`pred_offset_xy`/`pred_offset_depth` 开关）与
VGGT-Omega 固定的 adapter 通道布局不兼容，**本框架不移植**。VGGT-Omega 的通道布局由
`gs_channel_layout(sh_degree)` 固定：`xy_offset(2) + scales(3) + quat(4, XYZW) + sh(3·d_sh) +
depth_offset(1)`，末位再加一个 opacity 通道（头内 sigmoid）。

---

## 2. 训练数据流与 forward 细节

`GSModel.forward(batch)`（`model_wrapper.py`）是训练的核心。输入 batch 含 `images`(1,K+M,3,H,W)、
`num_context=K`、`target_images`(1,M,3,H,W)、`context_images`(1,K,3,H,W) 等。完整流程：

```
images (1, K+M, 3, H, W) [0,1]
        │
        │  ① 冻结前向（no_grad）
        ├─► aggregator(images)  ── autocast(bf16/fp16) ──► tokens_list, patch_start
        │        │ （fp32, autocast off）
        │        ├─► camera_head ──► pose_enc (1,K+M,9)
        │        └─► dense_head  ──► depth (1,K+M,H,W,1), depth_conf (1,K+M,H,W)
        │
        │  encoding_to_camera(pose_enc,(H,W)) ──► extrinsic(1,K+M,3,4) w2c, intrinsic(1,K+M,3,3) 像素系
        │
        │  ② GSDPT 头（可训练, fp32）—— 仅 context 帧
        ├─► gs_head(tokens[:, :K], images[:, :K]) ──► raw_gs(1,K,H,W,d_in), gs_opacity(1,K,H,W)
        │
        │  ③ 相机系 → 世界系高斯（B=1, M=K·H·W）
        ├─► gs_adapter(extrinsic[:,:K], intrinsic[:,:K], depth[:,:K,...,0], gs_opacity, raw_gs, images[:,:K])
        │        └─► Gaussians(means, harmonics, opacities, scales, rotations)
        │
        │  ④ 渲染到（预测的）目标视角
        ├─► render_gaussians(gaussians, extrinsic[0,K:], intrinsic[0,K:], H, W)
        │        └─► rendered_imgs(M,3,H,W), rendered_depths(M,H,W)
        │
        │  ⑤⑥⑦ 可选：context 监督渲染 / 逐视角 self-recon / 目标视角 mask
        └─► 返回 dict（见下）
```

### 2.1 为什么所有帧一起过 aggregator？（transductive 训练）

VGGT-Omega 的 aggregator 是**跨帧联合注意力**，需要所有 K+M 帧同时在场才能预测出彼此自洽的相机
与深度。因此 `images`（全部 K+M 帧）整体进 aggregator/camera_head/dense_head。但 `gs_head` 只在
**context 帧**（`tokens[:, :K]`、`images[:, :K]`）上运行——高斯只从 context 视角生成，再渲染到
target 视角，与 target GT 算损失。

⚠️ **这是 transductive（直推式）/ self-consistent 训练，而非严格 held-out 的新视角合成**：
target 图像**确实参与**冻结 VGGT 的相机/深度/token 推理（它们和 context 帧一起进 aggregator，
为彼此提供跨帧上下文，并得到自洽的预测相机），只是 `gs_head` 不在 target 帧上生成高斯。换言之，
target 帧的**像素**不进入高斯生成，但 target 帧的**存在**会影响 frozen 模型对整组帧的相机/深度估计。
这是 VGGT-Omega pose-free 联合注意力架构的必然结果——若要严格 held-out，需让 target 帧完全不进
aggregator，但那样模型就无法预测出与 context 同一坐标系的 target 相机，也就无法渲染监督。

### 2.2 梯度边界

- aggregator + camera_head + dense_head：`torch.no_grad()`，且 `requires_grad_(False)` + `.eval()`。
  这是显存与正确性的关键——只有 `gs_head` 一颗头反传。
- gs_head + gs_adapter + render：在 `autocast(enabled=False)`（fp32）下运行。gsplat CUDA 核、SH 旋转、
  仿射逆等几何运算需要 fp32；gs_head 卷积也跑 fp32 以保持一致。主要显存节省来自冻结 backbone 的
  `no_grad` 前向。
- 深度 teacher = `frozen_depth = depth.detach()`（同模型 dense_head 输出），不反传。

### 2.3 forward 返回字段

| 字段 | 形状 | 含义 |
|---|---|---|
| `rendered_images` | (M,3,H,W) | 目标视角渲染图（监督主信号）|
| `rendered_depths` | (M,H,W) | 目标视角渲染深度（render-depth 正则用）|
| `frozen_depth_context` | (1,K,H,W) | context 帧冻结深度（depth_offset 损失的 teacher）|
| `frozen_depth_target` | (1,M,H,W) | target 帧冻结深度（render-depth 正则的 teacher）|
| `frozen_depth_conf_context/target` | (1,K/M,H,W) | 冻结深度置信度（损失加权用）|
| `gs_depth_context` | (1,K,H,W) | 加了 depth_offset 的 context 深度（depth_offset 损失的 pred）|
| `raw_offset_xy` | (1,K,H,W,2) | 原始 xy 亚像素偏移（offset_xy 正则用）|
| `raw_scale_logits` | (1,K,H,W,3) | 原始 scale logits（scale 正则用，sigmoid 前）|
| `opacities` | (1,K,H,W) | 不透明度 [0,1]（opacity_entropy 正则用）|
| `rendered_context` | (K,3,H,W) 或 None | context 视角渲染图（context 监督用）|
| `rendered_self` | (K,3,H,W) 或 None | 逐视角自重建渲染图（self-recon 用）|
| `target_loss_mask` | (1,M,H,W) bool 或 None | 目标视角可见有效像素 mask |

---

## 3. 损失函数（`losses.py` + `compute_loss`）

`compute_loss(fwd, batch)` 汇总所有损失项。总损失为：

```
loss = lambda_depth · depth_offset_loss
     + [include_target] · ( photometric + lambda_lpips·LPIPS
                            + lambda_ssim·(1-SSIM) + lambda_sobel·Sobel
                            + lambda_render_depth_l1·RenderDepthL1
                            + lambda_render_depth_grad_l1·RenderDepthGradL1 )
     + lambda_offset_xy·OffsetXY + lambda_opacity_entropy·OpacityEntropy + lambda_scale·Scale
     + lambda_context·ContextLoss          (若 supervise_context)
     + lambda_self_recon·SelfReconLoss      (若 lambda_self_recon>0)
```

其中 `include_target = not context_only`。下表逐项说明。

### 3.1 核心损失（始终计算）

| 损失 | 函数 | 公式 / 说明 |
|---|---|---|
| **Photometric** | `mse_loss` / `l1_loss` | 渲染图 vs GT target 图的 MSE 或 L1（由 `photometric_loss` 选）。支持 mask。|
| **LPIPS** | `LPIPSLoss` | VGG 感知损失。`enable_target_loss_mask=True` 时用 spatial 模式（逐像素图，可被 mask）；否则标量模式（输入 [-1,1]）。VGG 参数冻结，梯度只回传到渲染图。|
| **Depth offset** | `depth_offset_loss` | L1(gs_depth_context, frozen_depth_context)。约束 GSDPT 预测的 depth_offset 不发散。有置信度时按 `\|pred-teacher\|·conf` 加权。|

> 注意：渲染图做 LPIPS 前会 `clamp(0,1)`，但梯度通过未 clamp 的 photometric 路径回传。

### 3.2 可选正则项（默认权重 0）

| 损失 | 函数 | 说明 | 参数 |
|---|---|---|---|
| **SSIM** | `compute_ssim` | `1 - SSIM`，avg_pool2d 近似，窗口 11。结构相似度。| `lambda_ssim` |
| **Sobel** | `sobel_loss` | 灰度图 Sobel 边缘梯度的 L1，提升锐度。| `lambda_sobel` |
| **Offset XY** | `offset_xy_loss` | 约束 xy 亚像素偏移趋近 0，L1 或 L2。| `lambda_offset_xy`, `offset_xy_norm` |
| **Opacity entropy** | `opacity_entropy_loss` | 二元熵 `-p·log p-(1-p)·log(1-p)`，推不透明度趋向 0/1。| `lambda_opacity_entropy` |
| **Scale** | （内联）| 对 `sigmoid(raw_scale_logits)` 做 L1/L2，把尺度推向 scale_min。| `lambda_scale`, `scale_norm` |

### 3.3 目标视角渲染深度正则（vs 冻结深度）

| 损失 | 函数 | 说明 | 参数 |
|---|---|---|---|
| **Render-depth L1** | `render_depth_l1_loss` | L1(渲染深度, 冻结 target 深度)，仅在渲染深度>0 的有效像素，可按置信度加权。| `lambda_render_depth_l1` |
| **Render-depth grad L1** | `render_depth_gradient_l1_loss` | (渲染-teacher) 沿 x/y 的有限差分梯度 L1，鼓励结构一致而非绝对对齐。| `lambda_render_depth_grad_l1` |

### 3.4 Context 监督与 Self-recon

- **Context 监督**（`supervise_context`）：把**所有 context 视角高斯的并集**渲染回每个 context 视角，
  与 context GT 算 photometric + LPIPS（+可选 SSIM）。外层乘 `lambda_context`。
  `context_only=True` 时丢弃 target 损失的反传（仍计算用于日志），并自动开启 `supervise_context`。
- **逐视角 self-recon**（`lambda_self_recon>0`）：每个 context 视角**只用自己的高斯**渲染回自己
  （对角监督），与该视角 GT 算损失。实现上把 `Gaussians`（B=1,M=K·H·W）按视角切成 K 份逐帧渲染。

### 3.5 目标视角可见有效像素 mask（`frustum_mask.py`）

`enable_target_loss_mask=True` 时，用冻结的 teacher 深度/相机计算一个逐 target 像素的 bool mask，
保留同时满足三条件的像素：
1. **在视锥内**：该像素反投影成世界点后，重投影落在**任一** context 视角图像内且在相机前方；
2. **深度有效**：源深度 > 1e-6；
3. **深度一致**：重投影深度与 context 视角采样深度匹配（`torch.isclose`，由 `depth_atol`/`depth_rtol` 控制）。

该 mask 作用于 target 视角的 photometric / LPIPS / SSIM / Sobel 损失，过滤掉被遮挡或不可见的像素。

### 3.6 评测指标

`compute_psnr`（峰值信噪比，越高越好）、`compute_ssim`（结构相似度）、LPIPS（越低越好）。验证时对每个
数据集分别计算并取均值；检查点按**平均 PSNR** 选 top-k。

---

## 4. 数据管线（`training/data/`）

### 4.1 输出 schema

每个 `__getitem__` 返回**一个完整场景**（batch_size 固定为 1）。`_collate_fn` 给每个张量加 batch 维：

| 键 | 形状 | 说明 |
|---|---|---|
| `images` | (1,K+M,3,H,W) | [0,1] RGB，context 帧在前、target 帧在后 |
| `extrinsics` | (1,K+M,4,4) | GT w2c（OpenCV），实际只 c2w 用于采样，模型不消费 |
| `intrinsics` | (1,K+M,3,3) | GT 像素系内参（同上，模型用自己预测的）|
| `target_images` | (1,M,3,H,W) | target 帧 GT（= images 的后 M 帧）|
| `context_images` | (1,K,3,H,W) | context 帧 GT（= images 的前 K 帧）|
| `num_context` | int | K |
| `num_target` | int | M |
| `image_h/w` | int | 当前分辨率 |

> 注：`extrinsics`/`intrinsics` 字段保留是为了 schema 完整，但 pose-free 训练中模型并不消费它们；
> GT 相机的作用仅限于数据集内部的 view sampling。

### 4.2 视角采样（`view_sampler.py`）

- **FPS（Farthest Point Sampling）**：在 6-DoF 相机空间（平移 + `rotation_weight`·测地旋转距离）
  上贪心选 K 个 context 视角，最大化视角多样性。
- **目标采样**：从非 context 帧里均匀子采样 M 个 target 视角。
- **Curriculum 采样**：训练初期基线小（易），随 `global_step` 在 `warmup_steps` 内线性增大帧间隔
  （`min_context_gap_*`/`max_context_gap_*`），在局部窗口 `[start, start+gap]` 内做 FPS，
  防止大基线导致早期训练崩溃。验证时不用 curriculum。`global_step` 来自训练循环写入的共享
  `step_counter`（`multiprocessing.Value`），每个优化器步更新一次，对 dataloader worker 可见。

### 4.3 分辨率调度

`resolution_schedules` 是一组 `(resolution, context_views, target_views, weight)`。训练时按 `weight`
随机选一档，验证时固定用第一档（最小分辨率）。**分辨率必须是 patch_size=16 的倍数**
（`round_to_patch` 会向下取整到 16 的倍数）。

### 4.4 三种数据集模式（`datamodule.py` / `dataset_mode`）

| 模式 | 训练数据 | 验证数据 | 采样器 |
|---|---|---|---|
| `dl3dv` | DL3DV 各 bucket | DL3DV benchmark_140 | DistributedSampler |
| `rendering` | 合成 rendering_dataset | 按 `val_split_ratio` 切分 | DistributedSampler |
| `mixed` | DL3DV + rendering | 两者各自独立验证 | `WeightedConcatSampler`（按 `mix_ratio`）|

- **DL3DV**（`dl3dv_dataset.py`）：读 `transforms.json`，c2w 从 OpenGL/nerfstudio 翻转为 OpenCV。
- **rendering**（`rendering_dataset.py`）：读 `perspective_camera/*.json`，已是 OpenCV 约定，**不翻转**。
- **WeightedConcatSampler**：每次抽样先按 `mix_ratio`（归一化为概率）选子数据集，再在其中**有放回**
  均匀抽场景，使小数据集不被大数据集淹没。DDP 下每个 rank 用不同的、按 epoch 确定的 seed，得到各自
  独立的有放回采样流——**并非对场景集合的不相交划分**，不同 rank 同一步可能抽到同一场景；对大场景池
  而言这是预期的 weighted-with-replacement 行为。
- **场景索引缓存**：首次构建索引后 pickle 缓存到 `~/.cache/vggt_omega_gsdpt/`，加速后续启动。

> ⚠️ `mix_ratio` 必须通过 YAML 设置，**不能用 CLI 覆盖**——CLI override 不解析列表字面量
> （与 DA3 行为一致）。两个 config 都已在 YAML 里写好 `mix_ratio`。

---

## 5. 训练循环（`train.py`）

支持单卡与多卡（`torchrun` + DDP）、混合精度、梯度累积、`torch.compile`、wandb、检查点管理。

主循环以**优化器步（optimizer step）**为单位：每个 step 内累积 `accum` 个 micro-batch
（取 batch → `model(batch)` → `compute_loss` → `scaler.scale(loss/accum).backward()`），
累积完成后做一次梯度裁剪 + `optimizer.step()` + `scheduler.step()`。要点：

- **step 语义**：`step` 计的是**优化器步**而非 micro-step。LR 调度、验证、保存、`max_iterations`
  全部以优化器步为单位，因此 `gradient_accumulation_steps>1` 表现为「更大的有效 batch」而非把
  进度快进 `accum` 倍。
- **AMP 边界**：外层**不**包 autocast——wrapper 内部按 `config.training.mixed_precision` 管理
  aggregator 的 autocast dtype（`bf16`/`fp16` 启用对应精度；`fp32` 则完全不 autocast），
  gs_head + adapter + 渲染恒为 fp32。`GradScaler` 仅在 fp16 下启用。
- **梯度累积**：DDP 下非最后一个 micro-step 用 `no_sync()` 跳过 all-reduce，仅在最后一个
  micro-step 同步梯度。
- **curriculum 推进**：主循环持有一个 `multiprocessing.Value` 的 `step_counter`，每个优化器步
  写入当前 step；它对（持久化）dataloader worker 可见，使数据集的 curriculum 帧间隔随训练增大。
- **梯度裁剪**：`gradient_clip_val>0` 时先 `unscale_` 再 `clip_grad_norm_`。
- **首步诊断**：打印渲染图形状/值域、各损失分量、`gs_head` 各参数梯度范数（确认梯度确实流入 gs_head）。
- **检查点**：`CheckpointManager` 保留 last-k（按 step）+ top-k（按验证平均 PSNR）。state 含
  `gs_head`/`optimizer`/`scheduler`/`scaler`/`config`/`step`。
- **断点续训**：`resume_full_checkpoint` 恢复全部训练状态（含 optimizer/scheduler/scaler/step），
  并把 `step_counter` 同步到 `start_step`。
- **DDP 验证汇总**：`validate()` 在 DDP 下对各 rank 的损失/PSNR/SSIM/LPIPS 累加和与样本数做
  `all_reduce(SUM)`，再算全局均值——避免每个 rank 只用本地 shard 估计、rank0 据此选 top-k 产生偏差。
  验证块在所有 rank 上对称执行（仅打印/wandb/落盘部分 gate 在 rank0），保证集合通信不死锁。

### 5.1 优化器与调度器

- **优化器**：AdamW，仅优化 `trainable_parameters()`（= `gs_head.parameters()`）。
- **调度器**：`CosineAnnealingWarmupScheduler`——warmup 期 `lr = base_lr·(step+1)/warmup_steps` 线性升，
  之后余弦退火到 `min_lr`。

---

## 6. 配置参数完整说明

配置由 `config.py` 的 dataclass 定义，可用 YAML 文件 + CLI 点号覆盖（如 `optimizer.lr=1e-4`）。

### 6.1 `model`（ModelConfig）

| 参数 | 默认 | 含义 |
|---|---|---|
| `checkpoint_path` | `ckpts/VGGT-Omega/vggt_omega_1b_512.pt` | VGGT-Omega 主 checkpoint（strict=False 加载，gs_head 用初始化）|
| `gs_sh_degree` | 0 | 球谐阶数 0~2。>0 时颜色含视角相关项，需 `e3nn`（SH 世界系旋转）|
| `resume_gsdpt_checkpoint` | null | 仅恢复 gs_head 权重（不含优化器状态）|
| `resume_full_checkpoint` | null | 恢复完整训练状态（gs_head+optimizer+scheduler+scaler+step）|
| `freeze_backbone` | true | 保留接口；本框架始终冻结非 gs_head |
| `init_mode` | `finetune` | `finetune`=保留当前 gs_head 权重；`reset_output`=重置输出层；`scratch`=全层重初始化 |

### 6.2 `data`（DataConfig，DL3DV）

| 参数 | 默认 | 含义 |
|---|---|---|
| `data_root` | DL3DV-10K/960p | 训练 bucket 根目录 |
| `train_buckets` | 1K~11K | 参与训练的 bucket 列表 |
| `val_root` | benchmark_140 | 验证集根目录 |
| `val_scene_subdir` | `nerfstudio` | 每个验证场景内含 transforms.json 的子目录 |
| `max_train_scenes`/`max_val_scenes` | null | 调试用场景数上限 |
| `num_workers` | 8 | dataloader 工作进程数 |
| `pin_memory` | true | 固定内存 |
| `image_dir_name` | `images_4` | 场景内图像子目录（4× 下采样）|
| `val_context_gap` | null | 验证时从前 N 帧采 context（None=全帧）|
| `view_sampling`/`curriculum` | null | 可选的逐数据集覆盖（None=用全局）|

### 6.3 `rendering_data`（RenderingDataConfig）

| 参数 | 默认 | 含义 |
|---|---|---|
| `data_root` | pano rendering_dataset | 合成数据根 |
| `val_split_ratio` | 0.02 | 每个 scene_type 末尾 ceil(ratio) 作验证 |
| `image_dir_name` | `perspective_frames` | 透视帧目录 |
| `camera_dir_name` | `perspective_camera` | 相机 JSON 目录 |
| `max_train_scenes`/`max_val_scenes`/`num_workers`/`pin_memory`/`val_context_gap` | — | 同上语义 |

### 6.4 `view_sampling`（ViewSamplingConfig）

| 参数 | 默认 | 含义 |
|---|---|---|
| `resolution_schedules` | 三档 | 列表，每档 `{resolution:[H,W], context_views, target_views, weight}`，**H/W 须为 16 倍数** |
| `fps_rotation_weight` | 1.0 | FPS 中旋转距离相对平移的权重 |
| `min_scene_frames` | 20 | 少于此帧数的场景跳过 |

每档 `ResolutionSchedule`：`resolution`（[H,W]）、`context_views`（K）、`target_views`（M）、
`weight`（被选中的相对概率）。

### 6.5 `curriculum`（CurriculumConfig）

| 参数 | 默认 | 含义 |
|---|---|---|
| `enabled` | true | 是否启用 curriculum 采样 |
| `min_context_gap_start`/`end` | 15/20 | context 边界视角最小帧间隔（warmup 内线性插值）|
| `max_context_gap_start`/`end` | 30/50 | 最大帧间隔 |
| `warmup_steps` | 10000 | 达到终值的步数 |

### 6.6 `training`（TrainingConfig）

| 参数 | 默认 | 含义 |
|---|---|---|
| `max_iterations` | 100000 | 总**优化器步**数（非 micro-step）|
| `batch_size` | 1 | 固定 1（每步一个场景；保留字段，未参与分派）|
| `gradient_accumulation_steps` | 1 | 梯度累积步数；>1 等效更大 batch，进度仍按优化器步计 |
| `mixed_precision` | `bf16` | `bf16`/`fp16`/`fp32`，控制冻结 aggregator 的 autocast dtype（gs_head/渲染恒 fp32）|
| `gradient_clip_val` | 1.0 | 梯度范数裁剪阈值（≤0 关闭）|
| `torch_compile` | true | 对 gs_head 应用 torch.compile |

### 6.7 `optimizer` / `scheduler`

| 参数 | 默认 | 含义 |
|---|---|---|
| `optimizer.name` | adamw | 优化器（当前仅 AdamW）|
| `optimizer.lr` | 2e-5 | 学习率 |
| `optimizer.weight_decay` | 0.05 | 权重衰减 |
| `optimizer.betas` | [0.9,0.95] | Adam β |
| `optimizer.eps` | 1e-8 | 数值稳定项 |
| `scheduler.name` | cosine_annealing | 调度器 |
| `scheduler.warmup_steps` | 2000 | 线性 warmup 步数 |
| `scheduler.min_lr` | 1e-6 | 退火下限 |

### 6.8 `loss`（LossConfig）—— 全部权重

| 参数 | 默认 | 含义 |
|---|---|---|
| `photometric_loss` | `mse` | `mse` 或 `l1` |
| `lambda_lpips` | 0.05 | LPIPS 权重 |
| `lpips_net` | `vgg` | LPIPS backbone |
| `lambda_depth` | 0.5 | depth_offset 损失权重 |
| `lambda_ssim` | 0.0 | (1-SSIM) 权重 |
| `lambda_sobel` | 0.0 | Sobel 边缘损失权重 |
| `lambda_offset_xy` | 0.0 | xy 偏移正则权重 |
| `offset_xy_norm` | `l2` | xy 正则范数（l1/l2）|
| `lambda_opacity_entropy` | 0.0 | 不透明度熵正则权重 |
| `supervise_context` | false | 是否额外监督 context 视角 |
| `lambda_context` | 0.5 | context 监督外层权重 |
| `context_only` | false | 仅 context 监督（丢弃 target 反传，隐含 supervise_context）|
| `lambda_self_recon` | 0.0 | 逐视角自重建权重 |
| `lambda_scale` | 0.0 | scale logit 正则权重 |
| `scale_norm` | `l2` | scale 正则范数（l1/l2）|
| `lambda_render_depth_l1` | 0.0 | 渲染深度 L1 正则权重 |
| `lambda_render_depth_grad_l1` | 0.0 | 渲染深度梯度 L1 正则权重 |
| `enable_target_loss_mask` | false | 启用目标视角可见有效像素 mask（并切到 spatial LPIPS）|
| `target_loss_mask_depth_atol` | 0.1 | mask 条件 3 深度匹配绝对容差 |
| `target_loss_mask_depth_rtol` | 0.0 | mask 条件 3 深度匹配相对容差 |

### 6.9 `validation` / `checkpoint` / `wandb` / `distributed` / 顶层

| 参数 | 默认 | 含义 |
|---|---|---|
| `validation.val_interval` | 500 | 每多少步验证一次 |
| `validation.num_preview_scenes` | 4 | wandb 预览场景数 |
| `checkpoint.save_top_k` | 10 | 按 PSNR 保留的最优检查点数 |
| `checkpoint.save_last_k` | 3 | 保留的最新检查点数 |
| `checkpoint.save_dir` | `checkpoints/gsdpt` | 检查点目录 |
| `wandb.project` | `vggt-omega-gsdpt-training` | wandb 项目名 |
| `wandb.name` | null | run 名 |
| `wandb.log_every_n_steps` | 50 | 日志间隔 |
| `wandb.offline` | false | 离线模式（不上传）|
| `distributed.backend` | `nccl` | DDP 后端（nccl/gloo）|
| `seed` | 42 | 随机种子（各 rank 加 rank 偏移）|
| `dataset_mode` | `dl3dv` | `dl3dv`/`rendering`/`mixed` |
| `mix_ratio` | [1.0,1.0] | mixed 模式下 [dl3dv, rendering] 采样权重（仅 YAML 可设）|

---

## 7. 使用方法

### 7.1 环境

训练依赖 `lpips`/`wandb`/`pyyaml`/`tqdm`（`requirements_train.txt`）+ `gsplat`/`e3nn`/`imageio`
（`requirements_gs.txt`）。本机已有 `da3` conda 环境装齐全部依赖：

```bash
PY=/root/miniconda3/envs/da3/bin/python   # torch 2.6+cu124, gsplat, lpips, wandb, e3nn 均已就绪
# 如需在新环境安装：
# pip install -r requirements_gs.txt -r requirements_train.txt
```

### 7.2 启动训练

```bash
# 单卡，默认 DL3DV 配置
$PY train_gs.py --config configs/gsdpt_training.yaml

# 多卡（8 卡），全功能 mixed 配置
torchrun --nproc_per_node=8 train_gs.py --config configs/gsdpt_post_training.yaml \
    training.max_iterations=30000 optimizer.lr=5e-5

# 断点续训
torchrun --nproc_per_node=8 train_gs.py --config configs/gsdpt_post_training.yaml \
    model.resume_full_checkpoint=checkpoints/gsdpt-post-training/gsdpt-latest-22400.pt
```

CLI 覆盖用点号路径（`optimizer.lr=`、`loss.lambda_lpips=` 等）；`mix_ratio` 等列表只能在 YAML 设。

### 7.3 冒烟测试（最小验证）

```bash
$PY train_gs.py --config configs/gsdpt_training.yaml \
    data.max_train_scenes=2 data.max_val_scenes=1 \
    training.max_iterations=2 validation.val_interval=2 \
    wandb.offline=true training.mixed_precision=fp32 \
    curriculum.enabled=false data.num_workers=0 \
    checkpoint.save_dir=/tmp/gsdpt_smoke
```

预期：模型加载（"GS head not in checkpoint; using initialized weights"）、渲染出 (M,3,H,W)、
loss 为正、首步 DIAG 显示 gs_head 68 个张量有梯度、验证打印 PSNR、检查点落盘。
其中首步 `depth=0.0000` 是**正常的**——finetune 初始化下 depth_offset 通道恰为 0，
`gs_depth == frozen_depth`，随训练才偏离。

---

## 8. 已验证清单

- 全部模块编译 + 导入通过；两个 config 加载正常；CLI 覆盖类型强转正确；未知 key 会告警。
- frustum mask：helper 形状正确；修复半像素双偏移后，自洽性测试（同视角同深度）在 atol=0.01 下
  即达 frac=1.0。
- 端到端冒烟（2 场景）：32.7M 可训练 / 1.18B 总参；渲染 (4,3,272,480)；68 个 gs_head 张量有梯度；
  验证 + top-k/last-k 检查点落盘。
- 梯度累积语义：`accum=2` 跑 3 步在 3 个**优化器步**后结束（非 6 micro-step），检查点名为
  `gsdpt-3-*`，LR/验证/保存均按优化器步推进。
- `mixed_precision=fp32`：渲染 dtype 为 fp32，aggregator 不再被强制 bf16 autocast，训练正常。
- curriculum 开启 + `step_counter` 联通：无崩溃，counter 每优化器步推进。
- 全部可选损失同时开启：无形状错误，loss/梯度随项增多上升。
- mixed 模式：DL3DV + rendering 均解析、训练，且两个验证集分别评测。
- 梯度隔离：恰好 gs_head（32,748,206 参数）可训练，其余为 0；冻结模块 eval、gs_head train。

### 8.1 配置校验与已知限制

`load_config` 加载后会调用 `_validate_config` 做 fail-fast 校验，避免「改了配置却静默不生效」：

- **配置文件路径**：显式传入但不存在的 `--config` 路径会 **raise `FileNotFoundError`**，
  不会静默回退到默认配置。
- **未知配置 key**：YAML 里的未知 key **告警并跳过**；CLI 覆盖里的未知 key **raise ValueError**
  （防 `training.lr` 这类 typo——正确键是 `optimizer.lr`）。
- **枚举类字段校验**：`dataset_mode`/`optimizer.name`/`scheduler.name`/`training.mixed_precision`/
  `loss.photometric_loss`/`loss.offset_xy_norm`/`loss.scale_norm` 取值不在白名单内会直接报错。
- **`training.batch_size` 必须为 1**：数据管线每步处理一个完整场景（K/M/分辨率可变，无法 stack），
  设其他值会报错；更大有效批量请用 `gradient_accumulation_steps` 或 DDP 多卡。
- **正整数 / 正数约束**：`max_iterations`/`gradient_accumulation_steps`/`val_interval`/
  `log_every_n_steps`/`warmup_steps` 须为正整数，`optimizer.lr` 须 >0。
- **分辨率须为 16 倍数**：全局与逐数据集的 `resolution_schedules` 都会校验（VGGT-Omega patch_size=16）。
- **`mix_ratio`**：`mixed` 模式下必须恰好 2 个元素 `[dl3dv, rendering]`；且只能 YAML 设置，
  CLI 不解析列表字面量（与 DA3 一致）。
- **`optimizer.name` / `scheduler.name`**：当前仅实现 AdamW + CosineAnnealingWarmup，白名单只含这两者。
- **`torch_compile`**：开启时 `gs_head` 被 `torch.compile` 包裹；检查点保存/加载已通过
  `_gs_head_module()` 解包 `_orig_mod`，所以权重对未 compile 的 `GSDPTHead` 可移植。
- **DDP 梯度累积通信**：非最后一个 micro-step 的 forward+backward 整体包在 `no_sync()` 内，
  跳过 all-reduce；仅最后一个 micro-step 同步（正确利用 DDP 的累积优化）。

---

## 9. 不改动的现有文件

训练框架复用了以下既有推理代码，**未修改**它们：
`vggt_omega/models/`（vggt_omega.py / gs_adapter.py / heads/gsdpt_head.py / aggregator.py /
camera_head.py / dense_head.py）、`utils/gs_renderer.py`、`utils/pose_enc.py`、
`utils/geometry.py`（`closed_form_inverse_se3`）、`inference_pipeline.py` 的 `load_model`。
GSDPT 输出层初始化复用 `gsdpt_head._init_gs_prediction_head`，使通道布局保持单一来源。

