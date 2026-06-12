# VGGT-Omega 技术说明文档

本文档详细说明 VGGT-Omega 模型的架构、各组件技术细节、数值精度策略、坐标系约定与推理管线，所有内容均依据本仓库代码（`vggt_omega/`）逐文件核对。

---

## 1. 总览

VGGT-Omega 是一个前馈式（feed-forward）多视图三维重建模型：输入 N 帧 RGB 图像，单次前向传播同时预测每帧的**深度图**、**深度置信度**和**相机位姿**（9D 编码，可解码为外参 + 内参）。模型不含点图（point map）头、跟踪头或高斯头。

整体结构（`vggt_omega/models/vggt_omega.py`）：

```
images (B,N,3,H,W)
   │  ResNet 归一化（Aggregator 内部）
   ▼
Aggregator（DINOv3 patch embed + 24 层交替注意力）
   │  缓存第 {4,11,17,23} 层输出，每层 2×1024=2048 维
   ▼
┌──────────────┬──────────────┬──────────────────────┐
│ CameraHead   │ DenseHead    │ TextAlignmentHead    │
│ → pose_enc   │ → depth      │ → text_alignment_*   │
│   (B,N,9)    │   depth_conf │   （仅 text 检查点）   │
└──────────────┴──────────────┴──────────────────────┘
```

### 参数量（meta device 实测）

| 组件 | 参数量 |
|---|---|
| Aggregator 合计 | 907.9 M |
| ├─ patch_embed（DINOv3 ViT-L） | 303.2 M |
| ├─ frame_blocks ×24 | 302.4 M |
| └─ inter_frame_blocks ×24 | 302.4 M |
| CameraHead | 203.6 M |
| DenseHead | 32.3 M |
| TextAlignmentHead（可选） | 205.7 M |
| **总计（512 检查点，无 text head）** | **1143.8 M** |
| **总计（256-text 检查点）** | **1349.5 M** |

### 构造参数（`VGGTOmega.__init__`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `patch_size` | 16 | ViT patch 大小 |
| `embed_dim` | 1024 | 主干 token 维度（head 输入为 2×1024） |
| `enable_camera` | True | 是否构建 CameraHead |
| `enable_depth` | True | 是否构建 DenseHead |
| `enable_alignment` | False | 是否构建 TextAlignmentHead（加载 `_text` 检查点时必须为 True） |

---

## 2. 输入预处理（`vggt_omega/utils/load_fn.py`）

`load_and_preprocess_images(image_path_list, mode="balanced", image_resolution=512, patch_size=16)` → `(N, 3, H, W)`，值域 [0, 1]。

处理步骤依次为：

1. **RGBA 处理**：RGBA 图像先合成到白色背景，再转 RGB。
2. **宽高比裁剪**（`_crop_to_supported_aspect_ratio`）：宽高比（H/W）被中心裁剪钳制到 **[0.5, 2.0]**。
3. **目标尺寸计算**：
   - `mode="balanced"`（默认）：保持总 patch token 数 ≈ `(image_resolution/patch_size)²`（512 时为 1024 个 token）。计算 `w_patches = √(token数/ar)`、`h_patches = token数/w_patches`，四舍五入后乘以 patch_size。例如 3:2 横版图 → 624×416（39×26=1014 个 patch）。
   - `mode="max_size"`：最长边缩放到 `image_resolution`，另一边取整到 patch_size 倍数。同等输入下 token 更少、显存更省。
4. **缩放**：PIL BICUBIC 插值。
5. **ToTensor**：(3, H, W) float32，[0, 1]。
6. **混合尺寸 padding**（`_pad_images_to_common_size`）：批内尺寸不一致时，对称 padding 到最大尺寸，**填充值为 1.0（白色）**，并发出 warning。

注意：**ResNet 归一化不在这里做**，而在 Aggregator 内部（mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]，注册为非持久 buffer）。脚本侧只需提供 [0,1] RGB。

`image_resolution` 必须能被 `patch_size` 整除；输出 H、W 始终是 16 的倍数。

---

## 3. Aggregator（`vggt_omega/models/aggregator.py`）

交替注意力（alternating-attention）编码器，是模型主体。

### 3.1 DINOv3 patch embed

`_build_patch_embed` 构建一个完整的 **DINOv3 ViT-L/16**（`models/layers/vision_transformer.py`）作为 patch tokenizer：

- embed_dim=1024，depth=24，heads=16，Mlp FFN（ratio 4），LayerScale init 1e-5；
- norm 层为 `layernormbf16`（LayerNorm，eps=1e-5）；
- 1 个 cls token + **4 个 storage token**（`n_storage_tokens=4`）；
- 自带 RoPE（base=100，`normalize_coords="max"`，fp32）；
- `mask_k_bias=True`（见 §4.2）。

Aggregator 只取其输出 dict 的 **`x_norm_patchtokens`**（norm 后的 patch token），cls/storage token 被丢弃。代码注释标注这是 VGGT-Omega 对 DINOv3 的改动：默认 forward 直接返回特征 dict。

注意整个 patch embed 是**一个 24 层 ViT**（303M 参数），不是单个卷积投影——每帧图像先被 DINOv3 完整编码一遍，得到的 patch 特征再进入交替注意力层。

### 3.2 特殊 token 与第一帧锚定

每帧 token 序列为：`[camera_token(1), register_tokens(16), patch_tokens(h/16 × w/16)]`，
`patch_token_start = 17`。

`camera_token` 形状 `(1, 2, 1, 1024)`、`register_token` 形状 `(1, 2, 16, 1024)`：**索引 0 专用于第 0 帧，索引 1 由其余所有帧共享**（`slice_expand_and_flatten`）。这使第一帧在模型中具有特殊身份——**世界坐标系锚定在第一个相机上**（实测第 0 帧输出外参 ≈ 单位矩阵）。

含义：输入帧的顺序有意义，第一帧定义世界系；交换帧序会改变输出坐标系。

### 3.3 交替注意力（24 层 × 2 块）

每层依次执行两个独立的 `SelfAttentionBlock`（dim=1024，16 头，qkv_bias，LayerScale 1e-5，**use_qk_norm=True**，mask_k_bias=True）：

1. **frame block（帧内注意力）**：token reshape 为 `(B·N, 17+hw, 1024)`，每帧内部做完整自注意力，patch token 使用 RoPE（camera/register token 不加 RoPE，见 §3.4）。
2. **inter-frame block（帧间注意力）**，按层分两种类型（`inter_frame_attention_types`）：
   - **`global`**（大多数层）：所有帧的所有 token 拼成 `(B, N·(17+hw), 1024)` 做全局注意力，不加 RoPE。这是跨帧几何信息交换的主通道，也是序列长度二次方显存开销的来源。
   - **`register`**（第 **[2, 6, 9, 14, 20]** 层）：只有各帧的 camera+register token（每帧 17 个）跨帧做注意力，patch token 原样跳过。相当于低成本的跨帧"摘要"交换层。

### 3.4 RoPE 位置编码（`models/layers/rope_position_encoding.py`）

- 轴向（axial）RoPE，无可学习参数（periods 为持久 buffer）；
- 坐标按 `normalize_coords="max"` 归一化：`(i+0.5)/max(H,W)`，再映射到 [-1, 1]——保证非正方形输入下两轴尺度一致（**released 检查点要求 "max"**，`_warn_if_rope_not_max` 会在不一致时发 warning）；
- 每头维度 D_head=64，periods = `100^(2i/32)`，i=0..15；角度 = `2π·coord/period`，sin/cos 形状 `[HW, 64]`，配对方式为 rotate-half（x_i 与 x_{i+32}）；
- Aggregator 的 RoPE 以 **fp32** 计算（`dtype=torch.float32`），q/k 先转 fp32 旋转再转回原 dtype；
- `apply_rope` 中通过 prefix 逻辑跳过序列前部的非 patch token（camera/register）；
- 训练期增广（shift/jitter/rescale coords）在推理时全部关闭；
- 每次 forward 在 `torch.no_grad()` 下只计算一次，供全部 24 层 frame block 复用。

### 3.5 层缓存

`cached_layer_indices = {4, 11, 17, 23}`。对每个缓存层，输出为
`cat([frame_block输出, inter_frame_block输出], dim=-1)` → **(B, N, 17+hw, 2048)**；
非缓存层位置存 `None` 以省显存。返回值是长度 24 的列表 + `patch_token_start`。

各 head 的输入因此都是 **2048 维**：CameraHead / TextAlignmentHead 只用最后一层（索引 23），DenseHead 用全部四个缓存层做多尺度融合。

---

## 4. 基础层细节（`vggt_omega/models/layers/`）

### 4.1 SelfAttentionBlock（`block.py`）

标准 pre-norm 结构：`x + LS1(Attn(LN(x)))` → `x + LS2(Mlp(LN(x)))`。LayerScale（`layer_scale.py`）初值 1e-5。drop_path（随机深度）仅训练用，推理路径为普通残差。`forward` 内部统一走 list 实现（`_forward_list`），单 tensor 会包装成 `[x]`。

### 4.2 SelfAttention（`attention.py`）

- QKV 合并线性层；`torch.nn.functional.scaled_dot_product_attention`（自动选 FlashAttention/高效后端）；
- **use_qk_norm**（VGGT-Omega 对 DINOv3 上游的改动，代码注释明确标注）：对 q、k 分别做 per-head `LayerNorm(head_dim=64, eps=1e-5)`。**Aggregator 的 48 个 block 启用；DINOv3 patch embed 内部和各 head 的 trunk 不启用**；
- **mask_k_bias**（`LinearKMaskedBias`）：QKV 投影的 bias 乘以 `bias_mask` buffer——q、v 三分之一段为 1，k 段为 0，即 **K 投影实际无 bias**，但保留权重布局兼容性。`bias_mask` 随检查点保存/加载；
- `CausalSelfAttention` / `CausalSelfAttentionBlock` 在代码中存在但**未被任何 head 使用**。

### 4.3 FFN 与 Norm

- 全模型实际使用的 FFN 都是 `Mlp`（fc1 → GELU → fc2，hidden = 4×dim）；`SwiGLUFFN` 与 `RMSNorm` 在 `ffn_layer_dict`/`norm_layer_dict` 中可选但未被当前配置启用。

---

## 5. CameraHead（`vggt_omega/models/heads/camera_head.py`）

输入：最后一个缓存层（2048 维）中每帧前 17 个 token（camera + register）。

流程：

1. `token_norm`（LayerNorm 2048）；
2. 所有帧的 camera/register token 拼成 `(B, N×17, 2048)`，过 **4 层 SelfAttentionBlock**（16 头，无 qk_norm，无 RoPE，mask_k_bias=True）——head 内部再次跨帧交换位姿信息；
3. `trunk_norm` 后**只取每帧 camera token（索引 0）**；
4. MLP：`Linear(2048→1024) → GELU → Linear(1024→9)`；
5. 激活（`_apply_camera_activation`）：平移 3 维原样；四元数 4 维**原样输出（不归一化）**；FoV 2 维 `ReLU(x)+0.01`（保证为正）。

这是单次前馈（4 个注意力块），不同于原版 VGGT 的迭代细化式 camera head。整个 head 在 fp32 下运行（输入强制 `.float()`）。

### 5.1 9D 位姿编码与解码（`vggt_omega/utils/pose_enc.py`）

编码布局：`[tx, ty, tz, qx, qy, qz, qw, fov_v, fov_w]`

| 段 | 含义 |
|---|---|
| [0:3] | 平移 T（camera-from-world，即 w2c 的 t） |
| [3:7] | 旋转四元数，**XYZW 顺序（scalar-last）**，标准化为 qw≥0 |
| [7] | 垂直 FoV：`2·atan((H/2)/fy)` |
| [8] | 水平 FoV：`2·atan((W/2)/fx)` |

`encoding_to_camera(pose_enc, image_size_hw)` 返回：

- **extrinsic (B,N,3,4)**：`[R|t]`，**camera-from-world（w2c），OpenCV 相机系约定**（+X 右、+Y 下、+Z 前）；
- **intrinsic (B,N,3,3)**：`fy = (H/2)/tan(fov_v/2)`、`fx = (W/2)/tan(fov_w/2)`，**主点固定为图像中心 (W/2, H/2)**，无 skew。这是模型的内置假设：无法表达偏心主点。

`quat_to_mat`（`utils/rotation.py`，改自 PyTorch3D）通过 `two_s = 2/|q|²` 内含归一化，因此 head 输出的非单位四元数也能得到正确旋转矩阵。`mat_to_quat` 用四候选数值稳定法。

### 5.2 坐标系与尺度

- 世界系 = 第一帧相机系（实测第 0 帧外参 ≈ I）；
- 单目/多视图重建固有**尺度不确定性**：输出深度与平移的绝对数值无米制意义，但两者尺度互相一致（同一场景内部自洽，可直接用于反投影）。实测场景中深度值通常在 0.1~几 的量级（场景尺度归一化）。

---

## 6. DenseHead（深度 + 置信度，`vggt_omega/models/heads/dense_head.py`）

DPT 风格多尺度融合头（注释标注受 Depth-Anything-V2 启发），仅 32M 参数。

### 6.1 多尺度特征构建

输入：缓存层 `intermediate_layer_idx = [4, 11, 17, 23]` 的 **patch token**（去掉前 17 个 token，2048 维，强制 fp32）。对每个尺度 i：

1. 共享 `LayerNorm(2048)`；
2. reshape 为 `(B·N, 2048, h, w)`，其中 h=H/16、w=W/16；
3. 1×1 conv 投影到通道 `[256, 512, 1024, 1024]`（浅层→深层）；
4. **加位置编码**（`_apply_pos_embed`）：`create_uv_grid`（MoGe 风格，按对角线归一化、保持宽高比的 UV 网格）→ `position_grid_to_embed`（sincos，ω₀=100）→ **乘 0.1** 后逐元素相加；
5. 尺度调整（`resize_layers`）：×4 ConvTranspose(k4,s4)、×2 ConvTranspose(k2,s2)、×1 Identity、×0.5 Conv(k3,s2) → 四个特征图相对原图分别为 1/4、1/8、1/16、1/32。

### 6.2 DPT 融合与输出

1. `scratch.layerX_rn`：各尺度 3×3 conv（无 bias）统一到 256 通道；
2. `refinenet4 → 3 → 2 → 1` 自深向浅融合：每个 `FeatureFusionBlock` = 残差分支 `ResidualConvUnit`（ReLU-conv-ReLU-conv + skip）+ 双线性上采样（**align_corners=True**）+ 1×1 conv。融合输出为原图 **1/4** 分辨率、256 通道；
3. 再次加同款位置编码；
4. 两个独立的 1×1 conv 预测头（`proj` 深度 / `proj_conf` 置信度），各输出 `(patch_size/4)² = 16` 通道；
5. `F.pixel_shuffle(4)`：从 1/4 分辨率重排到**全分辨率**单通道。

### 6.3 激活与置信度语义

```
depth      = exp(depth_logits)            # 恒正
depth_conf = 1 + exp(conf_logits)         # 恒 > 1，无上界
```

- 深度走对数参数化，无上限；
- **置信度不是 [0,1] 概率**：值域 (1, +∞)，实测典型范围约 1.0~20+。过滤点云时应使用**百分位阈值**（demo 默认保留前 80%，即 `conf >= np.percentile(conf, 20)`），绝对阈值（如 0.9）无意义；
- `proj_conf` 初始化：权重置零、bias = `log(0.05)`，使训练起点 conf ≈ 1.05（接近"不自信"），这是 DUSt3R/VGGT 一脉的 confidence-aware 损失常用设置；
- 输出强制校验为 fp32，否则抛 `TypeError`。

### 6.4 显存控制

- `frames_chunk_size=8`（forward 默认）：帧数多时 DenseHead 按 8 帧一组分块计算后 concat，主干特征复用，仅 head 激活分块；
- `custom_interpolate`：当插值元素数超过 `1610612736`（接近 int32 上限）时按 batch 维分块调用 `F.interpolate`，规避 PyTorch 大张量插值溢出。

---

## 7. TextAlignmentHead（`vggt_omega/models/heads/text_alignment_head.py`）

仅 `vggt_omega_1b_256_text.pt` 检查点包含此头（205.7M 参数）。从整段序列读出一个语言对齐的全局嵌入：

1. 取最后缓存层各帧 camera+register token（fp32 + LayerNorm）；
2. 前置一个**可学习 language token**（trunc_normal 0.02 初始化），拼成 `(B, 1+N×17, 2048)`；
3. 过 4 层 SelfAttentionBlock（同 CameraHead trunk 配置）；
4. 取出 language token → LayerNorm → 投影器 `Linear(2048→1024) → GELU → LayerNorm → Linear(1024→2048)`。

输出：

- `text_alignment_embedding`：**L2 归一化**后的 2048 维嵌入（用于与文本嵌入做余弦相似度）；
- `text_alignment_token`：归一化前的原始 token。

---

## 8. 数值精度策略（`VGGTOmega.forward`）

| 阶段 | 精度 |
|---|---|
| Aggregator（含 DINOv3） | `torch.autocast(cuda)`，**bf16**（不支持则 fp16） |
| RoPE sin/cos | fp32（旋转在 fp32 中进行后转回） |
| 所有 head | `autocast(enabled=False)` + 输入强制 `.float()`，**纯 fp32** |
| DenseHead 输出 | 强制 fp32 校验 |

**调用方不应在外层再包 autocast**——模型自己管理混合精度；外层只需 `torch.inference_mode()`。

其他 forward 行为：

- 输入接受 `(N,3,H,W)` 或 `(B,N,3,H,W)`（4 维自动 unsqueeze 出 B=1）；
- 输出 dict 固定包含 `camera_and_register_tokens` (B,N,17,2048)；按开关包含 `pose_enc`、`depth`、`depth_conf`、text 相关键；**eval 模式下额外返回 `images`**（归一化前的输入，供 head 之外的下游用）；
- 构造时 `_warn_if_rope_not_max` 检查 RoPE 配置与 released 检查点一致。

---

## 9. 显存与序列长度

README 实测（A100，512 检查点，624×416 输入，含权重加载的端到端峰值）：

| 帧数 | 1 | 10 | 25 | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|---|---|---|
| 峰值显存 (GB) | 6.02 | 6.67 | 7.80 | 9.66 | 13.37 | 20.82 | 28.26 | 35.71 | 43.15 |

主导项是 global inter-frame attention 的 `N·(17+hw)` 长度序列。降低显存的途径：减少帧数、降低 `image_resolution`、或用 `mode="max_size"`（同分辨率下 token 更少）。

---

## 10. 检查点与加载

| 文件 | 分辨率 | text head | 加载方式 |
|---|---|---|---|
| `ckpts/VGGT-Omega/vggt_omega_1b_512.pt` | 512 | 无 | `VGGTOmega()` |
| `ckpts/VGGT-Omega/vggt_omega_1b_256_text.pt` | 256 | 有 | `VGGTOmega(enable_alignment=True)` |

检查点是裸 `state_dict`（`torch.load` + `load_state_dict`，无包装 dict）。`inference_pipeline.load_model` 会根据 state_dict 中是否存在 `text_alignment_head.` 前缀自动选择 `enable_alignment`。使用 text 检查点时应配 `image_resolution=256`。

---

## 11. 推理管线与工具

### 11.1 标准推理流程

```python
images = load_and_preprocess_images(paths, image_resolution=512).to("cuda")  # (N,3,H,W) [0,1]
with torch.inference_mode():
    pred = model(images)
extrinsic, intrinsic = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
points = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)  # (N,H,W,3) 世界系
```

### 11.2 几何工具（`vggt_omega/utils/geometry.py`）

- `closed_form_inverse_se3`：SE(3) 闭式求逆（`R_t = Rᵀ`，`t' = -Rᵀt`），numpy/torch 通吃，用于 w2c ↔ c2w；
- `unproject_depth_map_to_point_map`：逐像素 `((u-cx)/fx·d, (v-cy)/fy·d, d)` → `Rᵀ(p_cam - t)` 反投影到世界系。

### 11.3 仓库内脚本

| 脚本 | 用途 |
|---|---|
| `test_inference.py` | 最小冒烟测试 |
| `inference_pipeline.py` | 图像目录 → 点云 PLY + 深度图（npy/png）+ cameras.npz；置信度百分位过滤（默认 20）、可选 `--edge_filter` 深度边缘飞点过滤、随机下采样（默认 1M 点） |
| `inference_pipeline_video.py` | 视频输入（按 fps 抽帧、处理旋转 metadata），复用上面的管线 |
| `demo_gradio.py` | 交互式 Web demo（GLB 可视化），依赖 `requirements_demo.txt` |
| `visual_util.py` | GLB 导出、相机视锥可视化、深度边缘检测、天空分割过滤（demo 专用，import 链含 trimesh/matplotlib） |

---

## 12. 注意事项与常见陷阱汇总

1. **不要外层套 autocast**：模型 forward 内部已做混合精度管理。
2. **`depth_conf` 无界（>1）**：用百分位过滤，不要用绝对阈值。
3. **第一帧定义世界系**：帧序影响输出坐标系；多段推理的结果之间坐标系不互通。
4. **输出尺度非米制**：深度与位姿尺度一致但整体 up-to-scale。
5. **主点固定在图像中心**：内参由 FoV 重建，无法表达偏心主点/skew。
6. **四元数 XYZW（scalar-last）**：与 scipy（xyzw）一致，与 PyTorch3D 默认（wxyz）不同，转换时注意顺序。
7. **宽高比钳制 [0.5, 2.0]**：超宽/超高图会被中心裁剪；全景图等极端比例输入会丢失边缘内容。
8. **混合尺寸批会被 padding（白色 1.0）**：padding 区域同样会产生深度预测，下游需自行裁掉。
9. **text 检查点需 `enable_alignment=True`** 且建议 256 分辨率。
10. **RoPE `normalize_coords` 必须为 "max"**：与检查点训练配置一致，构造时会自动校验并 warning。
11. **headless 脚本不要 import `demo_gradio`/`visual_util`**：会拉入 gradio/trimesh 等 demo 依赖；共享逻辑应放在 `vggt_omega/utils/`。
12. **`numpy<2`** 是硬性依赖约束（requirements.txt / pyproject.toml）。
