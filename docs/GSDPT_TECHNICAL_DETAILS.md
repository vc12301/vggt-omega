# VGGT-Omega 3DGS 分支（GSDPT）技术说明

本文档详细说明 VGGT-Omega 中新增的 3D Gaussian Splatting（3DGS）分支的全部技术细节，
包括 GSDPT 预测头、高斯参数适配器、渲染、PLY 导出、模型集成与测试脚本。设计参考自
Depth-Anything-3（da3）的同类模块，并结合 VGGT-Omega 自身的 DenseHead 结构。

> 适用范围：`VGGTOmega(enable_gs=True, gs_sh_degree=0..2)`。该分支在公开 checkpoint 中
> **没有训练权重**，依靠精心设计的初始化即可渲染出基本合理的场景，主要用于验证几何约定与
> 后续训练的脚手架。

---

## 0. 模块清单与职责

| 文件 | 模块 | 职责 |
|---|---|---|
| `vggt_omega/models/heads/gsdpt_head.py` | `GSDPTHead`、`gs_channel_layout` | 从聚合 token 预测每像素**原始**高斯参数（相机系，未激活）|
| `vggt_omega/models/gs_adapter.py` | `GaussianAdapter`、`Gaussians`、`rotate_sh` | 将原始参数转换为**世界系**标准高斯（激活 + 几何变换）|
| `vggt_omega/utils/gs_renderer.py` | `render_gaussians`、`interpolate_camera_path` | gsplat 渲染、相机轨迹插值 |
| `vggt_omega/utils/gs_ply.py` | `export_gaussians_ply`、`inverse_sigmoid` | 导出标准 3DGS PLY |
| `vggt_omega/models/vggt_omega.py` | `VGGTOmega`（`enable_gs`）| 在 forward 中串联 gs_head → adapter |
| `inference_pipeline.py` | `load_model`（`enable_gs`）| 以 `strict=False` 加载，保留 GS 头初始化 |
| `test_gs_inference.py` | 测试脚本 | 输出高斯 PLY、原视角对比图、插值轨迹视频 |

**数据流总览**：

```
images (B,N,3,H,W) ──► Aggregator ──► aggregated_tokens_list (24 层, 缓存 [4,11,17,23])
                                            │
                          ┌─────────────────┼──────────────────┐
                      CameraHead         DenseHead           GSDPTHead
                          │                 │                    │
                     pose_enc(9)     depth, depth_conf   raw_gs(d_in)+opacity
                          │                 │                    │
                  encoding_to_camera   depth[...,0]              │
                  extrinsic/intrinsic ─────┴──────────► GaussianAdapter ──► Gaussians
                                                                                │
                                                  ┌─────────────────────────────┤
                                          render_gaussians (gsplat)    export_gaussians_ply
                                                  │                             │
                                          colors/depths                    .ply 文件
```

---

## 1. 原始高斯参数通道布局

`gs_channel_layout(sh_degree)`（`gsdpt_head.py`）定义了 GSDPTHead 输出的通道排列，
顺序与 da3 GaussianAdapter 的剥离顺序一致：

| 通道切片 | 含义 | 维度 |
|---|---|---|
| `[0:2]` | `xy_offset` 像素平面亚像素偏移 | 2 |
| `[2:5]` | `scales` 各向异性尺度（logit）| 3 |
| `[5:9]` | `quaternion` 旋转，**XYZW 标量在后** | 4 |
| `[9 : 9+3·d_sh]` | `sh` 残差球谐系数（3 通道 × d_sh）| 3·d_sh |
| `[9+3·d_sh : 10+3·d_sh]` | `depth_offset` 深度偏移 | 1 |
| 最后 1 通道（不在 layout 中）| `opacity`（在头内 sigmoid 激活）| 1 |

其中 `d_sh = (sh_degree+1)²`。GSDPTHead 的 `output_dim = (9+3·d_sh+1) + 1 = d_in + 1`，
`d_in` 即 GaussianAdapter 接收的原始参数维度。

| sh_degree | d_sh | d_in | output_dim |
|---|---|---|---|
| 0 | 1 | 13 | 14 |
| 1 | 4 | 22 | 23 |
| 2 | 9 | 37 | 38 |

> **关键约定**：除 opacity 外，GSDPTHead **不对任何通道激活**（da3 中称 `activation="linear"`），
> 全部激活与几何变换都在 GaussianAdapter 中完成。这样可以把"网络预测"和"几何约定"清晰解耦。

---

## 2. GSDPTHead — 高斯参数预测头

### 2.1 与 DenseHead 的关系

GSDPTHead 复用 DenseHead 的 DPT 主干积木（直接 `from .dense_head import ...`）：
`_make_scratch`、`_make_fusion_block`、`_make_dense_resize_layer`、`custom_interpolate`，
以及 `_apply_pos_embed`/`scratch_forward` 的实现逻辑。两者**唯一的结构差异在输出段**：

- **DenseHead**：1/4 尺度融合特征 → 两个 `proj`/`proj_conf`（1×1 conv 到 `(patch_size/4)²` 通道）
  → `F.pixel_shuffle(4)` 上采样回全分辨率。
- **GSDPTHead**：1/4 尺度融合特征 → `output_conv1`（256→128）→ `custom_interpolate` 双线性
  上采样到 **(H,W) 全分辨率** → 在全分辨率上 **注入 RGB 图像特征**（`images_merger`）
  → `output_conv2`（128→32→output_dim）。

GS 头采用全分辨率注入而非 pixel_shuffle，是因为高斯颜色/几何需要原始图像的高频细节
（da3 GSDPT 的核心设计），而 pixel_shuffle 是从低分辨率特征"搬运"通道，无法引入新的高频信息。

### 2.2 前向流程（`_forward_impl`）

输入：`aggregated_tokens_list`（24 层，list of (B,N,T,2048) 或 None）、`images` (B,N,3,H,W)、
`patch_token_start`（=17）、`frames_chunk_size`（默认 8）。

```
对 4 个缓存层 [4,11,17,23]：
    x = tokens[layer][:, :, patch_token_start:]    # 只取 patch token，丢弃 camera/register
    x = x.float()                                  # 强制 fp32（见 §10 精度）
    x = LayerNorm(x)
    x = reshape 到 (B·N, 2048, patch_h, patch_w)
    x = projects[i](x)                             # 1×1 conv → [256,512,1024,1024]
    x = _apply_pos_embed(x)                         # UV-grid sincos 位置编码 ×0.1
    x = resize_layers[i](x)                         # ×4 / ×2 / ×1 / ×0.5
                                                    #   → 统一到 1/4 patch 尺度
fused = scratch_forward(...)                        # refinenet4→3→2→1 自顶向下融合
fused = _apply_pos_embed(fused)                     # 1/4 尺度再加一次位置编码
fused = output_conv1(fused)                         # 256 → 128, 3×3
fused = custom_interpolate(fused, (H, W))           # 双线性上采样到全分辨率
fused = fused + images_merger(images)               # 全分辨率 RGB 特征注入
fused = _apply_pos_embed(fused)                     # 全分辨率位置编码
logits = output_conv2(fused)                        # 128 → 32 → output_dim
logits = logits.permute(0,2,3,1)                    # (B·N, H, W, output_dim)

raw_gs    = logits[..., :-1]                         # (B,N,H,W,d_in)
gs_opacity = sigmoid(logits[..., -1])                # (B,N,H,W) ∈ (0,1)
```

`frames_chunk_size` 按帧分块以省显存（与 DenseHead 一致）；token 在每个 chunk 内重新切片。
输出强制要求 fp32，否则抛 `TypeError`。

### 2.3 `images_merger`

```python
nn.Sequential(
    nn.Conv2d(3, 16, 3, 1, 1), nn.GELU(),     # merger_dim//4 = 128//4 = 32... 见下
    nn.Conv2d(16, 32, 3, 1, 1), nn.GELU(),
    nn.Conv2d(32, 64, 3, 1, 1), nn.GELU(),    # 实际通道见代码：merger_dim = features//2 = 128
)
```

实际通道为 `3 → merger_dim//4 → merger_dim//2 → merger_dim`，其中 `merger_dim = features//2 = 128`。
输入是 [0,1] RGB（与 Aggregator 一样使用原始像素，不做归一化），在全分辨率上与 DPT 特征逐元素相加。

### 2.4 参数量（实测）

| sh_degree | GSDPTHead 总参数 |
|---|---|
| 0 | 32.748 M |
| 1 / 2 | 32.749 M |

输出层维度变化对总量影响极小。各子模块拆分（sh_degree=0）：

| 子模块 | 参数 |
|---|---|
| `projects`（4×1×1 conv）| 5.770 M |
| `resize_layers`（转置卷积/卷积）| 11.536 M |
| `scratch`（refinenet 融合）| 15.012 M |
| `output_conv1` | 0.295 M |
| `images_merger` | 0.093 M |
| `output_conv2` | 0.037 M |
| `norm` | 0.004 M |

GaussianAdapter **无可学习参数**（仅一个非持久化 `sh_mask` buffer）。

---

## 3. 输出层初始化（`_init_gs_prediction_head`）

这是让**未训练**网络也能渲染合理场景的关键。初始化作用于 `output_conv2` 的最后一层
1×1 conv：**权重全部置 0**，于是该层的输出恒等于其 bias，与前面所有层、输入 token 无关。
bias 全部设置在 **logit / 原始空间**（即激活函数的输入侧）。两个可配置参数控制初始外观：
`init_pixel_size`（默认 0.25，初始投影像素尺寸）与 `init_opacity`（默认 0.12，初始不透明度）。

| 通道 | 含义 | bias 值 | 经 adapter 激活后的效果 |
|---|---|---|---|
| `xy_offset` | 像素偏移 | 0 | 高斯落在像素中心，无偏移 |
| `scales` | 尺度 logit | `logit((s_t−min)/(max−min)) ≈ −3.135` | 投影约 0.25 像素（见推导）|
| `quaternion` | 旋转 XYZW | `(0,0,0,1)` | 单位四元数（adapter 会归一化）|
| `sh` | 残差球谐 | 0 | 颜色 = 像素 RGB（无残差）|
| `depth_offset` | 深度偏移 | 0 | 深度不变 |
| `opacity` | 不透明度 logit | `logit(0.12) ≈ −1.992` | sigmoid → 0.12 |

### 3.1 scale 初值推导

adapter 中世界尺度 = `act(raw) · depth · m`，其中投影补偿因子
`m = scale_multiplier · (1/fx + 1/fy)`（fx,fy 为像素焦距，见 §5）。
高斯投影到屏幕上的像素尺寸 ≈ `世界尺度 · fx / depth`（沿 x），代入得：

```
投影像素尺寸 ≈ act(raw) · depth · scale_multiplier·(1/fx+1/fy) · fx / depth
            = act(raw) · scale_multiplier · (1 + fx/fy)
            ≈ act(raw) · 2 · scale_multiplier      （方像素 fx≈fy）
```

要让投影约为 `init_pixel_size = 0.25` 像素，需
`act(raw) = init_pixel_size / (2·scale_multiplier) = 0.25 / 0.2 = 1.25`
（默认 `scale_multiplier = 0.1`）。而 `act(raw) = min + (max−min)·sigmoid(raw)`，
其中 `min=1e-5, max=30`，故：

```
p = (1.25 − 1e-5) / (30 − 1e-5) ≈ 0.0417
bias = logit(p) = ln(p/(1−p)) ≈ −3.135
```

opacity 在头内 sigmoid 激活，bias 取其 logit：`logit(0.12) = ln(0.12/0.88) ≈ −1.992`。

scale 只与场景尺度有关，**与 depth 和 fov 无关**——这正是 da3 设计中"预测场景尺度量、
由 adapter 消除投影偏差"的含义。

---

## 4. GaussianAdapter — 原始参数 → 世界系高斯

### 4.1 接口

```python
GaussianAdapter(sh_degree=0, gaussian_scale_min=1e-5,
                gaussian_scale_max=30.0, scale_multiplier=0.1)
```

`forward` 输入：
- `extrinsics` (B,N,3,4) 或 (B,N,4,4)，camera-from-world（w2c，OpenCV 约定）
- `intrinsics` (B,N,3,3)，**像素单位**，主点居中
- `depths` (B,N,H,W)
- `opacities` (B,N,H,W)，已是 [0,1]
- `raw_gaussians` (B,N,H,W,d_in)
- `images` (B,N,3,H,W)，[0,1] RGB（用于残差 SH 基底）

输出 `Gaussians` dataclass（M = N·H·W）：

```python
@dataclass
class Gaussians:
    means:     Tensor  # (B, M, 3)
    harmonics: Tensor  # (B, M, 3, d_sh)
    opacities: Tensor  # (B, M)，∈ [0,1]
    scales:    Tensor  # (B, M, 3)，线性（非 log）
    rotations: Tensor  # (B, M, 4)，单位四元数，WXYZ 标量在前
```

`d_sh = (sh_degree+1)²`，`d_in = 9 + 3·d_sh + 1`。adapter 内部全程 fp32。

### 4.2 逐步骤说明

**Step 1 — c2w 反演**（直接 torch 实现，不依赖外部工具）：

```
R_w2c = extrinsics[...,:3,:3]
t_w2c = extrinsics[...,:3,3]
R_c2w = R_w2cᵀ
t_c2w = −R_c2w · t_w2c
```

**Step 2 — 高斯中心（means）**：像素中心网格 `(j+0.5, i+0.5)` 加上预测偏移，
按 z-depth 反投影到相机系，再变换到世界系。

```
u = (j + 0.5) + xy_offset_x
v = (i + 0.5) + xy_offset_y
z = depth + depth_offset
p_cam = [(u − cx)/fx · z, (v − cy)/fy · z, z]
means = R_c2w · p_cam + t_c2w
```

这等价于 da3 用"未归一化方向向量 × z-depth"的反投影（`get_world_rays_zdepth`），
避免了用归一化方向导致的球面畸变。

**Step 3 — 尺度（scales）**：

```
s = min + (max − min) · sigmoid(raw_scales)        # ∈ (1e-5, 30)
m = scale_multiplier · (1/fx + 1/fy)               # 投影补偿因子
gs_scales = s · z · m                               # 各向异性 (3,)，世界单位
```

**Step 4 — 旋转（rotations）**：相机系 XYZW 四元数 → 归一化 → 旋转矩阵
→ 左乘 R_c2w 转到世界系 → 回到四元数 → 重排为 WXYZ。

```
q_cam = normalize(raw_quat)                         # XYZW
R_cam = quat_to_mat(q_cam)                          # vggt_omega/utils/rotation.py
R_world = R_c2w · R_cam
q_world_xyzw = mat_to_quat(R_world)                 # XYZW
q_world_wxyz = q_world_xyzw[..., [3,0,1,2]]          # → WXYZ（gsplat/PLY 约定）
```

> **不照搬 da3**：da3 的 `cam_quat_xyzw_to_world_quat_wxyz` 存在约定混用（把 wxyz 当作
> xyzw 喂给 `quat_to_mat`）。本实现使用 `vggt_omega/utils/rotation.py` 的 XYZW 约定
> 正确实现，最终统一输出 WXYZ。

**Step 5 — 残差球谐（harmonics）**：见 §6。

**Step 6 — opacity**：直接展平透传（已在头内 sigmoid）。

最后所有量按 `(B, N·H·W, ...)` 展平。

---

## 5. 透视对齐缩放原理

3DGS 渲染中，一个世界尺度为 `S` 的高斯，投影到屏幕的像素尺寸正比于 `S · f / z`
（f 焦距、z 深度）。如果让网络直接预测世界尺度 `S`，那么近处与远处、广角与长焦下，
同样的"物体表面小块"对应的 `S` 会差很多，网络难以学习。

da3 的做法（本实现沿用）是让网络预测一个**只与场景尺度相关、与 depth/fov 无关**的量
`s = act(raw)`，再由 adapter 乘上 `z · m` 还原世界尺度：

```
gs_scale = s · z · m,   m = scale_multiplier · (1/fx + 1/fy)
```

投影回屏幕：`gs_scale · f / z = s · m · f ≈ s · scale_multiplier · (1 + fx/fy) ≈ 2·s·scale_multiplier`，
**与 z、f 都无关**。于是网络只需预测"这个高斯在屏幕上覆盖几个像素"这一尺度无关量，
adapter 自动消除透视投影偏差。这也是 §3.1 中 scale 初值推导的依据。

---

## 6. 残差球谐（Residual SH）设计

这是本实现相对 da3 的主要改动。da3 直接预测 SH 系数；本实现预测**残差 SH**，
叠加在"像素 RGB 转 0 阶 SH"的基底上：

```
SH_C0 = 0.28209479177387814          # = 1/(2√π)，0 阶球谐基函数常数
base_dc = (rgb − 0.5) / SH_C0         # 像素 RGB → DC 系数（3DGS 标准转换）
sh = raw_sh.view(...,3,d_sh) · sh_mask
sh[..., 0] += base_dc                 # DC 通道叠加像素颜色基底
```

含义：3DGS 中渲染颜色（仅 DC）≈ `SH_C0 · f_dc + 0.5`。把 `(rgb−0.5)/SH_C0` 作为 DC 基底，
则在残差为 0 时渲染颜色恰好等于像素原始 RGB。网络只需学习相对像素颜色的**残差与视角相关项**，
训练初期颜色即正确，收敛更稳定。

### 6.1 `sh_mask`

非持久化 buffer，抑制高阶系数初始幅度，使颜色初期由 DC 主导：

```
sh_mask[0] = 1.0                              # DC（0 阶）
sh_mask[degree²:(degree+1)²] = 0.1 · 0.25^degree   # degree=1: 0.025, degree=2: 0.00625
```

### 6.2 SH 旋转（`rotate_sh`，仅 sh_degree>0）

当 `sh_degree > 0` 时，1 阶及以上的 SH 系数是视角相关的，必须随相机姿态旋转到世界系
（0 阶 DC 旋转不变，无需处理）。实现移植自 da3 `sh_helpers.rotate_sh`，依赖 `e3nn`：

```
1. 坐标轴置换 yzx → xyz（gsplat 与 e3nn 的轴序差异）：
   permuted_R = Pᵀ · R_c2w · P,   P = [[0,0,1],[1,0,0],[0,1,0]]
2. _project_to_so3(permuted_R)：SVD 投影到严格 SO(3)（e3nn 要求 det==1）
3. matrix_to_angles → 欧拉角 (α,β,γ)
4. 对每个 degree：wigner_D(degree, α, −β, γ) 作用于对应系数块
```

`rotate_sh` 全程在 `autocast(enabled=False)` 的 fp32 下计算。`_project_to_so3` 用 SVD
处理反射（det<0）并强制 det 严格为 1。

---

## 7. 坐标与四元数约定汇总

| 量 | 约定 |
|---|---|
| extrinsic | camera-from-world（w2c），OpenCV（x 右、y 下、z 前）|
| intrinsic | 像素单位，主点固定在图像中心 |
| 深度 | z-depth（沿相机 z 轴），非欧氏距离 |
| 仓库内部四元数 | **XYZW**（标量在后），见 `vggt_omega/utils/rotation.py`、`pose_enc.py` |
| `Gaussians.rotations` 输出 | **WXYZ**（标量在前）——gsplat 与 3DGS PLY 的约定 |
| 尺度输出 | 线性（非 log）；PLY 导出时才取 log |
| opacity 输出 | [0,1] 线性；PLY 导出时才取 logit |

> **易错点**：整个仓库内部都用 XYZW，唯独 GS 的最终输出（`Gaussians.rotations`、PLY 的
> `rot_*`）用 WXYZ。转换发生在 GaussianAdapter Step 4 末尾。

---

## 8. 渲染（`gs_renderer.py`）

### 8.1 `render_gaussians`

懒导入 `gsplat`（核心包不依赖 gsplat，符合仓库 import 边界）。要求 batch size = 1。

```python
render_gaussians(gaussians, extrinsics (V,3,4|4,4), intrinsics (V,3,3),
                 height, width, background_color=(0,0,0),
                 chunk_size=8, render_mode="RGB+D")
→ (colors (V,3,H,W), depths (V,H,W))
```

内部：
- extrinsic 补成 (V,4,4) 作为 `viewmats`（w2c）；intrinsic 像素单位**直接**作 `Ks`
  （无需 da3 的 fov 往返换算，因为我们的内参本就是像素单位、主点居中）。
- `harmonics` (M,3,d_sh) → permute 为 `colors` (M,d_sh,3)；`sh_degree` 由 d_sh 反推
  （`isqrt(d_sh)−1`）。
- 调用 `gsplat.rasterization(means, quats=WXYZ, scales=线性, opacities, colors=SH,
  viewmats=w2c, Ks=像素, render_mode, sh_degree, packed=False)`。
- 按 `chunk_size` 分视角块渲染后拼接。`render_mode="RGB+D"` 时输出最后一通道为深度。

### 8.2 `interpolate_camera_path`

为渲染平滑的环视/穿越视频生成相机轨迹。相邻输入相机两两插值，每对 `steps_per_pair`（默认 8）帧：

```
在 c2w 空间：
  平移：线性插值（lerp）
  旋转：四元数球面插值（slerp）
  内参：线性插值
缓动：t' = (cos(π(t+1)) + 1)/2   # cosine ease-in-out，避免端点处突变
```

输出 `V' = (V−1)·steps_per_pair + 1` 个 w2c extrinsic 与对应 intrinsic，端点严格对齐输入相机。
`_slerp` 处理了点积为负（取反走短弧）与近平行（退化为归一化 lerp）两种情况。
比 da3 的 pivot 点插值简单，足以满足验证需求。

---

## 9. PLY 导出（`gs_ply.py`）

`export_gaussians_ply(gaussians, ply_path, save_sh_dc_only=True)` 写出标准 3DGS PLY
（兼容 graphdeco-inria/gaussian-splatting 等查看器）。要求 batch size = 1。

字段顺序（全部 f4，二进制）：

```
x, y, z,                          # 世界坐标
nx, ny, nz,                       # 法线占位（全 0）
f_dc_0, f_dc_1, f_dc_2,           # SH DC 分量（3 通道）
[f_rest_0 ... f_rest_{3(d_sh−1)−1}]   # 仅当 save_sh_dc_only=False 且 sh_degree>0
opacity,                          # = inverse_sigmoid(opacity)   ← logit 空间
scale_0, scale_1, scale_2,        # = log(scale)                 ← log 空间
rot_0, rot_1, rot_2, rot_3        # WXYZ 单位四元数
```

**存储空间约定**（3DGS 查看器加载时会反向应用）：
- opacity 存 **logit**（`inverse_sigmoid`，带 [1e-6, 1−1e-6] 截断）；
- scales 存 **log**（带 `min=1e-10` 截断防 log(0)）；
- rotation 存 WXYZ。

`save_sh_dc_only=True`（默认，sh_degree=0 时）只写 DC，得到 17 个字段；
`save_sh_dc_only=False`（sh_degree=2 时）额外写 `3·(d_sh−1)=24` 个 `f_rest_*`，共 41 个字段。

---

## 10. 模型集成与精度

### 10.1 `VGGTOmega(enable_gs=...)`

```python
VGGTOmega(enable_gs=True, gs_sh_degree=0)
```

- 构造 `self.gs_adapter = GaussianAdapter(sh_degree=gs_sh_degree)` 与
  `self.gs_head = GSDPTHead(dim_in=2·embed_dim, patch_size, sh_degree)`；
  断言 `gs_head.output_dim == gs_adapter.d_in + 1`。
- 要求同时启用 camera 与 depth 头（否则抛错）——adapter 需要位姿与深度。

forward 中（在 `autocast(enabled=False)` 的 fp32 段内，depth 之后）：

```python
raw_gs, gs_opacity = self.gs_head(aggregated_tokens_list, images, patch_token_start)
predictions["raw_gs"]      = raw_gs
predictions["gs_opacity"]  = gs_opacity
extrinsic, intrinsic = encoding_to_camera(pose_enc, images.shape[-2:])
predictions["gaussians"]   = self.gs_adapter(
    extrinsics=extrinsic, intrinsics=intrinsic,
    depths=predictions["depth"][...,0], opacities=gs_opacity,
    raw_gaussians=raw_gs, images=images)
```

### 10.2 精度

- Aggregator 在 bf16/fp16 autocast 下运行；**所有头（含 GSDPTHead）与 adapter 在
  fp32 下运行**（forward 内部自带 autocast 管理）。
- GSDPTHead 内部把 token 强制 `.float()`，输出强制要求 fp32。
- `rotate_sh` 内部再次 `autocast(enabled=False)` 保证 e3nn 计算的 fp32。
- **不要**在外部再套 autocast，只用 `torch.inference_mode()`。

### 10.3 加载（`load_model(..., enable_gs=True)`）

公开 checkpoint 不含 GS 权重，因此 `enable_gs=True` 时用 `load_state_dict(strict=False)`，
并校验所有 missing key 都以 `gs_head.`/`gs_adapter.` 开头、无 unexpected key，
其余照常严格加载。这样 GS 头保留 §3 的初始化，主干仍精确加载预训练权重。

---

## 11. 测试脚本（`test_gs_inference.py`）

```bash
python test_gs_inference.py --image_dir path/to/images \
    [--sh_degree 0] [--steps_per_pair 8] [--fps 24] \
    [--conf_percentile 0] [--edge_filter] [--edge_rtol 0.03] [--edge_kernel_size 3] \
    [--max_frames N] [--resolution 512]
```

流程：
1. `load_model(enable_gs=True)` → forward；
2. `print_sanity_stats`：打印 raw 头输出统计，未训练时应等于初始化值
   （opacity 0.12、xy/depth offset 0、residual_sh 0、quat (0,0,0,1)、激活后 scale 1.25）；
3. 高斯过滤（构建逐高斯 keep mask 后一次性切片，两者可组合，默认都不过滤）：
   - `--conf_percentile`：按 `depth_conf` 百分位掩码（`conf_keep_mask`）；
   - `--edge_filter`：复用 `inference_pipeline.depth_edge` 标记深度不连续（飞边）像素，
     移除对应高斯（`edge_keep_mask`，参数 `--edge_rtol`/`--edge_kernel_size`）。
     可显著减少新视角下物体边缘的拉伸/拖尾高斯；
4. `export_gaussians_ply` 写 `gaussians.ply`；
5. 原视角渲染，与输入图左右拼接存 `compare/{i:03d}.png`（直接目视初始化是否合理）；
6. `interpolate_camera_path` 生成轨迹 → 渲染 → imageio 写 `render.mp4`（libx264, yuv420p）。

输出目录结构：

```
{output_dir}/
    gaussians.ply
    compare/{i:03d}.png
    render.mp4
```

---

## 12. 依赖

`requirements_gs.txt`（仅 GS 分支需要）：

| 依赖 | 用途 | 必需性 |
|---|---|---|
| `gsplat` | 高斯光栅化渲染 | 渲染时必需（懒导入）|
| `e3nn` | SH 系数旋转（wigner_D）| **仅 sh_degree>0** 时必需 |
| `imageio` + `imageio-ffmpeg` | 写 mp4 | 测试脚本必需 |
| `plyfile` | PLY 读写（核心 requirements 已含）| 导出必需 |

核心包 `vggt_omega/` 仍只依赖 torch/numpy/PIL/einops/cv2；gsplat、e3nn 都在调用处懒导入，
不污染 import 边界。

---

## 13. 验证结果（未训练初始化）

在 `snow_lift` 7 帧（实际聚合 3 帧、384×688）上：

| 项目 | 结果 |
|---|---|
| sanity stats | opacity 0.1200、offset/residual 全 0、quat (0,0,0,1)、scale 1.2500，与初始化精确一致 |
| 高斯数 | 1,849,344（= 3·384·688 附近）|
| PLY | 17 字段（sh_degree=0），opacity 列 ≈ logit(0.12) = -1.99，scale 列 = log(线性尺度) 有限，‖quat‖=1 |
| 原视角渲染 vs 输入 | PSNR ≈ 22 dB，视觉几乎一致 |
| 轨迹端点帧 vs 原视角渲染 | ≈ 37 dB（端点对齐正确）|
| sh_degree=2 | e3nn rotate_sh 路径正常，41 字段、24 个 f_rest（残差初始全 0）|

> 新视角（偏离输入位姿处）会出现拉伸条纹，这是各向同性亚像素高斯（约 0.25 px）在**未训练、无致密化**下
> 从单一深度面外推的预期表现，训练后会改善。初始化使原视角重建基本正确，证明几何约定无误。

---

## 14. 关键注意事项（Gotchas）

1. **四元数双约定**：仓库内部 XYZW，GS 最终输出（`Gaussians.rotations`、PLY `rot_*`）WXYZ。
   转换在 GaussianAdapter Step 4。
2. **残差 SH，非绝对 SH**：DC = 像素 RGB 基底 `(rgb−0.5)/SH_C0` + 预测残差。残差为 0 时
   渲染颜色 = 原像素。
3. **尺度是场景尺度量**：网络预测与 depth/fov 无关的尺度，adapter 乘 `z·m` 消除投影偏差。
4. **初始化在 logit/原始空间**：最终 1×1 conv 零权重，bias 设在激活函数输入侧。
5. **不要外层 autocast**：头与 adapter 必须 fp32，forward 内部已管理；只用 `inference_mode`。
6. **公开 checkpoint 无 GS 权重**：`enable_gs=True` 时 `strict=False` 加载，保留头初始化。
7. **e3nn 仅 sh_degree>0 必需**：sh_degree=0（默认）不触发 SH 旋转，无需 e3nn。
8. **PLY 存储空间**：opacity 存 logit、scales 存 log、rot 存 WXYZ——查看器加载时反向应用。
9. **gsplat/e3nn 懒导入**：核心包不依赖它们，缺失时仅在调用渲染/SH 旋转时报错。
10. **depth_conf 仍是无界正值**：`--conf_percentile` 过滤高斯时用百分位，不能用绝对阈值。
