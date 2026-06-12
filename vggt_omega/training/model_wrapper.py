"""Model wrapper for GSDPT head training (pure PyTorch, no Lightning).

Loads VGGT-Omega with the GS branch enabled, freezes everything except the
``gs_head``, and exposes ``forward`` + ``compute_loss`` for the training loop.

VGGT-Omega is pose-free: the aggregator does not consume camera tokens, so the
model predicts its own self-consistent cameras and depth. GT cameras from the
dataset are used ONLY for view sampling (in the dataset), never as model input.
Gaussian unprojection and novel-view rendering both use the model's *predicted*
extrinsics/intrinsics, so the geometry stays internally consistent. This is
equivalent to Depth-Anything-3 training with ``camera_dropout_prob=1.0``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from vggt_omega.models.gs_adapter import Gaussians
from vggt_omega.training.config import GSDPTTrainingConfig
from vggt_omega.training.frustum_mask import calculate_in_frustum_mask
from vggt_omega.training.losses import (
    LPIPSLoss,
    compute_ssim,
    depth_offset_loss,
    l1_loss,
    mse_loss,
    offset_xy_loss,
    opacity_entropy_loss,
    render_depth_gradient_l1_loss,
    render_depth_l1_loss,
    sobel_loss,
)
from vggt_omega.training.utils import (
    initialize_gsdpt_output_layer,
    reinitialize_gsdpt_scratch,
)
from vggt_omega.utils.geometry import closed_form_inverse_se3
from vggt_omega.utils.gs_renderer import render_gaussians
from vggt_omega.utils.pose_enc import encoding_to_camera

logger = logging.getLogger(__name__)


class GSModel(nn.Module):
    """Wraps VGGT-Omega for GSDPT head training."""

    def __init__(self, config: GSDPTTrainingConfig):
        super().__init__()
        self.cfg = config
        # context_only implies supervise_context — auto-fix for convenience.
        if self.cfg.loss.context_only and not self.cfg.loss.supervise_context:
            logger.warning("loss.context_only=True implies supervise_context=True; auto-enabling.")
            self.cfg.loss.supervise_context = True
        # Aggregator autocast dtype is driven by config.training.mixed_precision:
        #   bf16/fp16 -> autocast in that dtype; fp32 -> no autocast (full fp32).
        # The trainable gs_head + adapter + rendering always run in fp32.
        _amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._agg_dtype = _amp_map.get(config.training.mixed_precision, torch.bfloat16)
        self._agg_autocast = self._agg_dtype != torch.float32
        self._load_and_prepare_model()
        # Spatial LPIPS only when the target-view mask is enabled (so the
        # per-pixel map can be masked); otherwise scalar LPIPS.
        self.lpips_loss = LPIPSLoss(
            net=config.loss.lpips_net,
            spatial=config.loss.enable_target_loss_mask,
        )

    def _load_and_prepare_model(self):
        """Load VGGT-Omega (GS enabled), freeze all but the gs_head, (re)init head."""
        from inference_pipeline import load_model

        logger.info(f"Loading VGGT-Omega checkpoint: {self.cfg.model.checkpoint_path}")
        self.model = load_model(
            self.cfg.model.checkpoint_path,
            device="cpu",
            enable_gs=True,
            gs_sh_degree=self.cfg.model.gs_sh_degree,
        )

        # ---- Freeze everything, then re-enable only the gs_head ----
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        gs_head = self.model.gs_head
        self.channel_layout = gs_head.channel_layout

        # ---- Initialize / resume the GSDPT head ----
        if self.cfg.model.resume_gsdpt_checkpoint is not None:
            logger.info(f"Loading GSDPT checkpoint: {self.cfg.model.resume_gsdpt_checkpoint}")
            ckpt = torch.load(
                self.cfg.model.resume_gsdpt_checkpoint, map_location="cpu", weights_only=False
            )
            sd = ckpt["gs_head"] if isinstance(ckpt, dict) and "gs_head" in ckpt else ckpt
            gs_head.load_state_dict(sd)
        else:
            init_mode = self.cfg.model.init_mode
            if init_mode == "finetune":
                logger.info("init_mode=finetune: keeping current GSDPT weights")
            elif init_mode == "reset_output":
                logger.info("init_mode=reset_output: resetting output_conv2 last layer")
                initialize_gsdpt_output_layer(gs_head)
            elif init_mode == "scratch":
                logger.info("init_mode=scratch: reinitializing all GSDPT layers from scratch")
                reinitialize_gsdpt_scratch(gs_head)
            else:
                raise ValueError(f"Unknown init_mode: {init_mode}")

        gs_head.train()
        for p in gs_head.parameters():
            p.requires_grad_(True)

        n_trainable = sum(p.numel() for p in gs_head.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {n_trainable:,} / Total: {n_total:,}")

    def trainable_parameters(self):
        """Return only trainable (GSDPT) parameters."""
        return self.model.gs_head.parameters()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @staticmethod
    def _slice_tokens(tokens_list, k):
        """Slice each cached token tensor to the first ``k`` frames (dim=1)."""
        return [None if t is None else t[:, :k].contiguous() for t in tokens_list]

    @staticmethod
    def _w2c_to_c2w(extr):
        """(B, N, 3, 4) w2c -> (B, N, 4, 4) c2w."""
        b, n = extr.shape[:2]
        inv = closed_form_inverse_se3(extr.reshape(b * n, 3, 4))  # (B*N, 4, 4)
        return inv.reshape(b, n, 4, 4)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        images = batch["images"]  # (1, K+M, 3, H, W), [0, 1]
        K = int(batch["num_context"])
        _, num_frames, _, H, W = images.shape
        device_type = images.device.type

        # 1. Frozen forward: aggregator (autocast per config) + camera/depth heads (fp32).
        with torch.no_grad():
            with torch.autocast(
                device_type=device_type, dtype=self._agg_dtype, enabled=self._agg_autocast
            ):
                tokens_list, patch_start = self.model.aggregator(images)
            with torch.autocast(device_type=device_type, enabled=False):
                pose_enc = self.model.camera_head(tokens_list, patch_token_start=patch_start)
                depth, depth_conf = self.model.dense_head(
                    tokens_list, images=images, patch_token_start=patch_start
                )
        depth = depth[..., 0]  # (1, K+M, H, W)
        frozen_depth = depth.detach()
        frozen_depth_conf = depth_conf.detach()

        # Predicted (self-consistent) cameras drive all geometry.
        extrinsic, intrinsic = encoding_to_camera(pose_enc, (H, W))  # w2c (B,N,3,4), K (B,N,3,3)

        # 2. GSDPT head (TRAINABLE) — context views only, fp32.
        ctx_tokens = self._slice_tokens(tokens_list, K)
        ctx_images = images[:, :K]
        with torch.autocast(device_type=device_type, enabled=False):
            raw_gs, gs_opacity = self.model.gs_head(
                ctx_tokens, images=ctx_images, patch_token_start=patch_start
            )

            layout = self.channel_layout
            raw_offset_xy = raw_gs[..., layout["xy_offset"]].clone()  # (B, K, H, W, 2)
            raw_scale_logits = raw_gs[..., layout["scales"]].clone()  # (B, K, H, W, 3)

            ctx_extr = extrinsic[:, :K]
            ctx_intr = intrinsic[:, :K]
            ctx_depth = depth[:, :K]
            # gs_adapter applies depth offset internally; replicate for the loss.
            gs_depth_with_offset = ctx_depth + raw_gs[..., layout["depth_offset"]][..., 0]

            # 3. Camera-space raw GS -> world-space Gaussians (B=1, M=K*H*W).
            gaussians = self.model.gs_adapter(
                extrinsics=ctx_extr,
                intrinsics=ctx_intr,
                depths=ctx_depth,
                opacities=gs_opacity,
                raw_gaussians=raw_gs,
                images=ctx_images,
            )

            # 4. Render to (predicted) target views.
            tgt_extr = extrinsic[0, K:]  # (M, 3, 4) w2c
            tgt_intr = intrinsic[0, K:]  # (M, 3, 3) pixel-space
            rendered_imgs, rendered_depths = render_gaussians(
                gaussians, tgt_extr, tgt_intr, H, W
            )  # (M, 3, H, W), (M, H, W)

            # 5. (Optional) context-view supervision render.
            rendered_context = None
            if self.cfg.loss.supervise_context:
                rendered_context, _ = render_gaussians(
                    gaussians, extrinsic[0, :K], intrinsic[0, :K], H, W
                )

            # 6. (Optional) per-view self-reconstruction: each context view
            # renders ONLY its own gaussians back to itself (diagonal).
            rendered_self = None
            if self.cfg.loss.lambda_self_recon > 0:
                n_per_view = H * W
                self_views = []
                for k in range(K):
                    sl = slice(k * n_per_view, (k + 1) * n_per_view)
                    g_k = Gaussians(
                        means=gaussians.means[:, sl],
                        harmonics=gaussians.harmonics[:, sl],
                        opacities=gaussians.opacities[:, sl],
                        scales=gaussians.scales[:, sl],
                        rotations=gaussians.rotations[:, sl],
                    )
                    col_k, _ = render_gaussians(
                        g_k, extrinsic[0, k : k + 1], intrinsic[0, k : k + 1], H, W
                    )
                    self_views.append(col_k)
                rendered_self = torch.cat(self_views, dim=0)  # (K, 3, H, W)

        # 7. (Optional) target-view valid-visible-pixel mask (from predicted
        # depth/cameras of context vs target views).
        target_loss_mask = None
        if self.cfg.loss.enable_target_loss_mask:
            ctx_c2w = self._w2c_to_c2w(extrinsic[:, :K])  # (B, K, 4, 4)
            tgt_c2w = self._w2c_to_c2w(extrinsic[:, K:])  # (B, M, 4, 4)
            target_loss_mask = calculate_in_frustum_mask(
                frozen_depth[:, K:], intrinsic[:, K:], tgt_c2w,
                frozen_depth[:, :K], intrinsic[:, :K], ctx_c2w,
                depth_atol=self.cfg.loss.target_loss_mask_depth_atol,
                depth_rtol=self.cfg.loss.target_loss_mask_depth_rtol,
            )  # (B, M, H, W) bool

        return {
            "rendered_images": rendered_imgs,
            "rendered_depths": rendered_depths,
            "frozen_depth_context": frozen_depth[:, :K],
            "frozen_depth_target": frozen_depth[:, K:],
            "frozen_depth_conf_context": frozen_depth_conf[:, :K],
            "frozen_depth_conf_target": frozen_depth_conf[:, K:],
            "gs_depth_context": gs_depth_with_offset,
            "raw_offset_xy": raw_offset_xy,
            "raw_scale_logits": raw_scale_logits,
            "opacities": gs_opacity,
            "rendered_context": rendered_context,
            "rendered_self": rendered_self,
            "target_loss_mask": target_loss_mask,
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(
        self, fwd: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        target_gt = batch["target_images"]
        rendered = fwd["rendered_images"]

        if rendered.dim() == 4 and target_gt.dim() == 5:
            target_gt = target_gt.squeeze(0)
        elif rendered.dim() == 5:
            rendered = rendered.squeeze(0)
            target_gt = target_gt.squeeze(0)

        # Optional target-view valid-visible-pixel mask, aligned to rendered
        # (M, 3, H, W) as (M, 1, H, W). None -> all losses use the global path.
        tgt_mask = fwd.get("target_loss_mask")
        if tgt_mask is not None:
            if tgt_mask.dim() == 4:
                tgt_mask = tgt_mask.reshape(-1, tgt_mask.shape[-2], tgt_mask.shape[-1])
            tgt_mask = tgt_mask.unsqueeze(1)  # (M, 1, H, W)
            losses_mask_frac = tgt_mask.float().mean().detach()

        # Photometric loss (MSE or L1)
        if self.cfg.loss.photometric_loss == "l1":
            loss_photo = l1_loss(rendered, target_gt, mask=tgt_mask)
        else:
            loss_photo = mse_loss(rendered, target_gt, mask=tgt_mask)

        # LPIPS needs [0,1] input — clamp here but detach the clamp so gradients
        # flow through the unclamped photometric path instead.
        rendered_clamped = rendered.clamp(0.0, 1.0)
        loss_lpips = self.lpips_loss(rendered_clamped, target_gt, mask=tgt_mask)

        loss_depth = depth_offset_loss(
            fwd["gs_depth_context"], fwd["frozen_depth_context"],
            confidence=fwd.get("frozen_depth_conf_context"),
        )

        # When context_only, target-view photo/lpips/ssim/sobel are still
        # *computed* (so logging keeps showing them and validate() can read
        # `loss_lpips`), but they don't drive backprop.
        include_target = not self.cfg.loss.context_only

        total = self.cfg.loss.lambda_depth * loss_depth
        if include_target:
            total = total + loss_photo + self.cfg.loss.lambda_lpips * loss_lpips

        losses = {
            "loss_photo": loss_photo.detach(),
            "loss_mse": loss_photo.detach(),  # alias for backward compat
            "loss_lpips": loss_lpips.detach(),
            "loss_depth": loss_depth.detach(),
        }
        if tgt_mask is not None:
            losses["target_mask_frac"] = losses_mask_frac

        # --- Conditional new loss terms ---
        if self.cfg.loss.lambda_ssim > 0:
            loss_ssim = 1.0 - compute_ssim(rendered_clamped, target_gt, mask=tgt_mask)
            losses["loss_ssim"] = loss_ssim.detach()
            if include_target:
                total = total + self.cfg.loss.lambda_ssim * loss_ssim

        if self.cfg.loss.lambda_sobel > 0:
            loss_sobel = sobel_loss(rendered, target_gt, mask=tgt_mask)
            losses["loss_sobel"] = loss_sobel.detach()
            if include_target:
                total = total + self.cfg.loss.lambda_sobel * loss_sobel

        if self.cfg.loss.lambda_offset_xy > 0 and fwd.get("raw_offset_xy") is not None:
            loss_xy = offset_xy_loss(fwd["raw_offset_xy"], norm=self.cfg.loss.offset_xy_norm)
            total = total + self.cfg.loss.lambda_offset_xy * loss_xy
            losses["loss_offset_xy"] = loss_xy.detach()

        if self.cfg.loss.lambda_opacity_entropy > 0 and fwd.get("opacities") is not None:
            loss_ent = opacity_entropy_loss(fwd["opacities"])
            total = total + self.cfg.loss.lambda_opacity_entropy * loss_ent
            losses["loss_opacity_entropy"] = loss_ent.detach()

        if self.cfg.loss.lambda_scale > 0 and fwd.get("raw_scale_logits") is not None:
            # Penalize normalized post-sigmoid scale toward 0 → final scale gets
            # pushed toward scale_min. sigmoid(logits) ∈ (0,1) gives a bounded,
            # well-behaved gradient.
            normalized = torch.sigmoid(fwd["raw_scale_logits"])
            if self.cfg.loss.scale_norm == "l1":
                loss_scale = normalized.mean()
            else:  # l2
                loss_scale = (normalized ** 2).mean()
            total = total + self.cfg.loss.lambda_scale * loss_scale
            losses["loss_scale"] = loss_scale.detach()

        # --- Target-view rendered-depth regularization (vs frozen depth) ---
        if (
            (self.cfg.loss.lambda_render_depth_l1 > 0
             or self.cfg.loss.lambda_render_depth_grad_l1 > 0)
            and fwd.get("rendered_depths") is not None
            and fwd.get("frozen_depth_target") is not None
        ):
            rendered_depth = fwd["rendered_depths"]                  # (M, H, W)
            depth_target = fwd["frozen_depth_target"]                # (B, M, H, W)
            depth_flat = depth_target.reshape(
                -1, depth_target.shape[-2], depth_target.shape[-1]
            )
            depth_conf_flat = None
            if fwd.get("frozen_depth_conf_target") is not None:
                conf_target = fwd["frozen_depth_conf_target"]        # (B, M, H, W)
                depth_conf_flat = conf_target.reshape(
                    -1, conf_target.shape[-2], conf_target.shape[-1]
                )
            valid_mask = rendered_depth > 0

            if self.cfg.loss.lambda_render_depth_l1 > 0:
                loss_render_depth_l1 = render_depth_l1_loss(
                    rendered_depth, depth_flat, mask=valid_mask,
                    confidence=depth_conf_flat,
                )
                if include_target:
                    total = total + self.cfg.loss.lambda_render_depth_l1 * loss_render_depth_l1
                losses["loss_render_depth_l1"] = loss_render_depth_l1.detach()

            if self.cfg.loss.lambda_render_depth_grad_l1 > 0:
                loss_render_depth_grad_l1 = render_depth_gradient_l1_loss(
                    rendered_depth, depth_flat, mask=valid_mask,
                    confidence=depth_conf_flat,
                )
                if include_target:
                    total = total + (
                        self.cfg.loss.lambda_render_depth_grad_l1 * loss_render_depth_grad_l1
                    )
                losses["loss_render_depth_grad_l1"] = loss_render_depth_grad_l1.detach()

        # --- Context-view supervision (optional, off by default) ---
        if (
            self.cfg.loss.supervise_context
            and fwd.get("rendered_context") is not None
            and "context_images" in batch
        ):
            rendered_ctx = fwd["rendered_context"]
            ctx_gt = batch["context_images"]
            if rendered_ctx.dim() == 4 and ctx_gt.dim() == 5:
                ctx_gt = ctx_gt.squeeze(0)
            elif rendered_ctx.dim() == 5:
                rendered_ctx = rendered_ctx.squeeze(0)
                ctx_gt = ctx_gt.squeeze(0)

            if self.cfg.loss.photometric_loss == "l1":
                loss_ctx_photo = l1_loss(rendered_ctx, ctx_gt)
            else:
                loss_ctx_photo = mse_loss(rendered_ctx, ctx_gt)

            rendered_ctx_clamped = rendered_ctx.clamp(0.0, 1.0)
            loss_ctx_lpips = self.lpips_loss(rendered_ctx_clamped, ctx_gt)

            loss_ctx = loss_ctx_photo + self.cfg.loss.lambda_lpips * loss_ctx_lpips
            if self.cfg.loss.lambda_ssim > 0:
                loss_ctx_ssim = 1.0 - compute_ssim(rendered_ctx_clamped, ctx_gt)
                loss_ctx = loss_ctx + self.cfg.loss.lambda_ssim * loss_ctx_ssim
                losses["loss_context_ssim"] = loss_ctx_ssim.detach()

            total = total + self.cfg.loss.lambda_context * loss_ctx
            losses["loss_context"] = loss_ctx.detach()
            losses["loss_context_photo"] = loss_ctx_photo.detach()
            losses["loss_context_lpips"] = loss_ctx_lpips.detach()

        # --- Per-view self-reconstruction (optional, off by default) ---
        if (
            self.cfg.loss.lambda_self_recon > 0
            and fwd.get("rendered_self") is not None
            and "context_images" in batch
        ):
            rendered_self = fwd["rendered_self"]
            self_gt = batch["context_images"]
            if rendered_self.dim() == 4 and self_gt.dim() == 5:
                self_gt = self_gt.squeeze(0)
            elif rendered_self.dim() == 5:
                rendered_self = rendered_self.squeeze(0)
                self_gt = self_gt.squeeze(0)

            if self.cfg.loss.photometric_loss == "l1":
                loss_self_photo = l1_loss(rendered_self, self_gt)
            else:
                loss_self_photo = mse_loss(rendered_self, self_gt)

            rendered_self_clamped = rendered_self.clamp(0.0, 1.0)
            loss_self_lpips = self.lpips_loss(rendered_self_clamped, self_gt)

            loss_self = loss_self_photo + self.cfg.loss.lambda_lpips * loss_self_lpips
            if self.cfg.loss.lambda_ssim > 0:
                loss_self_ssim = 1.0 - compute_ssim(rendered_self_clamped, self_gt)
                loss_self = loss_self + self.cfg.loss.lambda_ssim * loss_self_ssim
                losses["loss_self_recon_ssim"] = loss_self_ssim.detach()

            total = total + self.cfg.loss.lambda_self_recon * loss_self
            losses["loss_self_recon"] = loss_self.detach()
            losses["loss_self_recon_photo"] = loss_self_photo.detach()
            losses["loss_self_recon_lpips"] = loss_self_lpips.detach()

        losses["loss"] = total
        return losses
