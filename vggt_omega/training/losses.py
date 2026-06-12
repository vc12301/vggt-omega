"""Loss functions for GSDPT training (ported verbatim from Depth-Anything-3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss wrapper (VGG backbone).

    When ``spatial=True`` the underlying LPIPS network returns a per-pixel map
    instead of a scalar, which enables ``forward(..., mask=...)`` to compute a
    masked mean over valid pixels (see GSDPT target-view mask). With the default
    ``spatial=False`` the behavior is unchanged (scalar LPIPS, [-1, 1] input).
    """

    def __init__(self, net: str = "vgg", spatial: bool = False):
        super().__init__()
        import lpips

        self.spatial = spatial
        self.lpips_fn = lpips.LPIPS(net=net, spatial=spatial)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute LPIPS loss.

        Args:
            pred: (B, 3, H, W) in [0, 1]
            target: (B, 3, H, W) in [0, 1]
            mask: optional (B, 1, H, W) or (B, H, W) — only valid in spatial mode;
                when given, returns the mask-weighted mean of the per-pixel LPIPS map.

        Note:
            This forward is deliberately NOT wrapped in ``torch.no_grad()`` so
            that gradients flow back into ``pred`` (the rendered image). The
            LPIPS/VGG network parameters are frozen in ``__init__`` via
            ``requires_grad_(False)``, so they are never updated regardless.
        """
        if self.spatial:
            # spatial LPIPS takes [0,1] inputs with normalize=True and returns
            # a per-pixel map (B, 1, H, W).
            lp = self.lpips_fn(pred, target, normalize=True)
            if mask is None:
                return lp.mean()
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, H, W)
            mask = mask.to(lp.dtype)
            denom = mask.sum().clamp_min(1.0)
            return (lp * mask).sum() / denom

        # LPIPS expects [-1, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        return self.lpips_fn(pred_scaled, target_scaled).mean()


def mse_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-pixel MSE loss. With ``mask`` (broadcastable to ``pred``), masked mean."""
    if mask is None:
        return F.mse_loss(pred, target)
    return _masked_mean((pred - target) ** 2, pred, mask)


def depth_offset_loss(
    pred_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depth regularization loss: penalizes the offset between predicted and teacher depth.

    pred_depth = teacher_depth + offset (from GSDPT).
    This loss directly penalizes the L1 norm of the offset, preventing it from
    diverging while still allowing small corrections.

    Args:
        pred_depth: (B, V, H, W) — depth with GSDPT offset applied.
        teacher_depth: (B, V, H, W) — frozen Depth Head output (no offset).
        confidence: optional (B, V, H, W) per-pixel weight (broadcast to pred_depth).
            When given, returns ``mean(|pred - teacher| * confidence)`` (simple mean,
            not normalized by Σ conf).
    """
    if confidence is None:
        return F.l1_loss(pred_depth, teacher_depth)
    return ((pred_depth - teacher_depth).abs() * confidence).mean()


def _masked_mean(
    elementwise: torch.Tensor, graph_ref: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean of ``elementwise`` over True pixels of ``mask`` (broadcast to it).

    ``mask`` may be (B, 1, H, W) or (B, H, W); it is broadcast across channels.
    Returns a graph-attached zero (via ``graph_ref``) when the mask is empty, so
    backward stays valid.
    """
    if mask.dim() == elementwise.dim() - 1:
        mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    mask = mask.to(elementwise.dtype)
    mask_b = mask.expand_as(elementwise)
    denom = mask_b.sum()
    if denom == 0:
        return graph_ref.sum() * 0.0
    return (elementwise * mask_b).sum() / denom


def l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-pixel L1 loss. With ``mask`` (broadcastable to ``pred``), masked mean."""
    if mask is None:
        return F.l1_loss(pred, target)
    return _masked_mean((pred - target).abs(), pred, mask)


def sobel_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Sobel edge L1 loss for sharpness.

    Args:
        pred: (B, 3, H, W) rendered images.
        target: (B, 3, H, W) ground truth images.
        mask: optional (B, 1, H, W) or (B, H, W) — masked mean over valid pixels.
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=pred.dtype, device=pred.device,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=pred.dtype, device=pred.device,
    ).reshape(1, 1, 3, 3)

    # RGB to grayscale
    pred_gray = pred.mean(dim=1, keepdim=True)
    target_gray = target.mean(dim=1, keepdim=True)

    pred_gx = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_gray, sobel_y, padding=1)
    pred_edge = (pred_gx ** 2 + pred_gy ** 2 + 1e-8).sqrt()

    target_gx = F.conv2d(target_gray, sobel_x, padding=1)
    target_gy = F.conv2d(target_gray, sobel_y, padding=1)
    target_edge = (target_gx ** 2 + target_gy ** 2 + 1e-8).sqrt()

    if mask is None:
        return F.l1_loss(pred_edge, target_edge)
    return _masked_mean((pred_edge - target_edge).abs(), pred_edge, mask)


def offset_xy_loss(offset_xy: torch.Tensor, norm: str = "l2") -> torch.Tensor:
    """XY offset regularization — encourage offsets to stay near zero.

    Args:
        offset_xy: (B, V, H, W, 2) predicted sub-pixel offsets.
        norm: "l1" or "l2".
    """
    if norm == "l1":
        return offset_xy.abs().mean()
    return (offset_xy ** 2).mean()


def opacity_entropy_loss(opacities: torch.Tensor) -> torch.Tensor:
    """Binary entropy regularization — encourage opacity toward 0 or 1.

    H = -p*log(p) - (1-p)*log(1-p), averaged over all elements.

    Args:
        opacities: opacity values in [0, 1], any shape.
    """
    p = opacities.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


def render_depth_l1_loss(
    rendered_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    mask: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 between rendered depth (target view) and frozen teacher depth.

    Args:
        rendered_depth: (N, H, W) — gsplat-rasterized depth flattened over batch & view.
        teacher_depth:  (N, H, W) — frozen depth on the same views.
        mask: optional (N, H, W) bool — if given, only mask=True pixels contribute.
        confidence: optional (N, H, W) per-pixel weight. When given, the L1 is
            element-wise multiplied by it before the masked mean (simple mean,
            not normalized by Σ conf).

    Returns a scalar tensor; if mask is empty, returns a graph-attached zero so
    backward stays valid.
    """
    diff = (rendered_depth - teacher_depth).abs()
    if confidence is not None:
        diff = diff * confidence
    if mask is None:
        return diff.mean()
    if mask.sum() == 0:
        return rendered_depth.sum() * 0.0
    return diff[mask].mean()


def render_depth_gradient_l1_loss(
    rendered_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    mask: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 of finite-difference gradients of (rendered - teacher) along x and y.

    Encourages structural agreement between rendered and teacher depth without
    forcing absolute alignment.

    Args:
        rendered_depth: (N, H, W).
        teacher_depth:  (N, H, W).
        mask: optional (N, H, W) bool. A finite-difference pair is valid only
            when both endpoints are masked True.
        confidence: optional (N, H, W) per-pixel weight. Each gradient pair is
            weighted by the mean of its two endpoints' confidences (simple mean,
            not normalized by Σ conf).
    """
    diff = rendered_depth - teacher_depth  # (N, H, W)
    dx = (diff[:, :, 1:] - diff[:, :, :-1]).abs()
    dy = (diff[:, 1:, :] - diff[:, :-1, :]).abs()

    if confidence is not None:
        conf_x = (confidence[:, :, 1:] + confidence[:, :, :-1]) * 0.5
        conf_y = (confidence[:, 1:, :] + confidence[:, :-1, :]) * 0.5
        dx = dx * conf_x
        dy = dy * conf_y

    if mask is None:
        return (dx.mean() + dy.mean()) * 0.5

    mask_x = mask[:, :, 1:] & mask[:, :, :-1]
    mask_y = mask[:, 1:, :] & mask[:, :-1, :]

    if mask_x.sum() == 0 and mask_y.sum() == 0:
        return rendered_depth.sum() * 0.0

    if mask_x.sum() > 0:
        loss_x = dx[mask_x].mean()
    else:
        loss_x = dx.sum() * 0.0
    if mask_y.sum() > 0:
        loss_y = dy[mask_y].mean()
    else:
        loss_y = dy.sum() * 0.0
    return (loss_x + loss_y) * 0.5


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute PSNR between pred and target images (both in [0, 1])."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0, device=pred.device)
    return -10.0 * torch.log10(mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SSIM using avg_pool2d approximation.

    Args:
        pred: (B, C, H, W) in [0, 1]
        target: (B, C, H, W) in [0, 1]
        mask: optional (B, 1, H, W) or (B, H, W) — masked mean over valid pixels.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pad = window_size // 2

    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=pad) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if mask is None:
        return ssim_map.mean()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    mask = mask.to(ssim_map.dtype)
    mask_b = mask.expand_as(ssim_map)
    denom = mask_b.sum().clamp_min(1.0)
    return (ssim_map * mask_b).sum() / denom
