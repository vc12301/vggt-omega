"""Native PyTorch training loop for GSDPT head retraining.

Supports:
- Single-GPU / multi-GPU via ``torchrun`` (DDP)
- Mixed precision (bf16 / fp16)
- Gradient accumulation
- torch.compile on the GSDPT head
- WandB logging
- Checkpoint management (top-K by val PSNR + last-K)
"""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import nullcontext
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from vggt_omega.training.config import (
    GSDPTTrainingConfig,
    load_config,
    parse_cli_overrides,
)
from vggt_omega.training.data.datamodule import build_dataloaders
from vggt_omega.training.losses import compute_psnr, compute_ssim
from vggt_omega.training.model_wrapper import GSModel
from vggt_omega.training.utils import CosineAnnealingWarmupScheduler
from vggt_omega.utils.gs_merge import get_curriculum_voxel_size

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Distributed helpers
# ──────────────────────────────────────────────────────────────────────
def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_main() -> bool:
    return _rank() == 0


def _setup_dist(backend: str = "nccl"):
    if "RANK" in os.environ:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _cleanup_dist():
    if _is_dist():
        dist.destroy_process_group()


def _gs_head_module(raw_model):
    """Return the underlying GSDPT head, unwrapping torch.compile if present.

    ``torch.compile`` wraps the module in an ``OptimizedModule`` whose
    ``state_dict()`` keys are prefixed with ``_orig_mod.``. Saving/loading the
    original module keeps checkpoints portable to an un-compiled GSDPTHead.
    """
    head = raw_model.model.gs_head
    return getattr(head, "_orig_mod", head)


# ──────────────────────────────────────────────────────────────────────
# Checkpoint management
# ──────────────────────────────────────────────────────────────────────
class CheckpointManager:
    """Keep top-K by metric + last-K checkpoints."""

    def __init__(self, save_dir: str, top_k: int = 10, last_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.last_k = last_k
        self._best: list[tuple[float, Path]] = []  # (val_psnr, path) — higher is better
        self._latest: list[Path] = []

    def save(self, state: dict, step: int, val_psnr: Optional[float] = None):
        # --- latest ---
        path_latest = self.save_dir / f"gsdpt-latest-{step}.pt"
        torch.save(state, path_latest)
        self._latest.append(path_latest)
        while len(self._latest) > self.last_k:
            old = self._latest.pop(0)
            old.unlink(missing_ok=True)

        # --- best (by PSNR, higher is better) ---
        if val_psnr is not None:
            path_best = self.save_dir / f"gsdpt-{step}-psnr{val_psnr:.2f}.pt"
            if path_best != path_latest:
                shutil.copy2(path_latest, path_best)
            self._best.append((val_psnr, path_best))
            self._best.sort(key=lambda x: -x[0])  # descending: highest PSNR first
            while len(self._best) > self.top_k:
                _, old_path = self._best.pop()
                if old_path not in set(self._latest):
                    old_path.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────
# WandB preview helper
# ──────────────────────────────────────────────────────────────────────
def _log_preview(wandb_run, model, batch, step: int, prefix: str = "val"):
    """Render one scene and log rendered/GT/depth images to WandB."""
    try:
        import numpy as np
        import wandb
    except ImportError:
        return
    if model.cfg.voxel_merge.enabled:
        batch["voxel_size"] = model.cfg.voxel_merge.eval_voxel_size
    model.eval()
    with torch.no_grad():
        fwd = model(batch)
    model.train()

    rendered = fwd["rendered_images"]
    if rendered.dim() == 5:
        rendered = rendered.squeeze(0)
    target_gt = batch["target_images"]
    if target_gt.dim() == 5:
        target_gt = target_gt.squeeze(0)

    rend_np = rendered[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().float().numpy()
    gt_np = target_gt[0].permute(1, 2, 0).cpu().float().numpy()
    comparison = np.concatenate([gt_np, rend_np], axis=1)  # side-by-side

    log_dict = {
        f"{prefix}_preview/gt_vs_rendered": wandb.Image(
            comparison, caption=f"step {step} | Left: GT  Right: Rendered"
        ),
    }

    depth = fwd.get("rendered_depths")
    if depth is not None and depth.numel() > 0:
        d = depth.squeeze()
        if d.dim() >= 2:
            d = d[0] if d.dim() == 3 else d
            d_np = d.cpu().float().numpy()
            d_min, d_max = d_np.min(), d_np.max()
            d_norm = (d_np - d_min) / (d_max - d_min + 1e-8)
            log_dict[f"{prefix}_preview/depth"] = wandb.Image(
                d_norm, caption=f"step {step} | depth [{d_min:.2f}, {d_max:.2f}]"
            )

    mask = fwd.get("target_loss_mask")
    if mask is not None and mask.numel() > 0:
        m = mask
        if m.dim() == 4:  # (B, M, H, W) -> first view
            m = m.reshape(-1, m.shape[-2], m.shape[-1])
        m0 = m[0]
        frac = m.float().mean().item()
        m_np = m0.cpu().float().numpy()
        log_dict[f"{prefix}_preview/target_mask"] = wandb.Image(
            m_np, caption=f"step {step} | valid frac {frac:.3f}"
        )

    wandb_run.log(log_dict, step=step)


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model: GSModel, val_loader: DataLoader, device):
    model.eval()
    total_loss, total_psnr, total_ssim, total_lpips, n = 0.0, 0.0, 0.0, 0.0, 0
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Validation always uses the fixed eval voxel size so PSNR/SSIM and
        # checkpoint selection stay comparable across steps.
        if model.cfg.voxel_merge.enabled:
            batch["voxel_size"] = model.cfg.voxel_merge.eval_voxel_size
        fwd = model(batch)
        losses = model.compute_loss(fwd, batch)

        rendered = fwd["rendered_images"]
        gt = batch["target_images"]
        if rendered.dim() == 5:
            rendered = rendered.squeeze(0)
        if gt.dim() == 5:
            gt = gt.squeeze(0)
        rendered = rendered.clamp(0.0, 1.0)

        total_loss += losses["loss"].item()
        total_psnr += compute_psnr(rendered, gt).item()
        total_ssim += compute_ssim(rendered, gt).item()
        total_lpips += losses["loss_lpips"].item()
        n += 1

    model.train()

    # Under DDP each rank only sees its own shard; all-reduce the running sums
    # and counts so every rank computes the same global mean (rank 0 then uses
    # it for checkpoint selection).
    if _is_dist():
        t = torch.tensor(
            [total_loss, total_psnr, total_ssim, total_lpips, float(n)],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_psnr, total_ssim, total_lpips, n_f = t.tolist()
        n = int(round(n_f))

    if n == 0:
        return {"val/loss": 0.0, "val/psnr": 0.0, "val/ssim": 0.0, "val/lpips": 0.0}
    return {
        "val/loss": total_loss / n,
        "val/psnr": total_psnr / n,
        "val/ssim": total_ssim / n,
        "val/lpips": total_lpips / n,
    }


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────
def main(config: GSDPTTrainingConfig | None = None):
    # ── Config ──
    if config is None:
        args = sys.argv[1:]
        yaml_path = None
        override_args = []
        for a in args:
            if a == "--config":
                continue
            elif a.startswith("--config="):
                yaml_path = a.split("=", 1)[1]
            elif a.endswith((".yaml", ".yml")):
                yaml_path = a
            elif "=" in a:
                override_args.append(a)
        for i, a in enumerate(args):
            if a == "--config" and i + 1 < len(args):
                yaml_path = args[i + 1]
        if yaml_path is None:
            yaml_path = str(
                Path(__file__).resolve().parent.parent.parent / "configs" / "gsdpt_training.yaml"
            )
        config = load_config(yaml_path, parse_cli_overrides(override_args))

    # ── Distributed ──
    _setup_dist(config.distributed.backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if _is_main():
        logger.info(f"World size: {_world()}, Rank: {_rank()}")

    torch.manual_seed(config.seed + _rank())

    # ── AMP scaler. The aggregator's autocast dtype is set from
    # config.training.mixed_precision inside GSModel; the trainable gs_head and
    # rendering always run in fp32. GradScaler is only needed for fp16. ──
    amp_dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = amp_dtype_map.get(config.training.mixed_precision, torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # ── Data ──
    # Shared step counter for curriculum view-sampling. A multiprocessing.Value
    # is visible to forked (persistent) dataloader workers; the main loop updates
    # it once per optimizer step so the dataset's curriculum gap can advance.
    import multiprocessing as _mp

    step_counter = _mp.Value("i", 0)
    train_loader, val_loaders = build_dataloaders(config, _world(), _rank(), step_counter=step_counter)

    # ── Model ──
    model = GSModel(config).to(device)

    if config.training.torch_compile:
        logger.info("Applying torch.compile to GSDPT head")
        model.model.gs_head = torch.compile(model.model.gs_head)

    if _is_dist():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model: GSModel = model.module if isinstance(model, DDP) else model

    # ── Optimizer / scheduler ──
    optimizer = torch.optim.AdamW(
        raw_model.trainable_parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
    )
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_steps=config.scheduler.warmup_steps,
        max_steps=config.training.max_iterations,
        min_lr=config.scheduler.min_lr,
    )

    # ── Resume ──
    start_step = 0
    if config.model.resume_full_checkpoint is not None:
        ckpt = torch.load(
            config.model.resume_full_checkpoint, map_location=device, weights_only=False
        )
        _gs_head_module(raw_model).load_state_dict(ckpt["gs_head"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed from step {start_step}")

    # ── WandB ──
    wandb_run = None
    if _is_main() and not config.wandb.offline:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.wandb.project, name=config.wandb.name, config=vars(config)
            )
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # ── Checkpoint manager ──
    ckpt_mgr = (
        CheckpointManager(
            config.checkpoint.save_dir,
            top_k=config.checkpoint.save_top_k,
            last_k=config.checkpoint.save_last_k,
        )
        if _is_main()
        else None
    )

    # ── Training loop ──
    accum = config.training.gradient_accumulation_steps
    log_interval = config.wandb.log_every_n_steps
    val_interval = config.validation.val_interval
    max_steps = config.training.max_iterations

    train_iter = iter(train_loader)
    model.train()
    optimizer.zero_grad()

    step = start_step  # counts OPTIMIZER steps (not micro-steps)
    step_counter.value = start_step
    log_loss_accum = 0.0
    log_comp_accum: dict = {}
    log_count = 0
    last_val_metrics: dict = {}
    diag_done = False

    from tqdm import tqdm

    pbar = tqdm(
        initial=start_step,
        total=max_steps,
        desc="Training",
        disable=not _is_main(),
        dynamic_ncols=True,
        smoothing=0.05,
    )

    while step < max_steps:
        # Accumulate `accum` micro-batches into one optimizer step. The LR
        # schedule, validation, checkpointing, and max_steps are all keyed to
        # optimizer steps (not micro-steps), so gradient_accumulation_steps>1
        # behaves like a larger effective batch rather than fast-forwarding.
        optimizer.zero_grad()
        micro_loss = 0.0
        micro_comp: dict = {}
        for micro in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                if _is_dist() and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(step)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            # Per-step voxel size for the (optional) Gaussian merge: sampled under
            # the curriculum schedule, or the fixed value. Computed in the main
            # process where `step` is known (no dataset-worker plumbing).
            if config.voxel_merge.enabled:
                batch["voxel_size"] = (
                    get_curriculum_voxel_size(step, config.voxel_merge)
                    if config.voxel_merge.curriculum
                    else config.voxel_merge.voxel_size
                )

            # On non-final micro-steps under DDP, skip the gradient all-reduce.
            # no_sync() must wrap BOTH forward and backward — DDP decides what to
            # reduce during the forward pass, so wrapping only backward leaves the
            # reducer armed and the optimization has no effect.
            is_last_micro = micro == accum - 1
            sync_ctx = (
                model.no_sync()
                if (isinstance(model, DDP) and not is_last_micro)
                else nullcontext()
            )
            with sync_ctx:
                # Forward + loss. The wrapper manages autocast internally
                # (aggregator per config.mixed_precision, gs_head + render in fp32).
                fwd = model(batch)
                losses = raw_model.compute_loss(fwd, batch)
                loss = losses["loss"] / accum

                if not diag_done and _is_main():
                    rendered = fwd["rendered_images"]
                    pbar.write(
                        f"[DIAG step {step}] rendered shape={tuple(rendered.shape)} "
                        f"dtype={rendered.dtype} "
                        f"range=[{rendered.min().item():.4f}, {rendered.max().item():.4f}]"
                    )
                    pbar.write(
                        f"[DIAG step {step}] loss={losses['loss'].item():.4f} "
                        f"photo={losses['loss_photo'].item():.4f} "
                        f"lpips={losses['loss_lpips'].item():.4f} "
                        f"depth={losses['loss_depth'].item():.4f}"
                    )

                scaler.scale(loss).backward()

            if not diag_done and _is_main():
                grad_norms = []
                for name, p in raw_model.model.gs_head.named_parameters():
                    if p.grad is not None:
                        grad_norms.append((name, p.grad.norm().item()))
                if grad_norms:
                    total_grad = sum(g for _, g in grad_norms)
                    pbar.write(
                        f"[DIAG step {step}] grad_norm total={total_grad:.6f}, "
                        f"num_params_with_grad={len(grad_norms)}"
                    )
                    grad_norms.sort(key=lambda x: -x[1])
                    for name, gn in grad_norms[:3]:
                        pbar.write(f"  grad {name}: {gn:.6f}")
                else:
                    pbar.write(f"[DIAG step {step}] WARNING: NO gradients in gs_head!")
                diag_done = True

            micro_loss += losses["loss"].item()
            for _k, _v in losses.items():
                if _k.startswith("loss_"):
                    micro_comp[_k] = micro_comp.get(_k, 0.0) + (
                        _v.item() if hasattr(_v, "item") else _v
                    )

        # ── Optimizer step (once per `accum` micro-steps) ──
        if config.training.gradient_clip_val > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                raw_model.trainable_parameters(), config.training.gradient_clip_val
            )
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        step += 1
        step_counter.value = step  # advance curriculum gap in dataset workers
        cur_loss = micro_loss / accum
        log_loss_accum += cur_loss
        log_count += 1
        for _k, _v in micro_comp.items():
            log_comp_accum[_k] = log_comp_accum.get(_k, 0.0) + _v / accum

        if _is_main():
            postfix = {
                "loss": f"{cur_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if last_val_metrics:
                postfix["v_psnr"] = f"{last_val_metrics['val/psnr']:.2f}"
            pbar.set_postfix(postfix, refresh=False)
            pbar.update(1)

        # ── WandB logging ──
        if _is_main() and step % log_interval == 0:
            avg_loss = log_loss_accum / log_count
            if wandb_run:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                for key, val_sum in log_comp_accum.items():
                    log_dict[f"train/{key[5:]}"] = val_sum / log_count
                if config.voxel_merge.enabled:
                    log_dict["train/num_gaussians"] = fwd["num_gaussians"]
                    log_dict["train/voxel_size"] = batch.get("voxel_size", 0.0)
                wandb_run.log(log_dict, step=step)
                _log_preview(wandb_run, raw_model, batch, step, prefix="train")
            log_loss_accum = 0.0
            log_comp_accum = {}
            log_count = 0

        # ── Validation + checkpoint ──
        if step % val_interval == 0:
            pbar.set_description("Validating")
            per_ds_metrics: dict = {}
            for ds_name, ds_loader in val_loaders.items():
                per_ds_metrics[ds_name] = validate(raw_model, ds_loader, device)

            n_ds = max(1, len(per_ds_metrics))
            mean_metrics = {
                "val/loss": sum(m["val/loss"] for m in per_ds_metrics.values()) / n_ds,
                "val/psnr": sum(m["val/psnr"] for m in per_ds_metrics.values()) / n_ds,
                "val/ssim": sum(m["val/ssim"] for m in per_ds_metrics.values()) / n_ds,
                "val/lpips": sum(m["val/lpips"] for m in per_ds_metrics.values()) / n_ds,
            }
            last_val_metrics = mean_metrics
            if _is_main():
                for ds_name, m in per_ds_metrics.items():
                    pbar.write(
                        f"[VAL step {step}] [{ds_name}]  loss={m['val/loss']:.4f}  "
                        f"psnr={m['val/psnr']:.2f}  ssim={m['val/ssim']:.4f}  "
                        f"lpips={m['val/lpips']:.4f}"
                    )
                if wandb_run:
                    wandb_run.log(mean_metrics, step=step)
                    for ds_name, m in per_ds_metrics.items():
                        wandb_run.log(
                            {f"val_{ds_name}/{k.split('/', 1)[1]}": v for k, v in m.items()},
                            step=step,
                        )
                    num_previews = config.validation.num_preview_scenes
                    for ds_name, ds_loader in val_loaders.items():
                        val_preview_iter = iter(ds_loader)
                        for pi in range(num_previews):
                            try:
                                preview_batch = next(val_preview_iter)
                            except StopIteration:
                                break
                            preview_batch = {
                                k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in preview_batch.items()
                            }
                            _log_preview(
                                wandb_run, raw_model, preview_batch, step,
                                prefix=f"val_{ds_name}_{pi}",
                            )

                state = {
                    "step": step,
                    "gs_head": _gs_head_module(raw_model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config,
                }
                ckpt_mgr.save(state, step, val_psnr=mean_metrics["val/psnr"])

            if _is_dist():
                dist.barrier()
            model.train()
            pbar.set_description("Training")

    pbar.close()

    # ── Final save ──
    if _is_main():
        state = {
            "step": step,
            "gs_head": _gs_head_module(raw_model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
        }
        ckpt_mgr.save(state, step)
        logger.info(f"Training complete. {step} steps.")
        if wandb_run:
            wandb_run.finish()

    _cleanup_dist()


if __name__ == "__main__":
    main()
