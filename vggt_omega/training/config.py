"""Training configuration for GSDPT head retraining (ported from Depth-Anything-3).

VGGT-Omega is pose-free: the aggregator does not consume camera tokens, so there
is no camera-dropout or extrinsic-normalization config (equivalent to DA3's
``camera_dropout_prob=1.0``). The DA3 ``slim`` GS variant is also dropped because
VGGT-Omega's GaussianAdapter has a fixed channel layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------
@dataclass
class ResolutionSchedule:
    resolution: List[int] = field(default_factory=lambda: [272, 480])
    context_views: int = 12
    target_views: int = 4
    weight: float = 0.5


@dataclass
class ModelConfig:
    # Local VGGT-Omega checkpoint (released checkpoints carry no GS weights; the
    # gs_head keeps its initialization, loaded with strict=False).
    checkpoint_path: str = "ckpts/VGGT-Omega/vggt_omega_1b_512.pt"
    gs_sh_degree: int = 0
    resume_gsdpt_checkpoint: Optional[str] = None
    resume_full_checkpoint: Optional[str] = None
    freeze_backbone: bool = True
    init_mode: str = "finetune"  # finetune | scratch | reset_output


@dataclass
class DataConfig:
    data_root: str = "/data-nas/data/dataset/open_source/DL3DV-10K/960p"
    train_buckets: List[str] = field(
        default_factory=lambda: [
            "1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "11K",
        ]
    )
    val_root: str = "/data-nas/data/dataset/open_source/DL3DV-10K/benchmark_140"
    val_scene_subdir: str = "nerfstudio"  # subdir within each val scene containing transforms.json
    max_train_scenes: Optional[int] = None  # limit for debugging
    max_val_scenes: Optional[int] = None  # limit for debugging
    num_workers: int = 8
    pin_memory: bool = True
    image_dir_name: str = "images_4"
    val_context_gap: Optional[int] = None  # if set, val context views sampled from first N frames
    # Per-dataset overrides (None -> fall back to the global view_sampling / curriculum).
    view_sampling: Optional["ViewSamplingConfig"] = None
    curriculum: Optional["CurriculumConfig"] = None


@dataclass
class RenderingDataConfig:
    """Config for the synthetic ``rendering_dataset`` (pano-perspective renders).

    Layout: ``{data_root}/{scene_type}/{idx}/perspective_frames/frame_XXXXX.png``
    with cameras at ``{idx}/perspective_camera/Camera{N}.json`` (+ ``meta.json``).
    """

    data_root: str = (
        "/data-nas/experiments/yemu/workspace/pano-video-data-process/rendering_dataset"
    )
    val_split_ratio: float = 0.02  # per scene_type, last ceil(ratio) by name -> val
    image_dir_name: str = "perspective_frames"
    camera_dir_name: str = "perspective_camera"
    max_train_scenes: Optional[int] = None  # limit for debugging
    max_val_scenes: Optional[int] = None  # limit for debugging
    num_workers: int = 8
    pin_memory: bool = True
    val_context_gap: Optional[int] = None  # if set, val context views sampled from first N frames
    # Per-dataset overrides (None -> fall back to the global view_sampling / curriculum).
    view_sampling: Optional["ViewSamplingConfig"] = None
    curriculum: Optional["CurriculumConfig"] = None


@dataclass
class ViewSamplingConfig:
    resolution_schedules: List[ResolutionSchedule] = field(
        default_factory=lambda: [
            ResolutionSchedule(
                resolution=[272, 480], context_views=12, target_views=4, weight=0.5
            ),
            ResolutionSchedule(resolution=[384, 672], context_views=8, target_views=3, weight=0.3),
            ResolutionSchedule(resolution=[512, 896], context_views=4, target_views=2, weight=0.2),
        ]
    )
    fps_rotation_weight: float = 1.0
    min_scene_frames: int = 20  # skip scenes with fewer frames


@dataclass
class CurriculumConfig:
    enabled: bool = True
    min_context_gap_start: int = 15  # initial min frame gap between context boundary views
    min_context_gap_end: int = 20  # final min gap (after warmup)
    max_context_gap_start: int = 30  # initial max frame gap
    max_context_gap_end: int = 50  # final max gap (after warmup)
    warmup_steps: int = 10000  # linear warmup duration


@dataclass
class TrainingConfig:
    max_iterations: int = 100000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # bf16 | fp16 | fp32
    gradient_clip_val: float = 1.0
    torch_compile: bool = True  # torch.compile on GSDPT head


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 2e-5
    weight_decay: float = 0.05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine_annealing"
    warmup_steps: int = 2000
    min_lr: float = 1e-6


@dataclass
class LossConfig:
    lambda_lpips: float = 0.05
    lambda_depth: float = 0.5
    lpips_net: str = "vgg"
    # --- Post-training loss terms (all default to 0 / inactive for backward compat) ---
    photometric_loss: str = "mse"  # "mse" | "l1"
    lambda_ssim: float = 0.0  # weight for (1 - SSIM) loss
    lambda_sobel: float = 0.0  # weight for Sobel edge L1 loss
    lambda_offset_xy: float = 0.0  # weight for XY offset regularization
    offset_xy_norm: str = "l2"  # "l1" | "l2"
    lambda_opacity_entropy: float = 0.0  # weight for opacity entropy regularization
    # --- Context-view supervision (default off; backward compatible) ---
    supervise_context: bool = False  # also render+supervise on context views
    lambda_context: float = 0.5  # outer scalar on the total context-view loss
    context_only: bool = (
        False  # if True, drop target-view loss entirely (implies supervise_context=True)
    )
    # --- Per-view self-reconstruction regularization (default off) ---
    # Each context view renders ONLY its OWN gaussians back to itself (diagonal
    # supervision), unlike supervise_context which renders the union of all
    # context-view gaussians to every context view.
    lambda_self_recon: float = 0.0  # outer scalar on the per-view self-recon loss
    # --- Scale regularization (logit-space, pre-sigmoid) ---
    lambda_scale: float = 0.0  # weight for raw-scale-logit regularization (0 = off)
    scale_norm: str = "l2"  # "l1" | "l2"
    # --- Target-view rendered-depth regularization (vs frozen depth) ---
    lambda_render_depth_l1: float = 0.0  # L1(rendered_depth, depth) on target views
    lambda_render_depth_grad_l1: float = (
        0.0  # L1 of xy gradients of (rendered - teacher) on target views
    )
    # --- Target-view valid-visible-pixel mask (default off; backward compatible) ---
    # When enabled, a per-target-pixel mask is computed (in-frustum of any context
    # view + valid depth + depth-consistent) and applied to the target-view
    # photometric / LPIPS / SSIM / Sobel losses. See training/frustum_mask.py:
    # calculate_in_frustum_mask.
    enable_target_loss_mask: bool = False
    target_loss_mask_depth_atol: float = 0.1  # condition-3 depth-match absolute tolerance
    target_loss_mask_depth_rtol: float = 0.0  # condition-3 depth-match relative tolerance


@dataclass
class ValidationConfig:
    val_interval: int = 500
    num_preview_scenes: int = 4


@dataclass
class CheckpointConfig:
    save_top_k: int = 10
    save_last_k: int = 3
    save_dir: str = "checkpoints/gsdpt"
    filename: str = "gsdpt-iter{step}-vloss{val_loss:.4f}"


@dataclass
class WandbConfig:
    project: str = "vggt-omega-gsdpt-training"
    name: Optional[str] = None
    log_every_n_steps: int = 50
    offline: bool = False


@dataclass
class DistributedConfig:
    backend: str = "nccl"  # nccl | gloo


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class GSDPTTrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rendering_data: RenderingDataConfig = field(default_factory=RenderingDataConfig)
    view_sampling: ViewSamplingConfig = field(default_factory=ViewSamplingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    seed: int = 42
    # --- Dataset selection ---
    # "dl3dv" (default) | "rendering" | "mixed"
    dataset_mode: str = "dl3dv"
    # Sampling weights for mixed mode: [dl3dv, rendering]. Only used when
    # dataset_mode == "mixed". Need not sum to 1 (normalized internally).
    mix_ratio: List[float] = field(default_factory=lambda: [1.0, 1.0])


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------
# Optional nested-dataclass fields that default to ``None`` (so the generic
# "recurse if current is a dataclass" branch can't reach them). Maps the YAML
# key to the dataclass to instantiate when the key is present.
_OPTIONAL_DATACLASS_FIELDS = {
    "view_sampling": ViewSamplingConfig,
    "curriculum": CurriculumConfig,
}


def _apply_dict_to_dataclass(dc, d: dict, strict: bool = False):
    """Recursively apply dict values to a dataclass instance.

    Args:
        strict: if True, raise ValueError on an unknown key (used for CLI
            overrides); if False, warn and skip (used for YAML).
    """
    import dataclasses

    # Field types are strings here because of `from __future__ import annotations`.
    field_types = (
        {f.name: f.type for f in dataclasses.fields(dc)} if dataclasses.is_dataclass(dc) else {}
    )
    for k, v in d.items():
        if not hasattr(dc, k):
            # Unknown key. For CLI overrides (strict=True) this is almost always
            # a typo (e.g. `training.lr` instead of `optimizer.lr`), so fail fast;
            # for YAML we warn but continue (forward/backward-compat friendly).
            msg = (
                f"Unknown config key '{k}' on {type(dc).__name__} "
                f"(not a recognized field)."
            )
            if strict:
                raise ValueError(msg)
            import warnings

            warnings.warn(f"Ignoring {msg}", stacklevel=2)
            continue
        current = getattr(dc, k)
        if hasattr(current, "__dataclass_fields__") and isinstance(v, dict):
            _apply_dict_to_dataclass(current, v, strict=strict)
        elif current is None and isinstance(v, dict) and k in _OPTIONAL_DATACLASS_FIELDS:
            # Optional nested dataclass (e.g. per-dataset view_sampling / curriculum
            # override) that defaults to None — instantiate, then recurse.
            obj = _OPTIONAL_DATACLASS_FIELDS[k]()
            _apply_dict_to_dataclass(obj, v, strict=strict)
            setattr(dc, k, obj)
        elif isinstance(v, list) and k == "resolution_schedules":
            setattr(
                dc,
                k,
                [ResolutionSchedule(**item) if isinstance(item, dict) else item for item in v],
            )
        else:
            # Coerce numeric strings to the declared type — protects against
            # YAML quirks where e.g. "5e-5" (no decimal point) parses as str
            # under PyYAML 1.1 rules.
            t = field_types.get(k)
            if isinstance(v, str) and t in ("float", "int", float, int):
                cast = float if t in ("float", float) else int
                try:
                    v = cast(v)
                except (ValueError, TypeError):
                    pass
            setattr(dc, k, v)


def load_config(
    yaml_path: Optional[str] = None, cli_overrides: Optional[dict] = None
) -> GSDPTTrainingConfig:
    """Load config from YAML and apply CLI overrides.

    CLI overrides are dot-separated keys, e.g. ``{"optimizer.lr": 1e-3}``.
    Unknown CLI keys raise ValueError (fail-fast); unknown YAML keys warn.
    """
    cfg = GSDPTTrainingConfig()

    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.exists():
            # An explicitly-passed config path that doesn't exist is almost
            # always a typo — fail fast rather than silently using defaults and
            # training for hours with the wrong settings.
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        _apply_dict_to_dataclass(cfg, raw)

    if cli_overrides:
        # Flatten dot-notation into nested dict
        nested: dict = {}
        for key, val in cli_overrides.items():
            parts = key.split(".")
            d = nested
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
        # CLI overrides are typed by hand at launch — fail fast on unknown keys.
        _apply_dict_to_dataclass(cfg, nested, strict=True)

    _validate_config(cfg)
    return cfg


# Allowed values for the enum-like string fields, and the constraints the
# current implementation actually honors. Validated up-front so a mistyped or
# unsupported value fails fast instead of silently falling back (or crashing
# mid-run).
_ALLOWED = {
    "dataset_mode": {"dl3dv", "rendering", "mixed"},
    "optimizer.name": {"adamw"},
    "scheduler.name": {"cosine_annealing"},
    "training.mixed_precision": {"bf16", "fp16", "fp32"},
    "loss.photometric_loss": {"mse", "l1"},
    "loss.offset_xy_norm": {"l1", "l2"},
    "loss.scale_norm": {"l1", "l2"},
}


def _validate_config(cfg: GSDPTTrainingConfig) -> None:
    """Fail-fast semantic validation of a fully-loaded config.

    Catches values that are accepted by the dataclass but not actually honored
    by the implementation (silent fallback) or that would crash later.
    """
    errors: List[str] = []

    # Enum-like string fields.
    enum_targets = {
        "dataset_mode": cfg.dataset_mode,
        "optimizer.name": cfg.optimizer.name,
        "scheduler.name": cfg.scheduler.name,
        "training.mixed_precision": cfg.training.mixed_precision,
        "loss.photometric_loss": cfg.loss.photometric_loss,
        "loss.offset_xy_norm": cfg.loss.offset_xy_norm,
        "loss.scale_norm": cfg.loss.scale_norm,
    }
    for key, val in enum_targets.items():
        allowed = _ALLOWED[key]
        if val not in allowed:
            errors.append(f"{key}={val!r} is not supported (allowed: {sorted(allowed)})")

    # batch_size is a reserved field; the data pipeline is hard-wired to 1 scene
    # per step (variable K/M/resolution can't be stacked). Reject other values
    # rather than letting the user think a larger batch took effect.
    if cfg.training.batch_size != 1:
        errors.append(
            f"training.batch_size={cfg.training.batch_size} is not supported; only 1 is "
            "implemented (use gradient_accumulation_steps / DDP for a larger effective batch)."
        )

    # Positive-integer / positive-float constraints.
    pos_int = {
        "training.max_iterations": cfg.training.max_iterations,
        "training.gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "validation.val_interval": cfg.validation.val_interval,
        "wandb.log_every_n_steps": cfg.wandb.log_every_n_steps,
        "scheduler.warmup_steps": cfg.scheduler.warmup_steps,
    }
    for key, val in pos_int.items():
        if not isinstance(val, int) or val < 1:
            errors.append(f"{key}={val!r} must be a positive integer")
    if cfg.optimizer.lr <= 0:
        errors.append(f"optimizer.lr={cfg.optimizer.lr!r} must be > 0")

    # Resolutions must be multiples of the patch size (16) for both global and
    # per-dataset view-sampling schedules.
    def _check_schedules(vs, where):
        if vs is None:
            return
        for i, s in enumerate(vs.resolution_schedules):
            h, w = s.resolution
            if h % 16 != 0 or w % 16 != 0:
                errors.append(
                    f"{where} resolution_schedules[{i}]={[h, w]} must be multiples of 16 "
                    "(VGGT-Omega patch_size)"
                )

    _check_schedules(cfg.view_sampling, "view_sampling")
    _check_schedules(cfg.data.view_sampling, "data.view_sampling")
    _check_schedules(cfg.rendering_data.view_sampling, "rendering_data.view_sampling")

    if cfg.dataset_mode == "mixed" and len(cfg.mix_ratio) != 2:
        errors.append(f"mix_ratio must have exactly 2 entries [dl3dv, rendering], got {cfg.mix_ratio}")

    if errors:
        raise ValueError("Invalid GSDPT training config:\n  - " + "\n  - ".join(errors))


def parse_cli_overrides(args: list[str]) -> dict:
    """Parse ``key=value`` pairs from CLI args into a dict."""
    overrides = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        # Attempt type coercion
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except (ValueError, TypeError):
                pass
        if val == "true":
            val = True
        elif val == "false":
            val = False
        elif val == "null" or val == "none":
            val = None
        overrides[key] = val
    return overrides
