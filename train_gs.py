"""Entry point for GSDPT head training.

Usage:
    # Single GPU
    python train_gs.py --config configs/gsdpt_training.yaml

    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 train_gs.py --config configs/gsdpt_training.yaml \\
        training.max_iterations=50000 optimizer.lr=5e-5

CLI overrides use dot notation (e.g. ``optimizer.lr=1e-4``, ``training.max_iterations=50000``).
List-valued keys (e.g. ``mix_ratio``) can only be set in YAML, not via CLI.
"""
import os
os.environ['WANDB_BASE_URL'] = 'https://api.bandw.top'
os.environ['http_proxy'] = 'socks5h://183.129.139.252:12126'
os.environ['https_proxy'] = 'socks5h://183.129.139.252:12126'
os.environ['ALL_PROXY'] = 'socks5h://183.129.139.252:12126'

from vggt_omega.training.train import main

if __name__ == "__main__":
    main()
