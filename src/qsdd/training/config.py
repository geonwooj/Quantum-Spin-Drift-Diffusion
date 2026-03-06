from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    k: int = 1000
    lr: float = 1e-4
    grad_clip: float = 100.0
    total_steps: int = 60000
    save_every: int = 5000
    resume: bool = True
    use_ema: bool = False
    lambda_rw: float = 1.0
    batch_domain: int = 16
    image_size: int = 128
    channels: int = 3
    seed_t_eps: int = 777
