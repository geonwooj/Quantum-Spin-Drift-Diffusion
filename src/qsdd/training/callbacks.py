from __future__ import annotations


def format_train_log(step: int, loss: float, grad_norm: float, grad_norm_ema: float, grad_norm_max: float) -> str:
    return (
        f"step {step:6d} | loss {loss:.12f} | "
        f"gn {grad_norm:8.3f} | gn_ema {grad_norm_ema:8.3f} | gn_max {grad_norm_max:8.3f}"
    )
