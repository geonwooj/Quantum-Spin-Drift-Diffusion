from __future__ import annotations

import glob
from pathlib import Path


def load_latest_weights(model, weights_dir: str | Path, pattern: str = "denoise_fn_step*.weights.h5", exclude_ema: bool = True):
    weights_dir = str(weights_dir)
    wlist = sorted(glob.glob(str(Path(weights_dir) / pattern)))
    if exclude_ema:
        wlist = [p for p in wlist if "_ema" not in Path(p).name]
    if not wlist:
        raise FileNotFoundError(f"No weights found in {weights_dir} with pattern={pattern}")
    latest = wlist[-1]
    model.load_weights(latest)
    return latest
