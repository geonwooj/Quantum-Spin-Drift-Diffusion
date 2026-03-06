from __future__ import annotations

import os
import tensorflow as tf


def configure_reproducibility(seed: int = 777, deterministic_ops: bool = True) -> int:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    tf.keras.utils.set_random_seed(seed)
    return seed
