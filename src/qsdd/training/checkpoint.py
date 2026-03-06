from __future__ import annotations

from pathlib import Path

import tensorflow as tf



def make_checkpoint(model, optimizer, step_var, rng) -> tf.train.Checkpoint:
    return tf.train.Checkpoint(step=step_var, model=model, optimizer=optimizer, rng=rng)



def restore_latest_checkpoint(ckpt: tf.train.Checkpoint, ckpt_dir: str | Path) -> str | None:
    latest = tf.train.latest_checkpoint(str(ckpt_dir))
    if latest:
        ckpt.restore(latest).expect_partial()
    return latest
