from __future__ import annotations

import tensorflow as tf


def ema_decay_schedule(step: int) -> float:
    return 0.0 if step < 30000 else 0.99


def _as_tensor(v):
    if hasattr(v, "read_value"):
        return v.read_value()
    if hasattr(v, "value"):
        try:
            return v.value()
        except TypeError:
            return v.value
    return tf.convert_to_tensor(v)


class EMAHelper:
    def __init__(self, model: tf.keras.Model):
        self.shadow_vars = [tf.Variable(_as_tensor(v), trainable=False, dtype=v.dtype) for v in model.trainable_variables]

    def sync_from_model(self, model: tf.keras.Model) -> None:
        for shadow, var in zip(self.shadow_vars, model.trainable_variables):
            shadow.assign(var)

    def update(self, model: tf.keras.Model, step: int) -> None:
        decay = ema_decay_schedule(step)
        d = tf.constant(decay, tf.float32)
        one_minus = tf.constant(1.0 - decay, tf.float32)
        for shadow, var in zip(self.shadow_vars, model.trainable_variables):
            shadow.assign(d * shadow + one_minus * var)

    def swap_into_model(self, model: tf.keras.Model):
        backup = [tf.identity(_as_tensor(v)) for v in model.trainable_variables]
        for var, shadow in zip(model.trainable_variables, self.shadow_vars):
            var.assign(shadow)
        return backup

    def restore_backup(self, model: tf.keras.Model, backup) -> None:
        for var, b in zip(model.trainable_variables, backup):
            var.assign(b)
