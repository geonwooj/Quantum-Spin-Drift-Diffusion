from __future__ import annotations

import math
import numpy as np
import tensorflow as tf


def cosine_beta_schedule(k: int, s: float = 0.008) -> tf.Tensor:
    steps = k + 1
    t = tf.linspace(0.0, float(k), steps)
    f = tf.math.cos(((t / float(k)) + s) / (1 + s) * math.pi / 2.0) ** 2
    alphabar = f / f[0]
    betas = 1.0 - (alphabar[1:] / alphabar[:-1])
    return tf.clip_by_value(betas, 1e-4, 1e-2)


def alpha_tables(betas: tf.Tensor):
    alphas = 1.0 - betas
    alphabars = tf.math.cumprod(alphas, 0)
    sigma_star = tf.sqrt(1.0 - alphabars)
    return alphas, alphabars, sigma_star


def make_tau_cosine(k: int, tau0: float = 1e-4):
    u = np.linspace(0, 1, k, endpoint=True)
    c = 0.5 * (1 - np.cos(np.pi * u))
    c = (c - c[0]) / (c[-1] - c[0] + 1e-12)
    tau = np.diff(np.concatenate([[0.0], c]))
    tau[0] = 0.0
    tau[1:] = np.maximum(tau[1:], tau0)
    tau = tau / np.sum(tau)
    ccum = np.cumsum(tau)
    return tau.astype(np.float32), ccum.astype(np.float32)


class TwoPhaseExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float = 1e-4, start_decay: int = 30000, total_steps: int = 60000, min_lr: float = 1e-5):
        super().__init__()
        self.base_lr = float(base_lr)
        self.start_decay = int(start_decay)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        base = tf.constant(self.base_lr, tf.float32)
        min_lr = tf.constant(self.min_lr, tf.float32)
        span = tf.maximum(1.0, tf.cast(self.total_steps - self.start_decay, tf.float32))
        tau = tf.clip_by_value((step - self.start_decay) / span, 0.0, 1.0)
        log_ratio = tf.math.log(min_lr / base)
        decayed = base * tf.exp(log_ratio * tau)
        return tf.where(step < self.start_decay, base, decayed)
