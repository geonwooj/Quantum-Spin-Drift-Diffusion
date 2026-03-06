from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..data.prototypes import build_or_load_prototype
from .schedules import alpha_tables, make_tau_cosine


@dataclass
class DriftCfg:
    k: int
    a: float = 3.0
    tau0: float = 1e-4
    lp_sigma: float = 3.0
    freeze_prototypes: bool = True


def _to_unit_hwk(x: tf.Tensor, eps: float = 1e-8, target: float = 0.6) -> tf.Tensor:
    x = tf.convert_to_tensor(x, tf.float32)
    norms = tf.norm(x, axis=-1, keepdims=True)
    mean_norm = tf.reduce_mean(norms)
    scale = target / (mean_norm + eps)
    return tf.stop_gradient(x * scale)


class DriftA_NoGain:
    def __init__(self, betas: tf.Tensor, cfg: DriftCfg):
        self.cfg = cfg
        self.betas = tf.cast(betas, tf.float32)
        self.alphas, self.alphabars, self.sigma_star = alpha_tables(self.betas)
        self.k = int(cfg.k)

        tau, ccum = make_tau_cosine(self.k, tau0=cfg.tau0)
        self.tau = tf.constant(tau, tf.float32)
        self.c_table = tf.constant(ccum, tf.float32)
        self.a = tf.Variable(float(cfg.a), dtype=tf.float32, trainable=False)
        gamma = float(self.a.numpy()) * ccum
        self.gamma_table = tf.constant(gamma.astype(np.float32), tf.float32)

        self.uhat16: tf.Variable | None = None
        self._uhat_cache: dict[tuple[int, int], tf.Tensor] = {}
        sigma_t = float(self.sigma_star[-1].numpy())
        gamma_t = float(self.gamma_table[-1].numpy())
        self.r_T = 2.0 * gamma_t / (sigma_t + 1e-12)

    def warmup_and_save_if_needed(
        self,
        flower_ds: tf.data.Dataset,
        leaf_ds: tf.data.Dataset,
        path: str | Path,
        target_count: int = 512,
    ) -> None:
        proto = build_or_load_prototype(flower_ds, leaf_ds, path, target_count=target_count)
        self.uhat16 = tf.Variable(proto, dtype=tf.float32, trainable=False)

    def _uhat_full(self, h: int, w: int) -> tf.Tensor:
        key = (int(h), int(w))
        if key in self._uhat_cache:
            return self._uhat_cache[key]
        if self.uhat16 is None:
            raise RuntimeError("uhat16 not initialized; call warmup_and_save_if_needed first.")
        u = tf.image.resize(self.uhat16[None, ...], (h, w), method="bilinear")[0]
        u = _to_unit_hwk(u)
        self._uhat_cache[key] = u
        return u

    def direction(self, x: tf.Tensor) -> tf.Tensor:
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        u = self._uhat_full(int(h), int(w))
        u_batch = tf.tile(u[None, ...], [b, 1, 1, 1])
        return tf.stop_gradient(u_batch)

    def c_t_batch(self, x: tf.Tensor, t_vec: tf.Tensor, s_vec: tf.Tensor) -> tf.Tensor:
        uhat = self.direction(x)
        g = tf.gather(self.gamma_table, tf.cast(t_vec, tf.int32))
        coeff = tf.reshape(tf.sign(s_vec) * g, [-1, 1, 1, 1])
        return coeff * uhat
