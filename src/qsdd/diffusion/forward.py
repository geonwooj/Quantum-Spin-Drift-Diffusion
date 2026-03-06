from __future__ import annotations

import tensorflow as tf


def make_eta_target(eps: tf.Tensor, r_map: tf.Tensor) -> tf.Tensor:
    return eps + r_map


def make_noisy_input(x0: tf.Tensor, eps: tf.Tensor, r_map: tf.Tensor, alphabars: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    alphabar_t = tf.gather(alphabars, t)
    sqrt_ab = tf.sqrt(alphabar_t)[:, None, None, None]
    sqrt_1m = tf.sqrt(1.0 - alphabar_t)[:, None, None, None]
    return sqrt_ab * x0 + sqrt_1m * (eps + r_map)
