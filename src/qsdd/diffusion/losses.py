from __future__ import annotations

import tensorflow as tf


def compute_reweighted_eta_loss(
    eta_hat: tf.Tensor,
    eta_target: tf.Tensor,
    r_map: tf.Tensor,
    lambda_rw: float = 1.0,
):
    err = eta_hat - eta_target
    mse_per = tf.reduce_mean(tf.square(err), axis=[1, 2, 3])
    r2_per = tf.reduce_mean(tf.reduce_sum(tf.square(r_map), axis=-1), axis=[1, 2])
    w = 1.0 / (1.0 + lambda_rw * r2_per)
    w = tf.stop_gradient(w)
    w = w / (tf.reduce_mean(w) + 1e-12)
    loss = tf.reduce_mean(w * mse_per)
    return loss, {
        "mse_per": mse_per,
        "r2_per": r2_per,
        "weights": w,
        "err": err,
    }
