from __future__ import annotations

import tensorflow as tf


def sinusoidal_time_embedding(t, dim: int = 128):
    half = dim // 2
    freq = tf.exp(tf.linspace(0.0, tf.math.log(10000.0), half) * (-1.0))
    args = tf.cast(tf.expand_dims(tf.cast(t, tf.float32), 1), tf.float32) * tf.expand_dims(freq, 0)
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    return emb
