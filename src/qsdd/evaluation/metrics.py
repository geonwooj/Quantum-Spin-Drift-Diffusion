from __future__ import annotations

import numpy as np
import tensorflow as tf


def real_stats(ds: tf.data.Dataset, n_batches: int = 50):
    abs_means = []
    for i, x in enumerate(ds):
        if i >= n_batches:
            break
        abs_means.append(tf.reduce_mean(tf.abs(x)).numpy())
    return float(np.mean(abs_means)), float(np.std(abs_means))
