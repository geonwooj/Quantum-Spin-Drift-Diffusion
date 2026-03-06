from __future__ import annotations

import tensorflow as tf


def ddpm_reverse_mean_eps(y_t, t_int: int, eps_hat, betas, alphas, alphabars):
    beta_t = betas[t_int]
    alpha_t = alphas[t_int]
    alphabar_t = alphabars[t_int]
    return (y_t - (beta_t / tf.sqrt(1.0 - alphabar_t)) * eps_hat) / tf.sqrt(alpha_t)


def sample_images_tf(model, drift, tables, n: int = 8, s_scalar: float = +1.0, shape=(128, 128, 3)):
    betas, alphas, alphabars = tables
    k = int(betas.shape[0])
    s_b = tf.ones([n], tf.float32) * float(s_scalar)
    z = tf.random.normal([n, *shape], dtype=tf.float32)
    t_T = tf.fill([n], k - 1)
    r_T = drift.c_t_batch(tf.zeros_like(z), t_T, s_b)
    sqrt1m_T = tf.sqrt(1.0 - alphabars[-1])
    x = sqrt1m_T * (z + r_T)

    for t_int in reversed(range(k)):
        t_vec = tf.fill([n], t_int)
        r_t = drift.c_t_batch(tf.zeros_like(x), t_vec, s_b)
        alphabar_t = alphabars[t_int]
        sqrt1m_t = tf.sqrt(1.0 - alphabar_t)
        y_t = x - sqrt1m_t * r_t
        eta_hat = model(x, t_vec, s_b, training=False)
        eps_hat = eta_hat - r_t
        y_mean = ddpm_reverse_mean_eps(y_t, t_int, eps_hat, betas, alphas, alphabars)

        if t_int > 0:
            alphabar_tm1 = alphabars[t_int - 1]
            beta_t = betas[t_int]
            beta_tilde = beta_t * (1.0 - alphabar_tm1) / (1.0 - alphabar_t)
            z = tf.random.normal(tf.shape(x), dtype=x.dtype)
            y_tm1 = y_mean + tf.sqrt(beta_tilde) * z
            t_prev = tf.fill([n], t_int - 1)
            r_tm1 = drift.c_t_batch(tf.zeros_like(x), t_prev, s_b)
            sqrt1m_tm1 = tf.sqrt(1.0 - alphabar_tm1)
            x = y_tm1 + sqrt1m_tm1 * r_tm1
        else:
            x = y_mean
    return tf.clip_by_value(x, -1.0, 1.0)


def sample_with_snapshots_tf(
    model,
    drift,
    tables,
    n: int = 8,
    s_scalar: float = +1.0,
    shape=(128, 128, 3),
    snapshot_ts=(999, 700, 500, 300, 100),
    seed: int = 777,
    init_mode: str = "your_init",
    return_stats: bool = True,
):
    betas, alphas, alphabars = tables
    k = int(betas.shape[0])
    snapshot_ts = sorted(set(int(t) for t in snapshot_ts if 0 <= t < k), reverse=True)

    tf.random.set_seed(int(seed))
    s_b = tf.ones([n], tf.float32) * float(s_scalar)
    z = tf.random.normal([n, *shape], dtype=tf.float32)
    t_T = tf.fill([n], k - 1)
    r_T = drift.c_t_batch(tf.zeros_like(z), t_T, s_b)
    sigma_T = tf.sqrt(1.0 - alphabars[-1])

    if init_mode == "ddpm_init":
        x = sigma_T * (z + r_T)
    else:
        x = z + sigma_T * r_T

    snaps = {}
    stats = {}

    def _stat(x_):
        absx = tf.abs(x_)
        return {
            "min": float(tf.reduce_min(x_).numpy()),
            "max": float(tf.reduce_max(x_).numpy()),
            "mean_abs": float(tf.reduce_mean(absx).numpy()),
            "sat_gt1": float(tf.reduce_mean(tf.cast(absx > 1.0, tf.float32)).numpy()),
            "sat_gt2": float(tf.reduce_mean(tf.cast(absx > 2.0, tf.float32)).numpy()),
        }

    for t_int in reversed(range(k)):
        t_vec = tf.fill([n], t_int)
        r_t = drift.c_t_batch(tf.zeros_like(x), t_vec, s_b)
        alphabar_t = alphabars[t_int]
        sigma_t = tf.sqrt(1.0 - alphabar_t)
        y_t = x - sigma_t * r_t
        eta_hat = model(x, t_vec, s_b, training=False)
        eps_hat = eta_hat - r_t
        y_mean = ddpm_reverse_mean_eps(y_t, t_int, eps_hat, betas, alphas, alphabars)

        if t_int > 0:
            alphabar_tm1 = alphabars[t_int - 1]
            beta_t = betas[t_int]
            beta_tilde = beta_t * (1.0 - alphabar_tm1) / (1.0 - alphabar_t)
            z2 = tf.random.normal(tf.shape(x), dtype=x.dtype)
            y_tm1 = y_mean + tf.sqrt(beta_tilde) * z2
            t_prev = tf.fill([n], t_int - 1)
            r_tm1 = drift.c_t_batch(tf.zeros_like(x), t_prev, s_b)
            sigma_tm1 = tf.sqrt(1.0 - alphabar_tm1)
            x = y_tm1 + sigma_tm1 * r_tm1
        else:
            x = y_mean

        if t_int in snapshot_ts:
            snaps[t_int] = tf.identity(x)
            if return_stats:
                stats[t_int] = _stat(x)

    return (snaps, stats) if return_stats else snaps
