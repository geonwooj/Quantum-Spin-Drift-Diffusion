from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import tensorflow as tf

from ..diffusion.forward import make_eta_target, make_noisy_input
from ..diffusion.losses import compute_reweighted_eta_loss
from ..diffusion.reverse import sample_images_tf
from ..diffusion.schedules import alpha_tables, cosine_beta_schedule
from ..models.unet import build_model
from ..utils.io import ensure_dir, save_json
from .checkpoint import make_checkpoint, restore_latest_checkpoint
from .config import TrainConfig
from .ema import EMAHelper


class QSDDTrainer:
    def __init__(self, cfg: TrainConfig, paths, logger=None):
        self.cfg = cfg
        self.paths = paths
        self.logger = logger

        ensure_dir(paths.output_dir)
        ensure_dir(paths.checkpoint_dir)
        ensure_dir(paths.weights_dir)
        ensure_dir(paths.proto_path.parent)
        save_json(asdict(cfg), Path(paths.output_dir) / "train_config.json")

        self.betas = cosine_beta_schedule(cfg.k)
        self.alphas, self.alphabars, self.sigma_star = alpha_tables(self.betas)
        self.model = build_model(cfg.image_size, cfg.channels)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
        self.rng = tf.random.Generator.from_seed(cfg.seed_t_eps)
        self.step_var = tf.Variable(0, dtype=tf.int64, name="global_step")
        self.ema = EMAHelper(self.model) if cfg.use_ema else None

    def _make_checkpoint(self):
        return make_checkpoint(self.model, self.optimizer, self.step_var, self.rng)

    def _build_batch(self, flower_batch: tf.Tensor, leaf_batch: tf.Tensor):
        x0 = tf.concat([flower_batch, leaf_batch], axis=0)
        s = tf.concat(
            [tf.ones([self.cfg.batch_domain], tf.float32), -tf.ones([self.cfg.batch_domain], tf.float32)],
            axis=0,
        )
        return tf.convert_to_tensor(x0, tf.float32), s

    def _compute_grad_norm(self, grads):
        grads_nonnull = [g for g in grads if g is not None]
        if not grads_nonnull:
            return tf.constant(0.0, tf.float32)
        return tf.linalg.global_norm(grads_nonnull)

    def _apply_global_clip(self, grads):
        grads_nonnull = [g for g in grads if g is not None]
        if not grads_nonnull or self.cfg.grad_clip <= 0:
            return grads
        clipped_nonnull, _ = tf.clip_by_global_norm(grads_nonnull, self.cfg.grad_clip)
        it = iter(clipped_nonnull)
        return [next(it) if g is not None else None for g in grads]

    def train(self, flower_ds16, leaf_ds16, drift):
        drift.warmup_and_save_if_needed(flower_ds16, leaf_ds16, self.paths.proto_path, target_count=512)
        ckpt = self._make_checkpoint()
        if self.cfg.resume:
            latest = restore_latest_checkpoint(ckpt, self.paths.checkpoint_dir)
            if latest and self.logger:
                self.logger.info("Resumed checkpoint: %s", latest)
            if latest and self.ema is not None:
                self.ema.sync_from_model(self.model)

        flower_iter = iter(flower_ds16.repeat())
        leaf_iter = iter(leaf_ds16.repeat())
        step = int(self.step_var.numpy())
        gn_ema = 0.0
        gn_ema_decay = 0.999
        gn_max = 0.0

        while step < self.cfg.total_steps:
            x_up = next(flower_iter)
            x_dn = next(leaf_iter)
            x0, s = self._build_batch(x_up, x_dn)
            batch_size = tf.shape(x0)[0]
            t = self.rng.uniform([batch_size], minval=0, maxval=self.cfg.k, dtype=tf.int32)
            eps = self.rng.normal(tf.shape(x0), dtype=x0.dtype)
            r_map = drift.c_t_batch(x0, t, s)
            x_t = make_noisy_input(x0, eps, r_map, self.alphabars, t)
            eta_target = make_eta_target(eps, r_map)

            with tf.GradientTape() as tape:
                eta_hat = self.model(x_t, t, s, training=True)
                loss, aux = compute_reweighted_eta_loss(
                    eta_hat=eta_hat,
                    eta_target=eta_target,
                    r_map=r_map,
                    lambda_rw=self.cfg.lambda_rw,
                )

            grads = tape.gradient(loss, self.model.trainable_variables)
            gn = self._compute_grad_norm(grads)
            grads = self._apply_global_clip(grads)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if self.ema is not None:
                self.ema.update(self.model, step)

            gnv = float(gn.numpy())
            gn_max = max(gn_max, gnv)
            gn_ema = gn_ema_decay * gn_ema + (1.0 - gn_ema_decay) * gnv

            step += 1
            self.step_var.assign(step)

            if self.logger and step % 100 == 0:
                self.logger.info(
                    "step %6d | loss %.12f | gn %.3f | gn_ema %.3f | gn_max %.3f",
                    step,
                    float(loss.numpy()),
                    gnv,
                    gn_ema,
                    gn_max,
                )

            if step % self.cfg.save_every == 0 or step == self.cfg.total_steps:
                self.save(ckpt)

        return self.model, drift, (self.betas, self.alphas, self.alphabars)

    def save(self, ckpt: tf.train.Checkpoint) -> None:
        ckpt_path = ckpt.save(str(Path(self.paths.checkpoint_dir) / "ckpt"))
        step = int(self.step_var.numpy())
        raw_w_path = Path(self.paths.weights_dir) / f"denoise_fn_step{step:07d}.weights.h5"
        self.model.save_weights(str(raw_w_path))

        if self.ema is not None:
            ema_w_path = Path(self.paths.weights_dir) / f"denoise_fn_step{step:07d}_ema.weights.h5"
            backup = self.ema.swap_into_model(self.model)
            self.model.save_weights(str(ema_w_path))
            self.ema.restore_backup(self.model, backup)

        if self.logger:
            self.logger.info("Saved checkpoint to %s", ckpt_path)

    def load_weights(self, weight_path: str | Path) -> None:
        self.model.load_weights(str(weight_path))

    def sample(self, drift, n: int = 8, s_scalar: float = 1.0):
        return sample_images_tf(
            self.model,
            drift,
            (self.betas, self.alphas, self.alphabars),
            n=n,
            s_scalar=s_scalar,
            shape=(self.cfg.image_size, self.cfg.image_size, self.cfg.channels),
        )
