from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, activations, layers

from .embeddings import sinusoidal_time_embedding
from .layers import ResidualBlock, SpatialSelfAttention


class UNetDenoiser(Model):
    def __init__(self, use_attn_bot: bool = True, use_attn_out: bool = False):
        super().__init__()
        self.use_attn_bot = use_attn_bot
        self.use_attn_out = use_attn_out
        self.time_mlp = tf.keras.Sequential([
            layers.Dense(256, activation=activations.swish),
            layers.Dense(512, activation=activations.swish),
        ])
        self.spin_mlp = tf.keras.Sequential([
            layers.Dense(128, activation=activations.swish),
            layers.Dense(512, activation=activations.swish),
        ])

        self.e1 = ResidualBlock(64, "e1")
        self.down1 = layers.AveragePooling2D(2)
        self.e2 = ResidualBlock(128, "e2")
        self.down2 = layers.AveragePooling2D(2)
        self.e3 = ResidualBlock(256, "e3")
        self.down3 = layers.AveragePooling2D(2)
        self.e4 = ResidualBlock(512, "e4")
        self.down4 = layers.AveragePooling2D(2)
        self.b1 = ResidualBlock(512, "b1")

        if self.use_attn_bot:
            self.attn_bot = SpatialSelfAttention(num_heads=4, window_size=8, name="attn_bot")

        self.up4 = layers.UpSampling2D(2, interpolation="bilinear")
        self.d4 = ResidualBlock(512, "d4")
        self.up3 = layers.UpSampling2D(2, interpolation="bilinear")
        self.d3 = ResidualBlock(256, "d3")
        self.up2 = layers.UpSampling2D(2, interpolation="bilinear")
        self.d2 = ResidualBlock(128, "d2")
        self.up1 = layers.UpSampling2D(2, interpolation="bilinear")
        self.d1 = ResidualBlock(64, "d1")
        self.final = layers.Conv2D(3, 1, kernel_initializer="zeros", name="final_conv", dtype="float32")

    def call(self, x_t, t, s_scalar, training: bool = False):
        temb = self.time_mlp(sinusoidal_time_embedding(t, 128))
        semb = self.spin_mlp(tf.reshape(s_scalar, [-1, 1]))
        cond = temb + semb

        e1 = self.e1(x_t, cond, training=training)
        p1 = self.down1(e1)
        e2 = self.e2(p1, cond, training=training)
        p2 = self.down2(e2)
        e3 = self.e3(p2, cond, training=training)
        p3 = self.down3(e3)
        e4 = self.e4(p3, cond, training=training)
        p4 = self.down4(e4)
        b = self.b1(p4, cond, training=training)
        if self.use_attn_bot:
            b = self.attn_bot(b, training=training)

        u4 = self.up4(b)
        d4 = self.d4(tf.concat([u4, e4], axis=-1), cond, training=training)
        u3 = self.up3(d4)
        d3 = self.d3(tf.concat([u3, e3], axis=-1), cond, training=training)
        u2 = self.up2(d3)
        d2 = self.d2(tf.concat([u2, e2], axis=-1), cond, training=training)
        u1 = self.up1(d2)
        d1 = self.d1(tf.concat([u1, e1], axis=-1), cond, training=training)
        return self.final(d1)


def build_model(image_size: int = 128, channels: int = 3) -> UNetDenoiser:
    model = UNetDenoiser(use_attn_bot=True, use_attn_out=False)
    _ = model(
        tf.zeros([1, image_size, image_size, channels]),
        tf.zeros([1], tf.int32),
        tf.zeros([1], tf.float32),
        training=False,
    )
    return model
