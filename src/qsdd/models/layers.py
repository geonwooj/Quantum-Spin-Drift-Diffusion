from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import activations, layers


class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self, groups: int = 32, axis: int = -1, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.gamma = self.add_weight(shape=(dim,), initializer="ones", trainable=True)
        self.beta = self.add_weight(shape=(dim,), initializer="zeros", trainable=True)

    def call(self, inputs, training: bool = False):
        n, h, w, c = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        g = tf.minimum(self.groups, c)
        x = tf.reshape(inputs, [n, h, w, g, c // g])
        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [n, h, w, c])
        return self.gamma * x + self.beta


class ResidualBlock(layers.Layer):
    def __init__(self, width: int, name: str | None = None):
        super().__init__(name=name)
        self.width = width
        self.gn = GroupNormalization(groups=32, axis=-1, epsilon=1e-5, name=f"{name}_gn")
        self.conv1 = layers.Conv2D(width, 3, padding="same", activation=activations.swish, name=f"{name}_conv1")
        self.conv2 = layers.Conv2D(width, 3, padding="same", name=f"{name}_conv2")
        self.proj = layers.Conv2D(width, 1, name=f"{name}_proj")
        self.time_proj = layers.Dense(width, name=f"{name}_timeproj")

    def call(self, x, cond, training: bool = False):
        residual = x
        x = self.gn(x, training=training)
        x = self.conv1(x)
        x = x + self.time_proj(cond)[:, None, None, :]
        x = self.conv2(x)
        if residual.shape[-1] != self.width:
            residual = self.proj(residual)
        return layers.Add()([x, residual])


class SpatialSelfAttention(layers.Layer):
    def __init__(self, num_heads: int = 4, dropout: float = 0.0, window_size: int = 8, name: str | None = None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.dropout = dropout
        self.window_size = window_size

    def build(self, input_shape):
        c = int(input_shape[-1])
        key_dim = max(16, c // self.num_heads)
        self.norm = layers.LayerNormalization(epsilon=1e-5, name=f"{self.name}_ln")
        try:
            self.mha = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=key_dim,
                dropout=self.dropout,
                output_shape=c,
                name=f"{self.name}_mha",
            )
            self.use_proj = False
        except TypeError:
            self.mha = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=key_dim,
                dropout=self.dropout,
                name=f"{self.name}_mha",
            )
            self.proj = layers.Dense(c, name=f"{self.name}_proj")
            self.use_proj = True

    def _mha_tokens(self, tokens, training: bool = False):
        x = self.norm(tokens)
        out = self.mha(x, x, training=training)
        if getattr(self, "use_proj", False):
            out = self.proj(out)
        return tokens + out

    def call(self, x, training: bool = False):
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        ws = self.window_size
        with tf.control_dependencies([
            tf.debugging.assert_equal(h % ws, 0),
            tf.debugging.assert_equal(w % ws, 0),
        ]):
            h_tiles = h // ws
            w_tiles = w // ws
            x_blocks = tf.reshape(x, [b, h_tiles, ws, w_tiles, ws, c])
            x_blocks = tf.transpose(x_blocks, [0, 1, 3, 2, 4, 5])
            x_blocks = tf.reshape(x_blocks, [-1, ws * ws, c])
            y_blocks = self._mha_tokens(x_blocks, training=training)
            y_blocks = tf.reshape(y_blocks, [b, h_tiles, w_tiles, ws, ws, c])
            y_blocks = tf.transpose(y_blocks, [0, 1, 3, 2, 4, 5])
            return tf.reshape(y_blocks, [b, h, w, c])
