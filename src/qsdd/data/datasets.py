from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

ALLOW_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class DatasetConfig:
    image_size: int = 128
    channels: int = 3
    batch_size: int = 16
    seed: int = 42
    shuffle_buffer: int = 8192
    deterministic: bool = True


def find_image_root(base_dir: str | Path) -> Path:
    base_dir = Path(base_dir)
    candidates = [
        "dataset/train",
        "train",
        "car_ims",
        "cars_train/cars_train",
        ".",
        "images",
        "image",
        "jpg",
        "dataset/jpg",
    ]
    for rel in candidates:
        p = base_dir / rel
        if p.is_dir():
            files = list(p.iterdir())
            if any(f.suffix.lower() in ALLOW_EXTENSIONS for f in files if f.is_file()):
                return p.resolve()

    best_dir: Path | None = None
    best_cnt = 0
    for root, _, files in os.walk(base_dir):
        cnt = sum(1 for f in files if Path(f).suffix.lower() in ALLOW_EXTENSIONS)
        if cnt > best_cnt:
            best_dir = Path(root)
            best_cnt = cnt

    if best_dir is None or best_cnt == 0:
        raise ValueError(f"No images found under {base_dir}")
    return best_dir.resolve()


def make_dataset(root_dir: str | Path, cfg: DatasetConfig) -> tf.data.Dataset:
    root_dir = str(root_dir)
    ds = image_dataset_from_directory(
        root_dir,
        labels=None,
        label_mode=None,
        image_size=(cfg.image_size, cfg.image_size),
        batch_size=cfg.batch_size,
        shuffle=False,
        interpolation="bilinear",
        seed=cfg.seed,
    )

    if cfg.deterministic:
        opt = tf.data.Options()
        opt.experimental_deterministic = True
        ds = ds.with_options(opt)

    ds = ds.map(
        lambda x: tf.cast(x, tf.float32) / 127.5 - 1.0,
        num_parallel_calls=1 if cfg.deterministic else tf.data.AUTOTUNE,
    )
    ds = ds.unbatch()
    ds = ds.shuffle(
        cfg.shuffle_buffer,
        seed=cfg.seed,
        reshuffle_each_iteration=False,
    )
    ds = ds.batch(cfg.batch_size, drop_remainder=True)
    return ds.prefetch(1 if cfg.deterministic else tf.data.AUTOTUNE)
