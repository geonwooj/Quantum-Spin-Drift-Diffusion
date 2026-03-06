from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def _mean_16x16(ds: tf.data.Dataset, n_imgs: int) -> np.ndarray:
    sum_16 = np.zeros((16, 16, 3), dtype=np.float32)
    count = 0
    for batch in ds:
        batch_np = batch.numpy()
        for k in range(batch_np.shape[0]):
            img = batch_np[k : k + 1]
            img16 = tf.image.resize(img, (16, 16), method="area").numpy()[0]
            sum_16 += img16
            count += 1
            if count >= n_imgs:
                break
        if count >= n_imgs:
            break

    if count == 0:
        raise ValueError("No images collected while building prototype")
    return sum_16 / float(count)


def build_or_load_prototype(
    flower_ds: tf.data.Dataset,
    leaf_ds: tf.data.Dataset,
    path: str | Path,
    target_count: int = 512,
) -> np.ndarray:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        data = np.load(path, allow_pickle=True)
        return data["uhat16"].astype(np.float32)

    u_up16 = _mean_16x16(flower_ds, target_count)
    u_down16 = _mean_16x16(leaf_ds, target_count)
    diff = (u_up16 - u_down16).astype(np.float32)
    np.savez(path, uhat16=diff, u_up16=u_up16, u_down16=u_down16)
    return diff
