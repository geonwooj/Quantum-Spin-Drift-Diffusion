from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def show_grid(imgs, cols: int = 4, title: str = "samples"):
    imgs = (imgs.numpy() * 127.5 + 127.5).astype(np.uint8)
    rows = int(np.ceil(len(imgs) / cols))
    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i, im in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_grid_autoscale(imgs, cols: int = 4, title: str = "", q: float = 0.01):
    x = imgs.numpy()
    lo = np.quantile(x, q)
    hi = np.quantile(x, 1.0 - q)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    x = (x * 255).astype(np.uint8)

    rows = int(np.ceil(len(x) / cols))
    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i, im in enumerate(x):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im)
        plt.axis("off")
    plt.suptitle(title + f"  (autoscale q={q})")
    plt.tight_layout()
    plt.show()


def show_snapshots_autoscale(snaps_dict, title_prefix: str = "", cols: int = 4, q: float = 0.01):
    for t in sorted(snaps_dict.keys(), reverse=True):
        show_grid_autoscale(snaps_dict[t], cols=cols, title=f"{title_prefix} x_{t}", q=q)
