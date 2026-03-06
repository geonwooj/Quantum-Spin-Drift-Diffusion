from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Sequence

from .datasets import ALLOW_EXTENSIONS


DEFAULT_PLANT_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
]


def build_leaf_domain_subset(
    plant_root: str | Path,
    output_root: str | Path,
    target_count: int = 8000,
    classes: Sequence[str] | None = None,
    seed: int = 42,
) -> Path:
    plant_root = Path(plant_root)
    output_root = Path(output_root)
    output_dir = output_root / "all"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        f for f in output_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOW_EXTENSIONS
    ]
    if len(existing) >= target_count:
        return output_root

    classes = list(classes or DEFAULT_PLANT_CLASSES)
    all_imgs: list[Path] = []
    for cls in classes:
        cls_dir = plant_root / cls
        if not cls_dir.is_dir():
            continue
        files = [
            f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOW_EXTENSIONS
        ]
        all_imgs.extend(files)

    rng = random.Random(seed)
    rng.shuffle(all_imgs)
    all_imgs = all_imgs[:target_count]

    for i, src in enumerate(all_imgs):
        dst = output_dir / f"leaf_{i:05d}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
    return output_root
