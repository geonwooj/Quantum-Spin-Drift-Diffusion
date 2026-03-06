from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    weights_dir: Path
    proto_path: Path

    @classmethod
    def from_root(cls, root: str | Path, run_tag: str) -> "ProjectPaths":
        root = Path(root)
        output_dir = root / "outputs" / run_tag
        checkpoint_dir = output_dir / "checkpoints"
        weights_dir = output_dir / "weights"
        proto_path = output_dir / "proto" / "uhat16_dataset_diff.npz"
        data_dir = root / "data"
        return cls(root, data_dir, output_dir, checkpoint_dir, weights_dir, proto_path)
