from __future__ import annotations

from pathlib import Path
import subprocess


def run_kaggle_download(dataset: str, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(output_dir),
            "--unzip",
            "-q",
        ],
        check=True,
    )
