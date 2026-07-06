#!/usr/bin/env python3
"""Download and unpack ADE20K/ADEChallengeData2016.

The expected output layout is:

    /data/cpc/root/dataset/ADE20K/ADEChallengeData2016/
      images/{training,validation}
      annotations/{training,validation}
      objectInfo150.txt

The script intentionally uses only the Python standard library so it can be
run before installing extra dataset tooling.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ADE20K_URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download ADE20K/ADEChallengeData2016")
    parser.add_argument("--output-root", default="/data/cpc/root/dataset/ADE20K")
    parser.add_argument("--url", default=ADE20K_URL)
    parser.add_argument("--keep-zip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_wget(url: str, zip_path: Path) -> None:
    wget = shutil.which("wget")
    if wget is None:
        raise RuntimeError("wget is not available. Install wget or download the zip manually.")
    cmd = [wget, "-c", "--show-progress", "-O", str(zip_path), url]
    subprocess.run(cmd, check=True)


def verify_layout(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "images" / "training",
        dataset_dir / "images" / "validation",
        dataset_dir / "annotations" / "training",
        dataset_dir / "annotations" / "validation",
        dataset_dir / "objectInfo150.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("ADE20K extraction is incomplete. Missing:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_root / "ADEChallengeData2016"
    zip_path = output_root / "ADEChallengeData2016.zip"

    if dataset_dir.exists() and not args.force:
        verify_layout(dataset_dir)
        print(f"ADE20K already exists: {dataset_dir}")
        return

    if not zip_path.exists() or args.force:
        print(f"Downloading ADE20K from {args.url}")
        run_wget(args.url, zip_path)

    print(f"Extracting {zip_path} to {output_root}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_root)

    verify_layout(dataset_dir)
    if not args.keep_zip:
        zip_path.unlink()
    print(f"ADE20K ready: {dataset_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
