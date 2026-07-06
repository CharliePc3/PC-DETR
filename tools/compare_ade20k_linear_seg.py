#!/usr/bin/env python3
"""Collect ADE20K linear segmentation probe results into a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare ADE20K linear segmentation probe runs")
    parser.add_argument("--root", default="output/ade20k_linear_seg")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        best = summary.get("best") or {}
        run_dir = summary_path.parent
        row = {
            "run": run_dir.name,
            "epoch": best.get("epoch"),
            "miou": best.get("miou"),
            "mean_acc": best.get("mean_acc"),
            "pixel_acc": best.get("pixel_acc"),
            "valid_classes": best.get("valid_classes"),
            "loss": best.get("loss"),
            "pretrained_encoder": (summary.get("args") or {}).get("pretrained_encoder"),
            "path": str(run_dir),
        }
        rows.append(row)

    output_path = Path(args.output).expanduser().resolve() if args.output else root / "comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "epoch", "miou", "mean_acc", "pixel_acc", "valid_classes", "loss", "pretrained_encoder", "path"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(output_path)
    for row in rows:
        miou = row["miou"]
        miou_str = f"{float(miou):.4f}" if miou is not None else "NA"
        print(f"{row['run']}: best_mIoU={miou_str} epoch={row['epoch']}")


if __name__ == "__main__":
    main()
