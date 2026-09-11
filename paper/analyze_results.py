#!/usr/bin/env python3
"""Build the canonical ICLR 2027 RESULTS.json from an experiment log."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


METRIC_NAMES = ("AP", "AP50", "AP75", "APS", "APM", "APL")
MEMORY_RE = re.compile(r"max mem:\s*(\d+)")


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_epoch_rows(path: Path) -> list[dict]:
    rows = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if "epoch" in row:
                rows[int(row["epoch"])] = row
    return [rows[epoch] for epoch in sorted(rows)]


def coco_metrics(values: list[float]) -> dict:
    if len(values) < len(METRIC_NAMES):
        raise SystemExit("COCO metric array has fewer than six entries")
    if any(value < 0 or value > 1 for value in values[:6]):
        raise SystemExit("Expected source COCO metrics in the 0-1 range")
    return {name: round(float(value) * 100, 6) for name, value in zip(METRIC_NAMES, values)}


def metric_branch(rows: list[dict], key: str) -> dict | None:
    usable = [row for row in rows if row.get(key)]
    if not usable:
        return None
    best = max(usable, key=lambda row: row[key][0])
    final = max(usable, key=lambda row: int(row["epoch"]))
    last_five = sorted(usable, key=lambda row: int(row["epoch"]))[-5:]
    return {
        "best_epoch": int(best["epoch"]) + 1,
        "best": coco_metrics(best[key]),
        "final": coco_metrics(final[key]),
        "last5_AP": round(sum(float(row[key][0]) for row in last_five) / len(last_five) * 100, 6),
    }


def duration_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def peak_memory(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    matches = MEMORY_RE.findall(path.read_text(errors="replace"))
    return max(map(int, matches)) if matches else None


def first_existing(*paths: Path) -> str | None:
    for path in paths:
        if path.is_file():
            return str(path.resolve())
    return None


def validate(result: dict) -> None:
    required = {"schema_version", "experiment", "protocol", "metrics", "efficiency", "artifacts", "environment", "provenance"}
    if set(result) != required:
        raise SystemExit(f"RESULTS.json root keys differ from schema: {set(result) ^ required}")
    if result["schema_version"] != "1.0.0":
        raise SystemExit("Unsupported RESULTS.json schema version")
    commit = result["experiment"]["frozen_commit"]
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SystemExit("Manifest does not contain a full frozen commit SHA")
    if result["metrics"]["raw"] is None:
        raise SystemExit("No raw COCO metrics found")
    if result["protocol"]["ema_enabled"] and result["experiment"]["status"] == "complete" and result["metrics"]["ema"] is None:
        raise SystemExit("Complete EMA-enabled run has no EMA metrics")
    if not result["provenance"]["git_clean_at_launch"]:
        raise SystemExit("Experiment was not launched from a clean worktree")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument(
        "--efficiency-json",
        type=Path,
        default=None,
        help="Optional FLOPs-only output from tools/benchmark_detector.py.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "RUN_MANIFEST.json"
    log_path = output_dir / "log.txt"
    if not manifest_path.is_file() or not log_path.is_file():
        raise SystemExit("Expected RUN_MANIFEST.json and log.txt in --output-dir")
    manifest = load_json(manifest_path)
    rows = load_epoch_rows(log_path)
    expected_epochs = int(manifest["expected_epochs"])
    completed_epochs = len(rows)
    raw = metric_branch(rows, "test_coco_eval_bbox")
    ema = metric_branch(rows, "ema_test_coco_eval_bbox")
    status = "complete" if completed_epochs >= expected_epochs else "partial"

    efficiency_profile = (
        load_json(args.efficiency_json)
        if args.efficiency_json is not None
        else None
    )
    if efficiency_profile is not None:
        if not efficiency_profile["environment"].get("latency_skipped"):
            raise SystemExit(
                "Factorial training results only accept a FLOPs-only efficiency profile "
                "created with --skip-latency."
            )
        profiled_total = efficiency_profile.get("parameters")
        logged_total = rows[-1].get("n_total_parameters") if rows else None
        if logged_total is not None and profiled_total != logged_total:
            raise SystemExit("Efficiency profile parameter count does not match training log")

    result = {
        "schema_version": "1.0.0",
        "experiment": {
            "id": output_dir.name,
            "recipe_id": manifest["recipe_id"],
            "frozen_commit": manifest["git_commit"],
            "track": manifest["track"],
            "system": manifest["system"],
            "status": status,
        },
        "protocol": {
            "seed": int(manifest["seed"]),
            "detector_init_seed": int(manifest["detector_init_seed"]),
            "expected_epochs": expected_epochs,
            "completed_epochs": completed_epochs,
            "ema_enabled": bool(manifest["ema_enabled"]),
            "recipe": manifest["resolved_recipe"],
        },
        "metrics": {"unit": "COCO AP points (0-100)", "raw": raw, "ema": ema},
        "efficiency": {
            "trainable_parameters": rows[-1].get("n_parameters") if rows else None,
            "total_parameters": rows[-1].get("n_total_parameters") if rows else None,
            "peak_train_memory_mib": peak_memory(args.train_log),
            "total_train_time_seconds": round(sum(duration_seconds(row.get("epoch_time")) for row in rows), 3) if rows else None,
            "inference_latency_ms": None,
            "gflops": (
                efficiency_profile.get("gflops")
                if efficiency_profile is not None
                else None
            ),
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "log": str(log_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "checkpoint_regular": first_existing(output_dir / "checkpoint_best_regular.pth", output_dir / "checkpoint.pth"),
            "checkpoint_ema": first_existing(output_dir / "checkpoint_best_ema.pth") if manifest["ema_enabled"] else None,
            "train_log": str(args.train_log.resolve()) if args.train_log else None,
            "efficiency_profile": (
                str(args.efficiency_json.resolve())
                if args.efficiency_json is not None
                else None
            ),
        },
        "environment": manifest["environment"],
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_clean_at_launch": bool(manifest["git_clean_at_launch"]),
            "command": manifest["command"],
        },
    }
    validate(result)
    output_path = args.output or output_dir / "RESULTS.json"
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
