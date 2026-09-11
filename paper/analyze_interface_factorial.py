#!/usr/bin/env python3
"""Aggregate M0-M3 RESULTS.json files and compute preregistered contrasts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


RUN_IDS = ("M0", "M1", "M2", "M3")
CONTRASTS = {
    "M1-M0_prior_with_p5": ("M1", "M0"),
    "M3-M2_prior_without_p5": ("M3", "M2"),
    "M2-M0_remove_p5_without_prior": ("M2", "M0"),
    "M3-M1_remove_p5_with_prior": ("M3", "M1"),
}


def subtract(left, right):
    if isinstance(left, dict):
        return {key: subtract(left[key], right[key]) for key in left}
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return round(float(left) - float(right), 6)
    raise TypeError(f"Cannot subtract {type(left).__name__} and {type(right).__name__}")


def metric_payload(result: dict) -> dict:
    payload = {}
    for branch in ("raw", "ema"):
        values = result["metrics"][branch]
        if values is None:
            raise SystemExit(f"Missing required {branch} metrics in {result['experiment']['id']}")
        payload[branch] = {
            "best": values["best"],
            "final": values["final"],
            "last5_AP": values["last5_AP"],
        }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    results = {
        run_id: json.loads((output_root / run_id / "RESULTS.json").read_text())
        for run_id in RUN_IDS
    }
    for run_id, result in results.items():
        if result["experiment"]["status"] != "complete":
            raise SystemExit(f"{run_id} is not complete")
        if result["experiment"]["recipe_id"] != "iclr27-msp-interface-factorial-2026-09-11":
            raise SystemExit(f"{run_id} has the wrong recipe_id")

    metrics = {run_id: metric_payload(result) for run_id, result in results.items()}
    contrasts = {
        name: subtract(metrics[left], metrics[right])
        for name, (left, right) in CONTRASTS.items()
    }
    interaction_prior_effect = subtract(
        contrasts["M3-M2_prior_without_p5"],
        contrasts["M1-M0_prior_with_p5"],
    )
    interaction_p5_effect = subtract(
        contrasts["M3-M1_remove_p5_with_prior"],
        contrasts["M2-M0_remove_p5_without_prior"],
    )
    if interaction_prior_effect != interaction_p5_effect:
        raise SystemExit("Algebraically equivalent interaction calculations disagree")
    payload = {
        "schema_version": "1.0.0",
        "recipe_id": "iclr27-msp-interface-factorial-2026-09-11",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit": "COCO AP points (0-100)",
        "runs": {
            run_id: {
                "metrics": metrics[run_id],
                "efficiency": results[run_id]["efficiency"],
                "frozen_commit": results[run_id]["experiment"]["frozen_commit"],
            }
            for run_id in RUN_IDS
        },
        "contrasts": contrasts,
        "interaction": {
            "expression": "(M3-M2)-(M1-M0) = (M3-M1)-(M2-M0)",
            "values": interaction_prior_effect,
        },
    }
    output = args.output or output_root / "FACTORIAL_RESULTS.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
