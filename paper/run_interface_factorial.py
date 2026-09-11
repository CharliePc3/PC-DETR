#!/usr/bin/env python3
"""Resolve or launch the preregistered MSP detector-interface 2x2 factorial."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
RECIPE_PATH = PAPER_DIR / "INTERFACE_FACTORIAL_RECIPE.json"
H_LAUNCHER_PATH = PAPER_DIR / "run_iclr27_experiment.py"
RUN_IDS = ("M0", "M1", "M2", "M3")


def _load_h_launcher():
    spec = importlib.util.spec_from_file_location("iclr27_h_launcher", H_LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_recipe() -> dict:
    recipe = json.loads(RECIPE_PATH.read_text())
    assert recipe["base_commit"] == "2b40a235d5ca3a40d5e39f5060efa0570a12a5ad"
    assert tuple(recipe["runs"]) == RUN_IDS
    prior_sum = sum(recipe["factors"]["p4_depth_prior"]["f11_heavy_fixed"])
    assert abs(prior_sum - 4.0) < 1e-6
    return recipe


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_launchable(recipe: dict) -> None:
    if git_value("status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("Refusing to launch the factorial from a dirty worktree.")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", recipe["base_commit"], "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise SystemExit("Experiment HEAD does not descend from the registered base commit.")


def replace_values(command: list[str], flag: str, old_count: int, values: list[str]) -> None:
    index = command.index(flag) + 1
    command[index : index + old_count] = values


def resolved_run(
    run_id: str,
    args: argparse.Namespace,
    recipe: dict,
) -> tuple[list[str], dict]:
    h_launcher = _load_h_launcher()
    h_recipe = h_launcher.load_recipe()
    h_args = argparse.Namespace(
        track="analysis",
        system="msp",
        scale_interface="p345",
        data_root=args.data_root,
        output_dir=args.output_root / run_id,
        seed=recipe["seed"],
        detector_init_seed=recipe["detector_init_seed"],
        num_workers=args.num_workers,
        device="cuda",
        use_ema=True,
    )
    command, resolved = h_launcher.build_command(h_args, h_recipe)
    factors = recipe["runs"][run_id]
    interface = recipe["factors"]["detector_interface"][factors["detector_interface"]]
    prior = recipe["factors"]["p4_depth_prior"][factors["p4_depth_prior"]]

    replace_values(command, "--projector-scale", 3, interface["projector_scale"])
    replace_values(
        command,
        "--dec-level-n-points",
        3,
        [str(value) for value in interface["dec_level_n_points"]],
    )
    replace_values(
        command,
        "--projector-distill-level-weights",
        3,
        [str(value) for value in interface["projector_distill_level_weights"]],
    )
    replace_values(
        command,
        "--projector-c2f-blocks",
        3,
        [str(value) for value in interface["msp_c2f_blocks"]],
    )
    command.extend(["--pretrained-encoder", str(args.pretrained_encoder)])
    if prior is not None:
        command.extend(["--projector-p4-depth-prior", *map(str, prior)])

    resolved["model"].update(
        {
            "scale_interface": factors["detector_interface"].lower(),
            "projector_scale": interface["projector_scale"],
            "dec_level_n_points": interface["dec_level_n_points"],
            "projector_distill_level_weights": interface[
                "projector_distill_level_weights"
            ],
            "msp_c2f_blocks": interface["msp_c2f_blocks"],
            "projector_p4_depth_prior": prior,
        }
    )
    resolved["dataset"]["data_root"] = str(args.data_root.resolve())
    resolved["experiment_factors"] = factors
    return command, resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_provenance(args: argparse.Namespace) -> dict:
    subset = args.data_root / "medium"
    annotations = subset / "annotations"
    train_annotations = annotations / "instances_train2017.json"
    val_annotations = annotations / "instances_val2017.json"
    dinov3_root = Path(os.environ.get("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3"))
    return {
        "coco_medium_root": str(subset.resolve()),
        "train_annotation_sha256": sha256(train_annotations),
        "val_annotation_sha256": sha256(val_annotations),
        "train_image_links": sum(1 for _ in (subset / "train2017").iterdir()),
        "val_image_links": sum(1 for _ in (subset / "val2017").iterdir()),
        "dinov3_repo": str(dinov3_root.resolve()),
        "dinov3_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=dinov3_root, text=True
        ).strip(),
        "pretrained_encoder": str(args.pretrained_encoder.resolve()),
        "pretrained_encoder_sha256": sha256(args.pretrained_encoder),
    }


def runtime_environment() -> dict:
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version = cuda_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda": cuda_version,
    }


def gpu_snapshot() -> list[dict] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    rows = []
    for line in output.splitlines():
        index, name, memory_used, memory_total, utilization = (
            value.strip() for value in line.split(",", 4)
        )
        rows.append(
            {
                "index": index,
                "name": name,
                "memory_used_mib": int(memory_used),
                "memory_total_mib": int(memory_total),
                "utilization_percent": int(utilization),
            }
        )
    return rows


def build_audit(args: argparse.Namespace, recipe: dict) -> dict:
    runs = {}
    # With the default two-GPU plan, use opposite factorial diagonals so neither
    # scientific main effect is confounded with physical GPU identity.
    device_indexes = (0, 1, 1, 0) if len(args.devices) == 2 else tuple(range(4))
    for index, run_id in enumerate(RUN_IDS):
        device = args.devices[device_indexes[index] % len(args.devices)]
        command, resolved = resolved_run(run_id, args, recipe)
        runs[run_id] = {
            "device": device,
            "output_dir": str((args.output_root / run_id).resolve()),
            "command": command,
            "resolved_recipe": resolved,
        }
    return {
        "schema_version": "1.0.0",
        "manifest_kind": "factorial_preflight",
        "recipe_id": recipe["recipe_id"],
        "base_commit": recipe["base_commit"],
        "git_head": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_clean": not bool(git_value("status", "--porcelain", "--untracked-files=normal")),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": runtime_environment(),
        "gpu_snapshot": gpu_snapshot(),
        "input_provenance": input_provenance(args),
        "runs": runs,
        "contrasts": recipe["contrasts"],
    }


def launch(args: argparse.Namespace, recipe: dict, audit: dict) -> int:
    require_launchable(recipe)
    if args.output_root.exists():
        raise SystemExit(f"Refusing to reuse output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    pending = list(RUN_IDS)
    active = {}
    completed = []
    for run_id in RUN_IDS:
        run = audit["runs"][run_id]
        output_dir = Path(run["output_dir"])
        output_dir.mkdir()
        manifest = {
            "schema_version": "1.0.0",
            "manifest_kind": "run",
            "recipe_id": recipe["recipe_id"],
            "git_commit": audit["git_head"],
            "base_commit": recipe["base_commit"],
            "track": "analysis",
            "system": "msp",
            "run_id": run_id,
            "scale_interface": recipe["runs"][run_id]["detector_interface"].lower(),
            "seed": recipe["seed"],
            "detector_init_seed": recipe["detector_init_seed"],
            "expected_epochs": 24,
            "ema_enabled": True,
            "git_clean_at_launch": True,
            "command": run["command"],
            "resolved_recipe": run["resolved_recipe"],
            "environment": audit["environment"],
            "input_provenance": audit["input_provenance"],
            "queued_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = output_dir / "RUN_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        completed.append((run_id, manifest, manifest_path))

    manifests = {run_id: (manifest, path) for run_id, manifest, path in completed}
    failed = False
    while pending or active:
        for device in args.devices:
            if device in active:
                continue
            next_index = next(
                (
                    index
                    for index, run_id in enumerate(pending)
                    if audit["runs"][run_id]["device"] == device
                ),
                None,
            )
            if next_index is None:
                continue
            run_id = pending.pop(next_index)
            run = audit["runs"][run_id]
            train_log = (Path(run["output_dir"]) / "train.log").open("w")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            source_path = str(REPO_ROOT / "src")
            environment["PYTHONPATH"] = source_path + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            )
            process = subprocess.Popen(
                run["command"],
                cwd=REPO_ROOT,
                env=environment,
                stdout=train_log,
                stderr=subprocess.STDOUT,
            )
            manifest, manifest_path = manifests[run_id]
            manifest["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            active[device] = (run_id, process, train_log)
            print(f"{run_id}: launched on physical GPU {device}", flush=True)

        finished_devices = []
        for device, (run_id, process, train_log) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            train_log.close()
            manifest, manifest_path = manifests[run_id]
            manifest["return_code"] = return_code
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            failed |= return_code != 0
            finished_devices.append(device)
            print(f"{run_id}: return_code={return_code}", flush=True)
        for device in finished_devices:
            del active[device]
        if active:
            time.sleep(2)
    return int(failed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-encoder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["0", "1"],
        help="One to four distinct physical GPU indexes; runs are queued round-robin.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument(
        "--confirm-launch",
        action="store_true",
        help="Launch all M0-M3 runs; without this flag only resolve the preflight audit.",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.pretrained_encoder = args.pretrained_encoder.resolve()
    args.output_root = args.output_root.resolve()
    if args.audit_output is not None:
        args.audit_output = args.audit_output.resolve()
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 1 <= len(args.devices) <= 4 or len(set(args.devices)) != len(args.devices):
        parser.error("--devices must name one to four distinct GPUs")
    return args


def main() -> int:
    args = parse_args()
    recipe = load_recipe()
    audit = build_audit(args, recipe)
    rendered = json.dumps(audit, indent=2, sort_keys=True)
    print(rendered)
    if args.audit_output is not None:
        if args.audit_output.exists():
            raise SystemExit(f"Refusing to overwrite audit file: {args.audit_output}")
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(rendered + "\n")
    if not args.confirm_launch:
        print("DRY RUN ONLY: pass --confirm-launch after A-F approval to launch all M0-M3.")
        return 0
    return launch(args, recipe, audit)


if __name__ == "__main__":
    raise SystemExit(main())
