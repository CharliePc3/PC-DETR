#!/usr/bin/env python3
"""Launch one experiment from the immutable ICLR 2027 paper recipe."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
RECIPE_PATH = PAPER_DIR / "ICLR27_RECIPE.json"
SCALE_INTERFACES = ("p4", "p345")
FROZEN_MSP_C2F_DEPTH = 3


def load_recipe() -> dict:
    with RECIPE_PATH.open() as handle:
        recipe = json.load(handle)
    # These are experiment-contract invariants, not tunable defaults.
    assert recipe["model"]["sdsr_projector_type"] == "sdsr_v40_p4_learnable"
    assert recipe["model"]["projector_scale"] == ["P3", "P4", "P5"]
    assert recipe["model"]["dec_level_n_points"] == [2, 3, 1]
    assert recipe["model"]["backbone_register_border_tokens"] == 0
    assert recipe["cdn"]["dn_total_query_budget"] == 300
    assert recipe["dense_o2o"]["mixup_prob"] == 0.0
    assert recipe["optimization"]["multi_scale_stop_epoch"] == -1
    return recipe


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def require_clean_worktree() -> None:
    status = git_value("status", "--porcelain", "--untracked-files=normal")
    if status:
        raise SystemExit(
            "Refusing to launch from a dirty worktree. Check out the frozen commit "
            "in a dedicated clean worktree."
        )


def resolve_scale_interface(scale_interface: str, model: dict) -> dict:
    """Resolve one pre-registered H-Validation detector scale interface."""
    if scale_interface not in SCALE_INTERFACES:
        raise ValueError(f"Unsupported scale interface: {scale_interface}")

    frozen_levels = tuple(model["projector_scale"])
    frozen_points = tuple(model["dec_level_n_points"])
    if len(frozen_levels) != len(frozen_points):
        raise ValueError("Frozen projector levels and decoder points are misaligned")

    if scale_interface == "p345":
        selected_indexes = tuple(range(len(frozen_levels)))
    else:
        # P4-only is a strict projection of the frozen P3/P4/P5 mapping. It is
        # intentionally not an independently configurable decoder recipe.
        selected_indexes = (frozen_levels.index("P4"),)

    projector_scale = [frozen_levels[index] for index in selected_indexes]
    dec_level_n_points = [frozen_points[index] for index in selected_indexes]
    return {
        "name": scale_interface,
        "projector_scale": projector_scale,
        "dec_level_n_points": dec_level_n_points,
        "msp_c2f_blocks": [FROZEN_MSP_C2F_DEPTH] * len(projector_scale),
    }


def build_command(args: argparse.Namespace, recipe: dict) -> tuple[list[str], dict]:
    model = recipe["model"]
    optimization = recipe["optimization"]
    cdn = recipe["cdn"]
    dense = recipe["dense_o2o"]
    track = recipe["tracks"][args.track]
    ema = recipe["ema"]
    scale = resolve_scale_interface(
        getattr(args, "scale_interface", "p345"), model
    )

    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "run_coco_subset.py"),
        "--subset", track["subset"],
        "--data-root", str(args.data_root),
        "--epochs", str(track["epochs"]),
        "--batch-size", str(optimization["batch_size"]),
        "--grad-accum-steps", str(optimization["grad_accum_steps"]),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--world-size", str(optimization["world_size"]),
        "--resolution", str(optimization["resolution"]),
        "--dec-layers", str(model["dec_layers"]),
        "--num-queries", str(model["num_queries"]),
        "--num-select", str(model["num_select"]),
        "--group-detr", str(model["group_detr"]),
        "--dec-n-points", str(model["dec_n_points"]),
        "--dec-level-n-points", *map(str, scale["dec_level_n_points"]),
        "--no-lite-refpoint-refine",
        "--bbox-refine-mode", model["bbox_refine_mode"],
        "--scale-routing",
        "--scale-routing-mode", model["scale_routing_mode"],
        "--p5-attention-bias", str(model["p5_attention_bias"]),
        "--out-feature-indexes", *map(str, model["out_feature_indexes"]),
        "--projector-scale", *scale["projector_scale"],
        "--projector-p5-mode", model["projector_p5_mode"],
        "--multi-scale",
        "--expanded-scales",
        "--multi-scale-stop-epoch", str(optimization["multi_scale_stop_epoch"]),
        "--aug-preset", optimization["augmentation_preset"],
        "--lr", str(optimization["lr"]),
        "--lr-encoder", str(optimization["lr_encoder"]),
        "--lr-drop", str(track["lr_drop"]),
        "--weight-decay", str(optimization["weight_decay"]),
        "--lr-vit-layer-decay", str(optimization["lr_vit_layer_decay"]),
        "--lr-component-decay", str(optimization["lr_component_decay"]),
        "--use-cdn",
        "--dn-number", str(cdn["dn_number"]),
        "--dn-total-query-budget", str(cdn["dn_total_query_budget"]),
        "--dn-box-noise-scale", str(cdn["dn_box_noise_scale"]),
        "--dn-label-noise-scale", str(cdn["dn_label_noise_scale"]),
        "--dn-loss-coef", str(cdn["dn_loss_coef"]),
        "--use-dense-o2o",
        "--dense-o2o-mode", dense["mode"],
        "--dense-o2o-start-epoch", str(dense["start_epoch"]),
        "--dense-o2o-image-stop-epoch", str(track["dense_o2o_image_stop_epoch"]),
        "--dense-o2o-copyblend-stop-epoch", str(track["dense_o2o_copyblend_stop_epoch"]),
        "--dense-o2o-mosaic-prob", str(dense["mosaic_prob"]),
        "--dense-o2o-mixup-prob", str(dense["mixup_prob"]),
        "--dense-o2o-copyblend-prob", str(dense["copyblend_prob"]),
        "--dense-o2o-copyblend-area-threshold", str(dense["copyblend_area_threshold"]),
        "--dense-o2o-copyblend-num-objects", str(dense["copyblend_num_objects"]),
        "--dense-o2o-copyblend-expand-ratios", *map(str, dense["copyblend_expand_ratios"]),
        "--backbone-register-border-tokens", "0",
        "--feature-adapter", model["feature_adapter"],
        "--online-refine-mode", model["online_refine_mode"],
        "--eval-max-dets", "100",
        "--seed", str(args.seed),
        "--detector-init-seed", str(args.detector_init_seed),
        "--output-dir", str(args.output_dir),
    ]

    if args.system == "sdsr_v40":
        command.extend(
            [
                "--projector-type", model["sdsr_projector_type"],
                "--sdsr-rank-channels", str(model["sdsr_rank_channels"]),
                "--sdsr-detail-channels", str(model["sdsr_detail_channels"]),
                "--sdsr-cross-scale-mode", model["sdsr_cross_scale_mode"],
                "--sdsr-cross-scale-rank", str(model["sdsr_cross_scale_rank"]),
                "--no-sdsr-use-local-reassembly",
                "--no-sdsr-use-directional-guide",
                "--sdsr-use-phase-downsample",
            ]
        )
    else:
        command.extend(
            [
                "--projector-type", model["msp_projector_type"],
                "--projector-source-mode", "mask",
                "--projector-c2f-blocks", *map(str, scale["msp_c2f_blocks"]),
                "--projector-resample-share", "none",
            ]
        )
    if args.use_ema:
        command.extend(
            ["--use-ema", "--ema-decay", str(ema["decay"]), "--ema-tau", str(ema["tau"])]
        )

    resolved_recipe = {
        "dataset": {"name": "COCO", "subset": track["subset"], "data_root": str(args.data_root.resolve())},
        "model": {
            **model,
            "selected_system": args.system,
            "scale_interface": scale["name"],
            "projector_scale": scale["projector_scale"],
            "dec_level_n_points": scale["dec_level_n_points"],
            "msp_c2f_blocks": (
                scale["msp_c2f_blocks"] if args.system == "msp" else None
            ),
        },
        "optimization": optimization,
        "cdn": cdn,
        "dense_o2o": {**dense, **track},
        "ema": {"enabled": args.use_ema, "decay": ema["decay"], "tau": ema["tau"]},
    }
    return command, resolved_recipe


def runtime_environment() -> dict:
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        torch_version = cuda_version = gpu = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda": cuda_version,
        "gpu": gpu,
    }


def manifest_payload(
    args: argparse.Namespace,
    recipe: dict,
    command: list[str],
    resolved_recipe: dict,
    *,
    dry_run: bool,
) -> dict:
    timestamp_key = "generated_at_utc" if dry_run else "launched_at_utc"
    payload = {
        "schema_version": "1.0.0",
        "manifest_kind": "dry_run" if dry_run else "run",
        "recipe_id": recipe["recipe_id"],
        "git_commit": git_value("rev-parse", "HEAD"),
        "track": args.track,
        "system": args.system,
        "scale_interface": args.scale_interface,
        "seed": args.seed,
        "detector_init_seed": args.detector_init_seed,
        "expected_epochs": recipe["tracks"][args.track]["epochs"],
        "ema_enabled": args.use_ema,
        "command": command,
        "resolved_recipe": resolved_recipe,
        "environment": runtime_environment(),
        timestamp_key: datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        payload["git_clean_at_generation"] = True
    else:
        payload["git_clean_at_launch"] = True
    return payload


def write_new_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise SystemExit(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("analysis", "system"), required=True)
    parser.add_argument("--system", choices=("sdsr_v40", "msp"), required=True)
    parser.add_argument(
        "--scale-interface",
        choices=SCALE_INTERFACES,
        default="p345",
        help="Pre-registered H-Validation detector interface (default: p345).",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--detector-init-seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--use-ema", action=argparse.BooleanOptionalAction, default=True,
        help="Paper tables default to EMA enabled; --no-use-ema is efficiency-only.",
    )
    parser.add_argument("--confirm-full-coco", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-manifest",
        type=Path,
        default=None,
        help="With --dry-run, save the resolved audit manifest without launching.",
    )
    args = parser.parse_args()
    if args.detector_init_seed is None:
        args.detector_init_seed = args.seed + 1000
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.track == "system" and not args.confirm_full_coco:
        parser.error("system track requires --confirm-full-coco")
    if args.track != "analysis" and args.scale_interface != "p345":
        parser.error("--scale-interface p4 is limited to the H-Validation analysis track")
    if args.dry_run_manifest is not None and not args.dry_run:
        parser.error("--dry-run-manifest requires --dry-run")
    return args


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir
    args.output_dir = args.output_dir.resolve()
    recipe = load_recipe()
    command, resolved_recipe = build_command(args, recipe)
    print(shlex.join(command))
    if args.dry_run:
        if args.dry_run_manifest is not None:
            require_clean_worktree()
            manifest = manifest_payload(
                args, recipe, command, resolved_recipe, dry_run=True
            )
            write_new_json(args.dry_run_manifest, manifest)
            print(args.dry_run_manifest.resolve())
        return 0

    require_clean_worktree()
    if not args.data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {args.data_root}")
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to reuse existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest = manifest_payload(
        args, recipe, command, resolved_recipe, dry_run=False
    )
    manifest_path = args.output_dir / "RUN_MANIFEST.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    environment = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    manifest["return_code"] = completed.returncode
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
