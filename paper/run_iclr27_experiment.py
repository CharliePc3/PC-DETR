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


def build_command(args: argparse.Namespace, recipe: dict) -> tuple[list[str], dict]:
    model = recipe["model"]
    optimization = recipe["optimization"]
    cdn = recipe["cdn"]
    dense = recipe["dense_o2o"]
    track = recipe["tracks"][args.track]
    ema = recipe["ema"]

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
        "--dec-level-n-points", *map(str, model["dec_level_n_points"]),
        "--no-lite-refpoint-refine",
        "--bbox-refine-mode", model["bbox_refine_mode"],
        "--scale-routing",
        "--scale-routing-mode", model["scale_routing_mode"],
        "--p5-attention-bias", str(model["p5_attention_bias"]),
        "--out-feature-indexes", *map(str, model["out_feature_indexes"]),
        "--projector-scale", *model["projector_scale"],
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
                "--projector-c2f-blocks", "3", "3", "3",
                "--projector-resample-share", "none",
            ]
        )
    if args.use_ema:
        command.extend(
            ["--use-ema", "--ema-decay", str(ema["decay"]), "--ema-tau", str(ema["tau"])]
        )

    resolved_recipe = {
        "dataset": {"name": "COCO", "subset": track["subset"], "data_root": str(args.data_root.resolve())},
        "model": {**model, "selected_system": args.system},
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("analysis", "system"), required=True)
    parser.add_argument("--system", choices=("sdsr_v40", "msp"), required=True)
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
    args = parser.parse_args()
    if args.detector_init_seed is None:
        args.detector_init_seed = args.seed + 1000
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.track == "system" and not args.confirm_full_coco:
        parser.error("system track requires --confirm-full-coco")
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
        return 0

    require_clean_worktree()
    if not args.data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {args.data_root}")
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to reuse existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest = {
        "schema_version": "1.0.0",
        "recipe_id": recipe["recipe_id"],
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_clean_at_launch": True,
        "track": args.track,
        "system": args.system,
        "seed": args.seed,
        "detector_init_seed": args.detector_init_seed,
        "expected_epochs": recipe["tracks"][args.track]["epochs"],
        "ema_enabled": args.use_ema,
        "command": command,
        "resolved_recipe": resolved_recipe,
        "environment": runtime_environment(),
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
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
