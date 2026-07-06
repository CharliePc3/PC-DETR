import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3")
os.environ.setdefault("DINOV3_WEIGHTS_DIR", str(PROJECT_ROOT / "weights" / "dinov3"))

from rfdetr.datasets.aug_config import (
    AUG_AERIAL,
    AUG_AGGRESSIVE,
    AUG_CONSERVATIVE,
    AUG_CONFIG,
    AUG_INDUSTRIAL,
)
from rfdetr_dinov3 import RFDETRDINOv3

DEFAULT_EPOCHS = {
    "smoke": 1,
    "overfit": 30,
    "quick": 15,
    "medium": 12,
    "strong": 24,
    "full": 24,
}
DEFAULT_ENCODER = "dinov3_small"
DEFAULT_RF_DETR_MEDIUM_WEIGHTS = PROJECT_ROOT.parent / "RF-DETR" / "fast" / "rf-detr-medium.pth"
AUG_PRESETS = {
    "default": AUG_CONFIG,
    "conservative": AUG_CONSERVATIVE,
    "aggressive": AUG_AGGRESSIVE,
    "aerial": AUG_AERIAL,
    "industrial": AUG_INDUSTRIAL,
    "none": {},
}


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_rank():
    return int(os.environ.get("RANK", "0"))


def is_main_process():
    return get_rank() == 0


def setup_distributed_device(device):
    if device != "cuda" or "LOCAL_RANK" not in os.environ:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed CUDA training was requested, but CUDA is not available.")
    torch.cuda.set_device(get_local_rank())


def set_seed(seed):
    rank_seed = seed + get_rank()
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def log_main(message):
    if is_main_process():
        print(message, flush=True)


def make_output_dir(args):
    if args.output_dir is not None:
        return Path(args.output_dir)

    if args.resume:
        resume_path = Path(args.resume)
        return resume_path.parent if resume_path.suffix else resume_path

    scale_tag = "".join(args.projector_scale).lower()
    run_tags = [
        f"coco_{args.subset}",
        DEFAULT_ENCODER.replace("_", ""),
        f"r{args.resolution}",
        f"d{args.dec_layers}",
        f"q{args.num_queries}",
        f"g{args.group_detr}",
        f"{scale_tag}",
    ]
    if args.multi_scale:
        run_tags.append("ms")
    if args.expanded_scales:
        run_tags.append("expanded")
    if args.backbone_register_border_tokens > 0:
        run_tags.extend(
            [
                f"regb{args.backbone_register_border_tokens}",
                args.backbone_register_fill,
                f"std{args.backbone_register_noise_std:g}",
            ]
        )
    if args.aug_preset != "default":
        run_tags.append(f"aug{args.aug_preset}")
    if args.use_cdn:
        neg_tag = "neg" if args.dn_negative else "pos"
        run_tags.extend(
            [
                "cdn",
                f"dn{args.dn_number}",
                f"box{args.dn_box_noise_scale:g}",
                f"lbl{args.dn_label_noise_scale:g}",
                f"loss{args.dn_loss_coef:g}",
                neg_tag,
            ]
        )
    else:
        run_tags.append("nocdn")

    base_dir = PROJECT_ROOT / "output" / "_".join(run_tags)
    if not base_dir.exists():
        return base_dir

    run_idx = 1
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_run{run_idx:02d}")
        if not candidate.exists():
            return candidate
        run_idx += 1


def parse_args():
    parser = argparse.ArgumentParser("Run RF-DETR-DINOv3 on prepared COCO subsets.")
    parser.add_argument("--subset", default="smoke", choices=("smoke", "overfit", "quick", "medium", "strong", "full"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--sync-bn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help="Accepted for compatibility with torch.distributed.launch; torchrun uses LOCAL_RANK.",
    )
    parser.add_argument("--data-root", default="/data/cpc/root/dataset/COCO_RFDETR_TEST")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument(
        "--detector-pretrain-weights",
        default=None,
        help="Optional full detector checkpoint loaded after DINOv3 backbone init. "
        f"For the RF-DETR Medium backbone-swap ablation, use {DEFAULT_RF_DETR_MEDIUM_WEIGHTS}.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--expanded-scales", action="store_true")
    parser.add_argument("--aug-preset", default="default", choices=tuple(AUG_PRESETS))
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-select", type=int, default=300)
    parser.add_argument("--group-detr", type=int, default=13)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--lr-drop", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-vit-layer-decay", type=float, default=0.8)
    parser.add_argument("--lr-component-decay", type=float, default=0.7)
    parser.add_argument(
        "--projector-scale",
        nargs="+",
        default=["P4"],
        choices=("P3", "P4", "P5"),
        help="Feature levels produced by MultiScaleProjector and consumed by the decoder.",
    )
    
    parser.add_argument("--eval-max-dets", type=int, default=100)
    parser.add_argument("--use-cdn", action="store_true")
    parser.add_argument("--dn-number", type=int, default=50)
    parser.add_argument("--dn-label-noise-scale", type=float, default=0.5)
    parser.add_argument("--dn-box-noise-scale", type=float, default=0.6)
    parser.add_argument("--no-dn-negative", dest="dn_negative", action="store_false")
    parser.set_defaults(dn_negative=True)
    parser.add_argument("--dn-loss-coef", type=float, default=0.5)
    parser.add_argument("--dn-neg-loss-coef", type=float, default=1.0)
    parser.add_argument(
        "--backbone-register-border-tokens",
        type=int,
        default=0,
        help="Patch-token border added around the image inside the DINOv3 backbone and cropped before projector.",
    )
    parser.add_argument("--backbone-register-fill", default="randn", choices=("randn", "rand", "zero"))
    parser.add_argument("--backbone-register-noise-std", type=float, default=1.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_distributed_device(args.device)
    set_seed(args.seed)

    data_root = Path(args.data_root)
    if args.subset == "full":
        if data_root.name == "COCO":
            subset_path = data_root
        elif (data_root / "COCO").exists():
            subset_path = data_root / "COCO"
        else:
            subset_path = Path("/data/cpc/root/dataset/COCO")
    else:
        subset_path = data_root / args.subset
    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRDINOv3(
        encoder=DEFAULT_ENCODER,
        pretrained_encoder=args.pretrained_encoder,
        pretrain_weights=args.detector_pretrain_weights,
        pretrain_exclude_keys=["backbone.0.encoder.*"] if args.detector_pretrain_weights else None,
        resolution=args.resolution,
        dec_layers=args.dec_layers,
        num_queries=args.num_queries,
        num_select=args.num_select,
        group_detr=args.group_detr,
        projector_scale=args.projector_scale,
        positional_encoding_size=args.resolution // 16,
        use_cdn=args.use_cdn,
        dn_number=args.dn_number,
        dn_label_noise_scale=args.dn_label_noise_scale,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_negative=args.dn_negative,
        register_border_tokens=args.backbone_register_border_tokens,
        register_fill=args.backbone_register_fill,
        register_noise_std=args.backbone_register_noise_std,
    )

    log_main(f"project_root={PROJECT_ROOT}")
    log_main(f"subset_path={subset_path}")
    log_main(f"output_dir={output_dir}")
    log_main(f"model=RFDETRDINOv3, encoder={DEFAULT_ENCODER}")
    log_main(f"pretrained_encoder={args.pretrained_encoder}")
    log_main(f"detector_pretrain_weights={args.detector_pretrain_weights}")
    if args.detector_pretrain_weights:
        log_main("detector_pretrain_exclude_keys=['backbone.0.encoder.*']")
    log_main(f"distributed_world_size={args.world_size}")
    log_main(f"dist_url={args.dist_url}")
    log_main(f"sync_bn={args.sync_bn}")
    log_main(f"seed={args.seed}")
    log_main(f"resolution={args.resolution}")
    log_main(f"dec_layers={args.dec_layers}")
    log_main(f"num_queries={args.num_queries}")
    log_main(f"num_select={args.num_select}")
    log_main(f"group_detr={args.group_detr}")
    log_main(f"projector_scale={args.projector_scale}")
    log_main(f"lr={args.lr}")
    log_main(f"lr_encoder={args.lr_encoder}")
    log_main(f"lr_drop={args.lr_drop}")
    log_main(f"warmup_epochs={args.warmup_epochs}")
    log_main(f"weight_decay={args.weight_decay}")
    log_main(f"lr_vit_layer_decay={args.lr_vit_layer_decay}")
    log_main(f"lr_component_decay={args.lr_component_decay}")
    log_main(f"multi_scale={args.multi_scale}")
    log_main(f"expanded_scales={args.expanded_scales}")
    log_main(f"aug_preset={args.aug_preset}")
    log_main(f"projector_type={getattr(model.model_config, 'projector_type', 'multiscale')}")
    log_main(f"eval_max_dets={args.eval_max_dets}")
    log_main(f"use_cdn={args.use_cdn}")
    log_main(f"dn_number={args.dn_number}")
    log_main(f"dn_negative={args.dn_negative}")
    log_main(f"backbone_register_border_tokens={args.backbone_register_border_tokens}")
    log_main(f"backbone_register_fill={args.backbone_register_fill}")
    log_main(f"backbone_register_noise_std={args.backbone_register_noise_std}")

    model.train(
        dataset_file="coco",
        coco_path=str(subset_path),
        dataset_dir=str(subset_path),
        epochs=args.epochs or DEFAULT_EPOCHS[args.subset],
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        device=args.device,
        world_size=args.world_size,
        dist_url=args.dist_url,
        sync_bn=args.sync_bn,
        num_workers=args.num_workers,
        output_dir=str(output_dir),
        seed=args.seed,
        resume=args.resume,
        pretrained_encoder=args.pretrained_encoder,
        eval=args.eval_only,
        resolution=args.resolution,
        dec_layers=args.dec_layers,
        num_queries=args.num_queries,
        num_select=args.num_select,
        group_detr=args.group_detr,
        projector_scale=args.projector_scale,
        positional_encoding_size=args.resolution // 16,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_drop=args.lr_drop,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        lr_vit_layer_decay=args.lr_vit_layer_decay,
        lr_component_decay=args.lr_component_decay,
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        aug_config=AUG_PRESETS[args.aug_preset],
        use_ema=args.use_ema,
        tensorboard=args.tensorboard,
        run_test=args.run_test,
        eval_max_dets=args.eval_max_dets,
        use_cdn=args.use_cdn,
        dn_number=args.dn_number,
        dn_label_noise_scale=args.dn_label_noise_scale,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_negative=args.dn_negative,
        dn_loss_coef=args.dn_loss_coef,
        dn_neg_loss_coef=args.dn_neg_loss_coef,
        register_border_tokens=args.backbone_register_border_tokens,
        register_fill=args.backbone_register_fill,
        register_noise_std=args.backbone_register_noise_std,
        progress_bar=args.progress_bar,
    )


if __name__ == "__main__":
    main()
