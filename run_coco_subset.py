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
        "idx" + "-".join(str(index) for index in args.out_feature_indexes),
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
    if args.feature_adapter != "none":
        adapter_tag = {
            "residual_ln_1x1": "faresln1x1",
            "ln_1x1": "faln1x1",
        }[args.feature_adapter]
        run_tags.append(adapter_tag)
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
    if args.use_budgeted_sa:
        budgets_tag = "-".join(str(budget) for budget in args.sa_total_budgets)
        run_tags.extend(["bsa" + budgets_tag, f"e{args.sa_start_epoch}-{args.sa_stop_epoch}"])
    if args.use_dense_o2o:
        run_tags.extend(
            [
                f"denseo2o-{args.dense_o2o_mode}",
                f"e{args.dense_o2o_start_epoch}-{args.dense_o2o_image_stop_epoch}-{args.dense_o2o_copyblend_stop_epoch}",
            ]
        )

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
    parser.add_argument(
        "--detector-pretrain-exclude-projector",
        action="store_true",
        help="When loading detector_pretrain_weights, also skip backbone.0.projector.* so the projector is reinitialized.",
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
    parser.add_argument(
        "--out-feature-indexes",
        type=int,
        nargs="+",
        default=[2, 5, 8, 11],
        help="Zero-based DINOv3 block indexes whose intermediate features are fused by the projector.",
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
        "--use-budgeted-sa",
        action="store_true",
        help="Use scale-budgeted extra positives only on intermediate decoder outputs during training.",
    )
    parser.add_argument("--sa-start-epoch", type=int, default=0, help="First epoch using budgeted SA (inclusive).")
    parser.add_argument("--sa-stop-epoch", type=int, default=0, help="First epoch returning to strict aux O2O.")
    parser.add_argument(
        "--sa-total-budgets",
        type=int,
        nargs=3,
        default=[6, 7, 9],
        metavar=("SMALL", "MEDIUM", "LARGE"),
        help="Total positives per GT across all DETR groups for each object scale.",
    )
    parser.add_argument(
        "--sa-area-thresholds",
        type=float,
        nargs=2,
        default=[32**2, 96**2],
        metavar=("SMALL_MAX", "MEDIUM_MAX"),
        help="Pixel-area boundaries used to classify target scales.",
    )
    parser.add_argument("--use-dense-o2o", action="store_true")
    parser.add_argument("--dense-o2o-mode", default="image", choices=("image", "enhanced"))
    parser.add_argument("--dense-o2o-start-epoch", type=int, default=2)
    parser.add_argument("--dense-o2o-image-stop-epoch", type=int, default=12)
    parser.add_argument("--dense-o2o-copyblend-stop-epoch", type=int, default=21)
    parser.add_argument("--dense-o2o-mosaic-prob", type=float, default=0.5)
    parser.add_argument("--dense-o2o-mixup-prob", type=float, default=0.5)
    parser.add_argument("--dense-o2o-copyblend-prob", type=float, default=0.5)
    parser.add_argument("--dense-o2o-copyblend-area-threshold", type=float, default=100.0)
    parser.add_argument("--dense-o2o-copyblend-num-objects", type=int, default=3)
    parser.add_argument("--dense-o2o-copyblend-expand-ratios", type=float, nargs=2, default=[0.1, 0.25])
    parser.add_argument(
        "--backbone-register-border-tokens",
        type=int,
        default=0,
        help="Patch-token border added around the image inside the DINOv3 backbone and cropped before projector.",
    )
    parser.add_argument("--backbone-register-fill", default="randn", choices=("randn", "rand", "zero"))
    parser.add_argument("--backbone-register-noise-std", type=float, default=1.0)
    parser.add_argument(
        "--feature-adapter",
        default="none",
        choices=("none", "residual_ln_1x1", "ln_1x1"),
        help="Optional lightweight adapter applied to DINOv3 feature maps before the projector.",
    )
    parser.add_argument("--feature-adapter-init-scale", type=float, default=1.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    effective_epochs = args.epochs or DEFAULT_EPOCHS[args.subset]
    if not args.out_feature_indexes:
        raise ValueError("out_feature_indexes must contain at least one block index.")
    if args.out_feature_indexes != sorted(set(args.out_feature_indexes)):
        raise ValueError("out_feature_indexes must be unique and in ascending order.")
    if args.out_feature_indexes[0] < 0 or args.out_feature_indexes[-1] >= 12:
        raise ValueError("DINOv3-S has 12 blocks, so out_feature_indexes must be in [0, 11].")
    if args.use_budgeted_sa:
        if args.sa_start_epoch < 0 or not args.sa_start_epoch < args.sa_stop_epoch <= effective_epochs:
            raise ValueError("Budgeted SA requires 0 <= sa_start_epoch < sa_stop_epoch <= epochs.")
        if any(budget < args.group_detr for budget in args.sa_total_budgets):
            raise ValueError("Every SA total budget must be at least group_detr.")
        if args.sa_area_thresholds[0] <= 0 or args.sa_area_thresholds[0] >= args.sa_area_thresholds[1]:
            raise ValueError("SA area thresholds must be positive and increasing.")
    if args.use_dense_o2o:
        if args.use_budgeted_sa:
            raise ValueError("Dense O2O and Budgeted SA are independent experiments and cannot be enabled together.")
        if args.batch_size * args.grad_accum_steps < 4 and args.dense_o2o_mosaic_prob > 0:
            raise ValueError("Dense O2O Mosaic requires a local effective batch size of at least 4.")
        if not 0 <= args.dense_o2o_start_epoch < args.dense_o2o_image_stop_epoch <= effective_epochs:
            raise ValueError("Dense O2O requires 0 <= start < image_stop <= epochs.")
        if args.dense_o2o_mode == "enhanced" and not (
            args.dense_o2o_start_epoch < args.dense_o2o_copyblend_stop_epoch <= effective_epochs
        ):
            raise ValueError("Dense O2O requires start < copyblend_stop <= epochs.")
        for name in ("mosaic_prob", "mixup_prob", "copyblend_prob"):
            probability = getattr(args, f"dense_o2o_{name}")
            if not 0 <= probability <= 1:
                raise ValueError(f"dense_o2o_{name} must be in [0, 1].")
        if args.dense_o2o_copyblend_num_objects < 1:
            raise ValueError("dense_o2o_copyblend_num_objects must be positive.")
        if (
            args.dense_o2o_copyblend_expand_ratios[0] < 0
            or args.dense_o2o_copyblend_expand_ratios[0] > args.dense_o2o_copyblend_expand_ratios[1]
        ):
            raise ValueError("Dense O2O CopyBlend expansion ratios must be non-negative and increasing.")
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

    pretrain_exclude_keys = None
    if args.detector_pretrain_weights:
        pretrain_exclude_keys = ["backbone.0.encoder.*"]
        if args.detector_pretrain_exclude_projector:
            pretrain_exclude_keys.append("backbone.0.projector.*")

    model = RFDETRDINOv3(
        encoder=DEFAULT_ENCODER,
        pretrained_encoder=args.pretrained_encoder,
        pretrain_weights=args.detector_pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        resolution=args.resolution,
        dec_layers=args.dec_layers,
        num_queries=args.num_queries,
        num_select=args.num_select,
        group_detr=args.group_detr,
        out_feature_indexes=args.out_feature_indexes,
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
        feature_adapter=args.feature_adapter,
        feature_adapter_init_scale=args.feature_adapter_init_scale,
    )

    log_main(f"project_root={PROJECT_ROOT}")
    log_main(f"subset_path={subset_path}")
    log_main(f"output_dir={output_dir}")
    log_main(f"model=RFDETRDINOv3, encoder={DEFAULT_ENCODER}")
    log_main(f"pretrained_encoder={args.pretrained_encoder}")
    log_main(f"detector_pretrain_weights={args.detector_pretrain_weights}")
    if args.detector_pretrain_weights:
        log_main(f"detector_pretrain_exclude_keys={pretrain_exclude_keys}")
    log_main(f"distributed_world_size={args.world_size}")
    log_main(f"dist_url={args.dist_url}")
    log_main(f"sync_bn={args.sync_bn}")
    log_main(f"seed={args.seed}")
    log_main(f"resolution={args.resolution}")
    log_main(f"dec_layers={args.dec_layers}")
    log_main(f"num_queries={args.num_queries}")
    log_main(f"num_select={args.num_select}")
    log_main(f"group_detr={args.group_detr}")
    log_main(f"out_feature_indexes={args.out_feature_indexes}")
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
    log_main(f"use_budgeted_sa={args.use_budgeted_sa}")
    log_main(f"sa_epoch_window=[{args.sa_start_epoch}, {args.sa_stop_epoch})")
    log_main(f"sa_total_budgets={args.sa_total_budgets}")
    log_main(f"sa_area_thresholds={args.sa_area_thresholds}")
    log_main(f"use_dense_o2o={args.use_dense_o2o}")
    log_main(f"dense_o2o_mode={args.dense_o2o_mode}")
    log_main(
        "dense_o2o_epoch_windows="
        f"image[{args.dense_o2o_start_epoch}, {args.dense_o2o_image_stop_epoch}), "
        f"copyblend[{args.dense_o2o_start_epoch}, {args.dense_o2o_copyblend_stop_epoch})"
    )
    log_main(
        "dense_o2o_probabilities="
        f"mosaic={args.dense_o2o_mosaic_prob}, mixup={args.dense_o2o_mixup_prob}, "
        f"copyblend={args.dense_o2o_copyblend_prob}"
    )
    log_main(
        "dense_o2o_copyblend="
        f"area_threshold={args.dense_o2o_copyblend_area_threshold}, "
        f"num_objects={args.dense_o2o_copyblend_num_objects}, "
        f"expand_ratios={args.dense_o2o_copyblend_expand_ratios}"
    )
    log_main(f"backbone_register_border_tokens={args.backbone_register_border_tokens}")
    log_main(f"backbone_register_fill={args.backbone_register_fill}")
    log_main(f"backbone_register_noise_std={args.backbone_register_noise_std}")
    log_main(f"feature_adapter={args.feature_adapter}")
    log_main(f"feature_adapter_init_scale={args.feature_adapter_init_scale}")

    model.train(
        dataset_file="coco",
        coco_path=str(subset_path),
        dataset_dir=str(subset_path),
        epochs=effective_epochs,
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
        out_feature_indexes=args.out_feature_indexes,
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
        use_budgeted_sa=args.use_budgeted_sa,
        sa_start_epoch=args.sa_start_epoch,
        sa_stop_epoch=args.sa_stop_epoch,
        sa_total_budgets=tuple(args.sa_total_budgets),
        sa_area_thresholds=tuple(args.sa_area_thresholds),
        use_dense_o2o=args.use_dense_o2o,
        dense_o2o_mode=args.dense_o2o_mode,
        dense_o2o_start_epoch=args.dense_o2o_start_epoch,
        dense_o2o_image_stop_epoch=args.dense_o2o_image_stop_epoch,
        dense_o2o_copyblend_stop_epoch=args.dense_o2o_copyblend_stop_epoch,
        dense_o2o_mosaic_prob=args.dense_o2o_mosaic_prob,
        dense_o2o_mixup_prob=args.dense_o2o_mixup_prob,
        dense_o2o_copyblend_prob=args.dense_o2o_copyblend_prob,
        dense_o2o_copyblend_area_threshold=args.dense_o2o_copyblend_area_threshold,
        dense_o2o_copyblend_num_objects=args.dense_o2o_copyblend_num_objects,
        dense_o2o_copyblend_expand_ratios=tuple(args.dense_o2o_copyblend_expand_ratios),
        register_border_tokens=args.backbone_register_border_tokens,
        register_fill=args.backbone_register_fill,
        register_noise_std=args.backbone_register_noise_std,
        feature_adapter=args.feature_adapter,
        feature_adapter_init_scale=args.feature_adapter_init_scale,
        progress_bar=args.progress_bar,
    )


if __name__ == "__main__":
    main()
