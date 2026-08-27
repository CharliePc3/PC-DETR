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
DEFAULT_RF_DETR_MEDIUM_WEIGHTS = (
    PROJECT_ROOT.parent / "RF-DETR" / "fast" / "rf-detr-medium.pth"
)
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
        raise RuntimeError(
            "Distributed CUDA training was requested, but CUDA is not available."
        )
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
    if args.projector_type in {
        "sdsr",
        "sdsr_v2",
        "sdsr_v3",
        "sdsr_v4",
        "sdsr_v5",
        "sdsr_v6",
        "sdsr_v7",
        "sdsr_v8",
        "sdsr_v9",
        "sdsr_v10",
        "sdsr_v11",
        "sdsr_v12",
        "sdsr_v13",
        "sdsr_v14",
        "sdsr_v15",
        "sdsr_v16",
        "sdsr_v17",
        "sdsr_v18",
        "sdsr_v19",
        "sdsr_v20",
        "sdsr_v21",
        "sdsr_v22",
        "sdsr_v23",
        "sdsr_v24",
        "sdsr_v25",
        "sdsr_v26",
        "sdsr_v27",
        "sdsr_v28",
        "sdsr_v29",
        "sdsr_v30",
        "sdsr_v31",
        "sdsr_v32",
        "sdsr_v33",
        "sdsr_v34",
        "sdsr_v35",
        "sdsr_v36",
    }:
        projector_tag = {
            "sdsr": "sdsr",
            "sdsr_v2": "sdsrv2",
            "sdsr_v3": "sdsrv3",
            "sdsr_v4": "sdsrv4",
            "sdsr_v5": "sdsrv5",
            "sdsr_v6": "sdsrv6",
            "sdsr_v7": "sdsrv7",
            "sdsr_v8": "sdsrv8",
            "sdsr_v9": "sdsrv9",
            "sdsr_v10": "sdsrv10",
            "sdsr_v11": "sdsrv11",
            "sdsr_v12": "sdsrv12",
            "sdsr_v13": "sdsrv13",
            "sdsr_v14": "sdsrv14",
            "sdsr_v15": "sdsrv15",
            "sdsr_v16": "sdsrv16",
            "sdsr_v17": "sdsrv17",
            "sdsr_v18": "sdsrv18",
            "sdsr_v19": "sdsrv19",
            "sdsr_v20": "sdsrv20",
            "sdsr_v21": "sdsrv21",
            "sdsr_v22": "sdsrv22",
            "sdsr_v23": "sdsrv23",
            "sdsr_v24": "sdsrv24",
            "sdsr_v25": "sdsrv25",
            "sdsr_v26": "sdsrv26",
            "sdsr_v27": "sdsrv27",
            "sdsr_v28": "sdsrv28",
            "sdsr_v29": "sdsrv29",
            "sdsr_v30": "sdsrv30",
            "sdsr_v31": "sdsrv31",
            "sdsr_v32": "sdsrv32",
            "sdsr_v33": "sdsrv33",
            "sdsr_v34": "sdsrv34",
            "sdsr_v35": "sdsrv35",
            "sdsr_v36": "sdsrv36",
        }[args.projector_type]
        run_tags.extend([projector_tag, f"dc{args.sdsr_detail_channels}"])
        if args.projector_type in {
            "sdsr_v2",
            "sdsr_v3",
            "sdsr_v4",
            "sdsr_v5",
            "sdsr_v6",
            "sdsr_v7",
            "sdsr_v8",
            "sdsr_v9",
            "sdsr_v10",
            "sdsr_v11",
            "sdsr_v12",
            "sdsr_v13",
            "sdsr_v14",
            "sdsr_v15",
            "sdsr_v16",
            "sdsr_v17",
            "sdsr_v18",
            "sdsr_v19",
            "sdsr_v20",
            "sdsr_v21",
            "sdsr_v22",
            "sdsr_v23",
            "sdsr_v24",
            "sdsr_v25",
            "sdsr_v26",
            "sdsr_v27",
            "sdsr_v28",
            "sdsr_v29",
            "sdsr_v30",
            "sdsr_v31",
            "sdsr_v32",
            "sdsr_v33",
            "sdsr_v34",
            "sdsr_v35",
            "sdsr_v36",
        }:
            run_tags.append(f"rank{args.sdsr_rank_channels}")
        if args.projector_type in {
            "sdsr_v7",
            "sdsr_v8",
            "sdsr_v9",
            "sdsr_v10",
            "sdsr_v11",
            "sdsr_v12",
            "sdsr_v13",
            "sdsr_v14",
            "sdsr_v15",
            "sdsr_v16",
            "sdsr_v17",
            "sdsr_v18",
            "sdsr_v19",
            "sdsr_v20",
            "sdsr_v21",
            "sdsr_v22",
            "sdsr_v23",
            "sdsr_v24",
            "sdsr_v25",
            "sdsr_v26",
            "sdsr_v27",
            "sdsr_v28",
            "sdsr_v29",
            "sdsr_v30",
            "sdsr_v31",
            "sdsr_v32",
            "sdsr_v33",
            "sdsr_v34",
            "sdsr_v35",
            "sdsr_v36",
        }:
            run_tags.extend(
                [
                    f"xscale-{args.sdsr_cross_scale_mode}",
                    f"xrank{args.sdsr_cross_scale_rank}",
                ]
            )
        if not args.sdsr_use_local_reassembly:
            run_tags.append("nolocal")
        if not args.sdsr_use_directional_guide:
            run_tags.append("nodir")
        if (
            args.projector_type
            in {
                "sdsr",
                "sdsr_v3",
                "sdsr_v4",
                "sdsr_v5",
                "sdsr_v6",
                "sdsr_v7",
                "sdsr_v8",
                "sdsr_v9",
                "sdsr_v10",
                "sdsr_v11",
                "sdsr_v12",
                "sdsr_v13",
                "sdsr_v14",
                "sdsr_v15",
                "sdsr_v16",
                "sdsr_v17",
                "sdsr_v18",
                "sdsr_v19",
                "sdsr_v20",
                "sdsr_v21",
                "sdsr_v22",
                "sdsr_v23",
                "sdsr_v24",
                "sdsr_v25",
                "sdsr_v26",
                "sdsr_v27",
                "sdsr_v28",
                "sdsr_v29",
                "sdsr_v30",
                "sdsr_v31",
                "sdsr_v32",
                "sdsr_v33",
                "sdsr_v34",
                "sdsr_v35",
                "sdsr_v36",
            }
            and not args.sdsr_use_phase_downsample
        ):
            run_tags.append("nophase")
    elif args.projector_p5_mode != "full":
        run_tags.append(f"p5{args.projector_p5_mode}")
    if args.detector_init_seed is not None:
        run_tags.append(f"detinit{args.detector_init_seed}")
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
    if args.query_init != "learned":
        detach_tag = "detach" if args.query_memory_detach else "grad"
        run_tags.extend([args.query_init, detach_tag, f"gate{args.query_init_gate:g}"])
    if not args.lite_refpoint_refine:
        run_tags.append("iterref")
        if args.bbox_refine_mode != "shared":
            run_tags.append(f"bbox{args.bbox_refine_mode}")
    if args.dec_n_points != 2:
        run_tags.append(f"points{args.dec_n_points}")
    if args.scale_routing:
        run_tags.append("scaleroute")
        if args.scale_routing_mode != "legacy":
            run_tags.append(args.scale_routing_mode)
        if args.scale_routing_layers is not None:
            run_tags.append(
                "layers" + "-".join(str(layer) for layer in args.scale_routing_layers)
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
    if args.use_budgeted_sa:
        budgets_tag = "-".join(str(budget) for budget in args.sa_total_budgets)
        run_tags.extend(
            ["bsa" + budgets_tag, f"e{args.sa_start_epoch}-{args.sa_stop_epoch}"]
        )
    if args.use_dense_o2o:
        run_tags.extend(
            [
                f"denseo2o-{args.dense_o2o_mode}",
                f"e{args.dense_o2o_start_epoch}-{args.dense_o2o_image_stop_epoch}-{args.dense_o2o_copyblend_stop_epoch}",
            ]
        )
    if args.segmentation_head:
        run_tags.extend(
            [
                "seg",
                f"maskds{args.mask_downsample_ratio}",
                f"masklv{args.mask_feature_levels}",
            ]
        )
        if args.segmentation_head_only:
            run_tags.append("maskonly")
        if args.mask_boundary_sample_ratio > 0:
            run_tags.append(f"boundary{args.mask_boundary_sample_ratio:g}")

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
    parser.add_argument(
        "--subset",
        default="smoke",
        choices=("smoke", "overfit", "quick", "medium", "strong", "full"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1"))
    )
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument(
        "--sync-bn", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help="Accepted for compatibility with torch.distributed.launch; torchrun uses LOCAL_RANK.",
    )
    parser.add_argument(
        "--data-root", default="/data/cpc/root/dataset/COCO_RFDETR_TEST"
    )
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
        help="When loading detector_pretrain_weights, also skip backbone.0.projector.* so the projector is reinitialized. "
        "This is automatic when projector_type is not multiscale.",
    )
    parser.add_argument(
        "--detector-pretrain-include-encoder",
        action="store_true",
        help=(
            "Load the detector checkpoint's DINOv3 encoder as well. The default "
            "keeps the separately selected pretrained/refined encoder; this option "
            "provides an exact detection-to-segmentation warm start."
        ),
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
    parser.add_argument("--dec-n-points", type=int, default=2)
    parser.add_argument(
        "--lite-refpoint-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the initial reference boxes at every decoder layer. Disable for iterative box refinement.",
    )
    parser.add_argument(
        "--bbox-refine-mode",
        default="shared",
        choices=("shared", "layerwise", "residual"),
        help="Share the bbox head, clone one per decoder layer, or add zero-initialized layer residuals.",
    )
    parser.add_argument(
        "--query-init", default="learned", choices=("learned", "hybrid_topk")
    )
    parser.add_argument(
        "--query-memory-detach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop gradients through top-k proposal memory when it initializes decoder query content.",
    )
    parser.add_argument("--query-init-gate", type=float, default=0.0)
    parser.add_argument(
        "--scale-routing",
        action="store_true",
        help="Condition per-level deformable-attention weights on the current query and reference-box geometry.",
    )
    parser.add_argument(
        "--scale-routing-mode",
        default="legacy",
        choices=("legacy", "cell"),
        help="Use global normalized-box geometry or per-level feature-cell geometry for scale routing.",
    )
    parser.add_argument(
        "--scale-routing-layers",
        type=int,
        nargs="+",
        default=None,
        help="Zero-based decoder layers that use scale routing. By default all decoder layers use it.",
    )
    parser.add_argument(
        "--p5-attention-bias",
        type=float,
        default=0.0,
        help="Initial deformable-attention logit bias for the last P5 level.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--lr-drop", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler", choices=("step", "cosine"), default="step"
    )
    parser.add_argument("--lr-min-factor", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-vit-layer-decay", type=float, default=0.8)
    parser.add_argument("--lr-component-decay", type=float, default=0.7)
    parser.add_argument(
        "--backbone-refine-blocks",
        type=int,
        nargs="+",
        default=None,
        help="DINOv3 blocks whose refined initialization should be protected; final norm is included.",
    )
    parser.add_argument("--backbone-refine-lr-scale", type=float, default=1.0)
    parser.add_argument("--backbone-refine-anchor-coef", type=float, default=0.0)
    parser.add_argument("--backbone-refine-anchor-stop-epoch", type=int, default=20)
    parser.add_argument(
        "--online-refine-mode",
        choices=("none", "object_token", "feature_teacher"),
        default="none",
    )
    parser.add_argument("--online-refine-start-epoch", type=int, default=0)
    parser.add_argument("--online-refine-stop-epoch", type=int, default=6)
    parser.add_argument("--online-refine-coef", type=float, default=0.0)
    parser.add_argument("--online-refine-layers", type=int, nargs="+", default=[8, 11])
    parser.add_argument(
        "--online-refine-layer-weights", type=float, nargs="+", default=[0.35, 0.65]
    )
    parser.add_argument("--online-refine-background-weight", type=float, default=0.1)
    parser.add_argument("--online-refine-margin", type=float, default=0.2)
    parser.add_argument("--online-refine-min-box-tokens", type=int, default=1)
    parser.add_argument(
        "--projector-scale",
        nargs="+",
        default=["P4"],
        choices=("P3", "P4", "P5"),
        help="Feature levels produced by the projector and consumed by the decoder.",
    )
    parser.add_argument(
        "--projector-p3-indexes",
        type=int,
        nargs="+",
        default=None,
        help="DINOv3 block indexes retained by P3; omitted means all extracted indexes.",
    )
    parser.add_argument(
        "--projector-p4-indexes",
        type=int,
        nargs="+",
        default=None,
        help="DINOv3 block indexes retained by P4; omitted means all extracted indexes.",
    )
    parser.add_argument(
        "--projector-p5-indexes",
        type=int,
        nargs="+",
        default=None,
        help="DINOv3 block indexes retained by P5; omitted means all extracted indexes.",
    )
    parser.add_argument(
        "--projector-source-mode",
        choices=("mask", "prune"),
        default="mask",
        help="Mask unused per-scale sources or physically prune their projector branches.",
    )
    parser.add_argument(
        "--projector-c2f-blocks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "C2f bottleneck counts aligned with --projector-scale; "
            "omitted keeps the original three blocks at every level."
        ),
    )
    parser.add_argument(
        "--projector-resample-share",
        choices=("none", "p3", "p5", "p3_p5"),
        default="none",
        help=(
            "Share compatible resampling convolution kernels at P3, P5, or both; "
            "source-specific bias and normalization remain independent."
        ),
    )
    parser.add_argument("--projector-distill-teacher", default=None)
    parser.add_argument("--projector-distill-coef", type=float, default=0.0)
    parser.add_argument("--projector-distill-stop-epoch", type=int, default=20)
    parser.add_argument(
        "--projector-distill-level-weights",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 0.5],
    )
    parser.add_argument(
        "--projector-type",
        default="multiscale",
        choices=(
            "multiscale",
            "sdsr",
            "sdsr_v2",
            "sdsr_v3",
            "sdsr_v4",
            "sdsr_v5",
            "sdsr_v6",
            "sdsr_v7",
            "sdsr_v8",
            "sdsr_v9",
            "sdsr_v10",
            "sdsr_v11",
            "sdsr_v12",
            "sdsr_v13",
            "sdsr_v14",
            "sdsr_v15",
            "sdsr_v16",
            "sdsr_v17",
            "sdsr_v18",
            "sdsr_v19",
            "sdsr_v20",
            "sdsr_v21",
            "sdsr_v22",
            "sdsr_v23",
            "sdsr_v24",
            "sdsr_v25",
            "sdsr_v26",
            "sdsr_v27",
            "sdsr_v28",
            "sdsr_v29",
            "sdsr_v30",
            "sdsr_v31",
            "sdsr_v32",
            "sdsr_v33",
            "sdsr_v34",
            "sdsr_v35",
            "sdsr_v36",
        ),
    )
    parser.add_argument(
        "--projector-p5-mode",
        default="full",
        choices=(
            "full",
            "pool",
            "dwconv",
            "group2",
            "group2_mix",
            "group2_mix128",
            "group2_fullmix",
            "group2_first_full",
            "group2_first_full_dw",
            "group2_first2_full",
            "group2_last_full",
            "group4",
            "group8",
            "fusion",
            "fusion_wide",
            "fusion_refine",
            "fusion_residual",
        ),
        help="Build P5 independently (full) or derive it cheaply from P4.",
    )
    parser.add_argument("--sdsr-rank-channels", type=int, default=64)
    parser.add_argument("--sdsr-detail-channels", type=int, default=32)
    parser.add_argument(
        "--sdsr-use-local-reassembly",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sdsr-use-directional-guide",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sdsr-use-phase-downsample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sdsr-cross-scale-mode",
        choices=("none", "topdown", "bidirectional"),
        default="none",
    )
    parser.add_argument("--sdsr-cross-scale-rank", type=int, default=32)
    parser.add_argument(
        "--detector-init-seed",
        type=int,
        default=None,
        help="Reset torch RNG before constructing the transformer and detection heads.",
    )
    parser.add_argument(
        "--out-feature-indexes",
        type=int,
        nargs="+",
        default=[2, 5, 8, 11],
        help="Zero-based DINOv3 block indexes whose intermediate features are fused by the projector.",
    )

    parser.add_argument("--eval-max-dets", type=int, default=100)
    parser.add_argument(
        "--segmentation-head",
        action="store_true",
        help=(
            "Train and evaluate the query-conditioned RF-DETR instance mask head. "
            "This enables COCO masks and square-resize batching."
        ),
    )
    parser.add_argument(
        "--segmentation-head-only",
        action="store_true",
        help=(
            "Freeze the pretrained detector (including normalization state) and "
            "train only the query-conditioned mask head."
        ),
    )
    parser.add_argument("--mask-downsample-ratio", type=int, default=4)
    parser.add_argument(
        "--mask-feature-levels",
        type=int,
        default=1,
        help="Number of projector levels used by the mask head (1=P3, 3=P3+P4+P5).",
    )
    parser.add_argument("--mask-point-sample-ratio", type=int, default=16)
    parser.add_argument(
        "--mask-boundary-sample-ratio",
        type=float,
        default=0.0,
        help="Fraction of mask-loss points sampled from target boundaries.",
    )
    parser.add_argument("--mask-ce-loss-coef", type=float, default=5.0)
    parser.add_argument("--mask-dice-loss-coef", type=float, default=5.0)
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
    parser.add_argument(
        "--sa-start-epoch",
        type=int,
        default=0,
        help="First epoch using budgeted SA (inclusive).",
    )
    parser.add_argument(
        "--sa-stop-epoch",
        type=int,
        default=0,
        help="First epoch returning to strict aux O2O.",
    )
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
    parser.add_argument(
        "--dense-o2o-mode", default="image", choices=("image", "enhanced")
    )
    parser.add_argument("--dense-o2o-start-epoch", type=int, default=2)
    parser.add_argument("--dense-o2o-image-stop-epoch", type=int, default=12)
    parser.add_argument("--dense-o2o-copyblend-stop-epoch", type=int, default=21)
    parser.add_argument("--dense-o2o-mosaic-prob", type=float, default=0.5)
    parser.add_argument("--dense-o2o-mixup-prob", type=float, default=0.5)
    parser.add_argument("--dense-o2o-copyblend-prob", type=float, default=0.5)
    parser.add_argument(
        "--dense-o2o-copyblend-area-threshold", type=float, default=100.0
    )
    parser.add_argument("--dense-o2o-copyblend-num-objects", type=int, default=3)
    parser.add_argument(
        "--dense-o2o-copyblend-expand-ratios", type=float, nargs=2, default=[0.1, 0.25]
    )
    parser.add_argument(
        "--backbone-register-border-tokens",
        type=int,
        default=0,
        help="Patch-token border added around the image inside the DINOv3 backbone and cropped before projector.",
    )
    parser.add_argument(
        "--backbone-register-fill", default="randn", choices=("randn", "rand", "zero")
    )
    parser.add_argument("--backbone-register-noise-std", type=float, default=1.0)
    parser.add_argument(
        "--feature-adapter",
        default="none",
        choices=("none", "residual_ln_1x1", "ln_1x1"),
        help="Optional lightweight adapter applied to DINOv3 feature maps before the projector.",
    )
    parser.add_argument("--feature-adapter-init-scale", type=float, default=1.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument(
        "--square-resize-div-64",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize images to a square shape divisible by 64 before batching.",
    )
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
    if args.backbone_refine_blocks is None:
        args.backbone_refine_blocks = []
    if args.backbone_refine_blocks != sorted(set(args.backbone_refine_blocks)):
        raise ValueError("--backbone-refine-blocks must be unique and ascending.")
    if any(index < 0 or index >= 12 for index in args.backbone_refine_blocks):
        raise ValueError("--backbone-refine-blocks must be in [0, 11] for DINOv3-small.")
    if args.backbone_refine_lr_scale <= 0:
        raise ValueError("--backbone-refine-lr-scale must be positive.")
    if args.backbone_refine_anchor_coef < 0:
        raise ValueError("--backbone-refine-anchor-coef must be non-negative.")
    if args.backbone_refine_anchor_coef > 0 and not args.backbone_refine_blocks:
        raise ValueError("L2-SP anchoring requires --backbone-refine-blocks.")
    if args.backbone_refine_anchor_stop_epoch < 0:
        raise ValueError("--backbone-refine-anchor-stop-epoch must be non-negative.")
    if args.online_refine_start_epoch < 0:
        raise ValueError("--online-refine-start-epoch must be non-negative.")
    if args.online_refine_stop_epoch <= args.online_refine_start_epoch:
        raise ValueError("--online-refine-stop-epoch must exceed the start epoch.")
    if args.online_refine_coef < 0:
        raise ValueError("--online-refine-coef must be non-negative.")
    if args.online_refine_mode != "none" and args.online_refine_coef <= 0:
        raise ValueError("An enabled online refinement mode requires a positive coefficient.")
    if args.online_refine_layers != sorted(set(args.online_refine_layers)):
        raise ValueError("--online-refine-layers must be unique and ascending.")
    if any(layer not in args.out_feature_indexes for layer in args.online_refine_layers):
        raise ValueError("--online-refine-layers must be included in --out-feature-indexes.")
    if len(args.online_refine_layer_weights) != len(args.online_refine_layers):
        raise ValueError("Online refinement requires one weight per selected layer.")
    if sum(args.online_refine_layer_weights) <= 0:
        raise ValueError("Online refinement layer weights must have a positive sum.")
    if args.online_refine_background_weight < 0:
        raise ValueError("--online-refine-background-weight must be non-negative.")
    if args.online_refine_min_box_tokens < 1:
        raise ValueError("--online-refine-min-box-tokens must be at least one.")
    if args.out_feature_indexes[0] < 0 or args.out_feature_indexes[-1] >= 12:
        raise ValueError(
            "DINOv3-S has 12 blocks, so out_feature_indexes must be in [0, 11]."
        )
    per_scale_indexes = {
        "P3": args.projector_p3_indexes,
        "P4": args.projector_p4_indexes,
        "P5": args.projector_p5_indexes,
    }
    configured_source_indexes = [
        indexes for indexes in per_scale_indexes.values() if indexes is not None
    ]
    if configured_source_indexes:
        missing_scales = [
            scale for scale in args.projector_scale if per_scale_indexes[scale] is None
        ]
        if missing_scales:
            raise ValueError(
                "Per-scale projector selection requires indexes for every configured "
                f"scale; missing {missing_scales}."
            )
        args.projector_source_indexes = {
            scale: per_scale_indexes[scale] for scale in args.projector_scale
        }
    else:
        args.projector_source_indexes = None
    if args.projector_c2f_blocks is not None:
        if len(args.projector_c2f_blocks) != len(args.projector_scale):
            raise ValueError(
                "--projector-c2f-blocks must provide one value for every "
                "--projector-scale entry."
            )
        if any(blocks < 0 for blocks in args.projector_c2f_blocks):
            raise ValueError("--projector-c2f-blocks values must be non-negative.")
        args.projector_c2f_blocks = dict(
            zip(args.projector_scale, args.projector_c2f_blocks)
        )
    else:
        args.projector_c2f_blocks = None
    if args.projector_distill_coef < 0:
        raise ValueError("--projector-distill-coef must be non-negative.")
    if args.projector_distill_coef > 0 and not args.projector_distill_teacher:
        raise ValueError(
            "--projector-distill-teacher is required when distillation is enabled."
        )
    if args.projector_distill_stop_epoch < 0:
        raise ValueError("--projector-distill-stop-epoch must be non-negative.")
    if len(args.projector_distill_level_weights) != len(args.projector_scale):
        raise ValueError(
            "--projector-distill-level-weights must match --projector-scale."
        )
    if sum(args.projector_distill_level_weights) <= 0:
        raise ValueError("Projector distillation level weights must sum to > 0.")
    if args.projector_type in {
        "sdsr",
        "sdsr_v2",
        "sdsr_v3",
        "sdsr_v4",
        "sdsr_v5",
        "sdsr_v6",
        "sdsr_v7",
        "sdsr_v8",
        "sdsr_v9",
        "sdsr_v10",
        "sdsr_v11",
        "sdsr_v12",
        "sdsr_v13",
        "sdsr_v14",
        "sdsr_v15",
        "sdsr_v16",
        "sdsr_v17",
        "sdsr_v18",
        "sdsr_v19",
        "sdsr_v20",
        "sdsr_v21",
        "sdsr_v22",
        "sdsr_v23",
        "sdsr_v24",
        "sdsr_v25",
        "sdsr_v26",
        "sdsr_v27",
        "sdsr_v28",
        "sdsr_v29",
        "sdsr_v30",
        "sdsr_v31",
        "sdsr_v32",
        "sdsr_v33",
        "sdsr_v34",
        "sdsr_v35",
        "sdsr_v36",
    }:
        if (
            args.projector_type
            in {
                "sdsr_v2",
                "sdsr_v3",
                "sdsr_v4",
                "sdsr_v5",
                "sdsr_v6",
                "sdsr_v7",
                "sdsr_v8",
                "sdsr_v9",
                "sdsr_v10",
                "sdsr_v11",
                "sdsr_v12",
                "sdsr_v13",
                "sdsr_v14",
                "sdsr_v15",
                "sdsr_v16",
                "sdsr_v17",
                "sdsr_v18",
                "sdsr_v19",
                "sdsr_v20",
                "sdsr_v21",
                "sdsr_v22",
                "sdsr_v23",
                "sdsr_v24",
                "sdsr_v25",
                "sdsr_v26",
                "sdsr_v27",
                "sdsr_v28",
                "sdsr_v29",
                "sdsr_v30",
                "sdsr_v31",
                "sdsr_v32",
                "sdsr_v33",
                "sdsr_v34",
                "sdsr_v35",
                "sdsr_v36",
            }
            and args.sdsr_rank_channels < 1
        ):
            raise ValueError(
                "SDSR-v2/v3/v4/v5/v6/v7/v8/v9 rank channels must be positive."
            )
        if args.sdsr_detail_channels < 8 or args.sdsr_detail_channels % 4 != 0:
            raise ValueError(
                "SDSR detail channels must be at least 8 and divisible by 4."
            )
        if args.sdsr_use_directional_guide and not args.sdsr_use_local_reassembly:
            log_main(
                "SDSR directional guide is inactive because local reassembly is disabled."
            )
        if args.sdsr_cross_scale_rank < 1:
            raise ValueError("SDSR cross-scale rank must be positive.")
        if (
            args.projector_type
            not in {
                "sdsr_v7",
                "sdsr_v8",
                "sdsr_v9",
                "sdsr_v10",
                "sdsr_v11",
                "sdsr_v12",
                "sdsr_v13",
                "sdsr_v14",
                "sdsr_v15",
                "sdsr_v16",
                "sdsr_v17",
                "sdsr_v18",
                "sdsr_v19",
                "sdsr_v20",
                "sdsr_v21",
                "sdsr_v22",
                "sdsr_v23",
                "sdsr_v24",
                "sdsr_v25",
                "sdsr_v26",
                "sdsr_v27",
                "sdsr_v28",
                "sdsr_v29",
                "sdsr_v30",
                "sdsr_v31",
                "sdsr_v32",
                "sdsr_v33",
                "sdsr_v34",
                "sdsr_v35",
                "sdsr_v36",
            }
            and args.sdsr_cross_scale_mode != "none"
        ):
            raise ValueError(
                "Cross-scale calibration is only supported by SDSR-v7/v8/v9."
            )
    if args.use_budgeted_sa:
        if (
            args.sa_start_epoch < 0
            or not args.sa_start_epoch < args.sa_stop_epoch <= effective_epochs
        ):
            raise ValueError(
                "Budgeted SA requires 0 <= sa_start_epoch < sa_stop_epoch <= epochs."
            )
        if any(budget < args.group_detr for budget in args.sa_total_budgets):
            raise ValueError("Every SA total budget must be at least group_detr.")
        if (
            args.sa_area_thresholds[0] <= 0
            or args.sa_area_thresholds[0] >= args.sa_area_thresholds[1]
        ):
            raise ValueError("SA area thresholds must be positive and increasing.")
    if args.use_dense_o2o:
        if args.use_budgeted_sa:
            raise ValueError(
                "Dense O2O and Budgeted SA are independent experiments and cannot be enabled together."
            )
        if (
            args.batch_size * args.grad_accum_steps < 4
            and args.dense_o2o_mosaic_prob > 0
        ):
            raise ValueError(
                "Dense O2O Mosaic requires a local effective batch size of at least 4."
            )
        if (
            not 0
            <= args.dense_o2o_start_epoch
            < args.dense_o2o_image_stop_epoch
            <= effective_epochs
        ):
            raise ValueError("Dense O2O requires 0 <= start < image_stop <= epochs.")
        if args.dense_o2o_mode == "enhanced" and not (
            args.dense_o2o_start_epoch
            < args.dense_o2o_copyblend_stop_epoch
            <= effective_epochs
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
            or args.dense_o2o_copyblend_expand_ratios[0]
            > args.dense_o2o_copyblend_expand_ratios[1]
        ):
            raise ValueError(
                "Dense O2O CopyBlend expansion ratios must be non-negative and increasing."
            )
    if args.segmentation_head:
        if args.use_cdn:
            raise ValueError(
                "The current RF-DETR mask path does not support CDN denoising; "
                "disable --use-cdn for the segmentation baseline."
            )
        if args.use_dense_o2o:
            raise ValueError(
                "Dense O2O does not yet transform instance masks; disable "
                "--use-dense-o2o for the segmentation baseline."
            )
        if args.mask_downsample_ratio < 1:
            raise ValueError("--mask-downsample-ratio must be positive.")
        if not 1 <= args.mask_feature_levels <= len(args.projector_scale):
            raise ValueError(
                "--mask-feature-levels must be between 1 and the number of projector scales."
            )
        if args.mask_point_sample_ratio < 1:
            raise ValueError("--mask-point-sample-ratio must be positive.")
        if not 0.0 <= args.mask_boundary_sample_ratio <= 1.0:
            raise ValueError("--mask-boundary-sample-ratio must lie in [0, 1].")
        if args.mask_ce_loss_coef < 0 or args.mask_dice_loss_coef < 0:
            raise ValueError("Mask loss coefficients must be non-negative.")
    elif args.segmentation_head_only:
        raise ValueError("--segmentation-head-only requires --segmentation-head.")
    if not args.scale_routing and (
        args.scale_routing_mode != "legacy" or args.scale_routing_layers is not None
    ):
        raise ValueError("--scale-routing-mode/layers require --scale-routing.")
    if args.lite_refpoint_refine and args.bbox_refine_mode != "shared":
        raise ValueError(
            "--bbox-refine-mode layerwise/residual requires --no-lite-refpoint-refine."
        )
    if args.p5_attention_bias != 0.0:
        if not args.projector_scale or args.projector_scale[-1] != "P5":
            raise ValueError(
                "--p5-attention-bias requires P5 as the last --projector-scale."
            )
    if args.projector_p5_mode != "full":
        if args.projector_type != "multiscale":
            raise ValueError(
                "A lightweight --projector-p5-mode requires --projector-type multiscale."
            )
        if "P5" not in args.projector_scale:
            raise ValueError(
                "A lightweight --projector-p5-mode requires P5 in --projector-scale."
            )
        p5_index = args.projector_scale.index("P5")
        if p5_index == 0 or args.projector_scale[p5_index - 1] != "P4":
            raise ValueError(
                "A lightweight --projector-p5-mode requires P4 immediately before P5."
            )
    if args.scale_routing_layers is not None:
        if len(set(args.scale_routing_layers)) != len(args.scale_routing_layers):
            raise ValueError("--scale-routing-layers must not contain duplicates.")
        if any(
            layer < 0 or layer >= args.dec_layers for layer in args.scale_routing_layers
        ):
            raise ValueError(
                f"--scale-routing-layers must be in [0, {args.dec_layers - 1}]."
            )
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
        pretrain_exclude_keys = []
        if not args.detector_pretrain_include_encoder:
            pretrain_exclude_keys.append("backbone.0.encoder.*")
        if (
            args.detector_pretrain_exclude_projector
            or args.projector_type != "multiscale"
            or args.projector_p5_mode != "full"
        ):
            pretrain_exclude_keys.append("backbone.0.projector.*")
        if not pretrain_exclude_keys:
            pretrain_exclude_keys = None

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
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_refine_mode=args.bbox_refine_mode,
        query_init=args.query_init,
        query_memory_detach=args.query_memory_detach,
        query_init_gate=args.query_init_gate,
        scale_routing=args.scale_routing,
        scale_routing_mode=args.scale_routing_mode,
        scale_routing_layers=args.scale_routing_layers,
        p5_attention_bias=args.p5_attention_bias,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        projector_source_indexes=args.projector_source_indexes,
        projector_source_mode=args.projector_source_mode,
        projector_c2f_blocks=args.projector_c2f_blocks,
        projector_resample_share=args.projector_resample_share,
        projector_type=args.projector_type,
        projector_p5_mode=args.projector_p5_mode,
        sdsr_rank_channels=args.sdsr_rank_channels,
        sdsr_detail_channels=args.sdsr_detail_channels,
        sdsr_use_local_reassembly=args.sdsr_use_local_reassembly,
        sdsr_use_directional_guide=args.sdsr_use_directional_guide,
        sdsr_use_phase_downsample=args.sdsr_use_phase_downsample,
        sdsr_cross_scale_mode=args.sdsr_cross_scale_mode,
        sdsr_cross_scale_rank=args.sdsr_cross_scale_rank,
        detector_init_seed=args.detector_init_seed,
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
        segmentation_head=args.segmentation_head,
        mask_downsample_ratio=args.mask_downsample_ratio,
        mask_feature_levels=args.mask_feature_levels,
    )

    log_main(f"project_root={PROJECT_ROOT}")
    log_main(f"subset_path={subset_path}")
    log_main(f"output_dir={output_dir}")
    log_main(f"model=RFDETRDINOv3, encoder={DEFAULT_ENCODER}")
    log_main(f"pretrained_encoder={args.pretrained_encoder}")
    log_main(f"detector_pretrain_weights={args.detector_pretrain_weights}")
    log_main(
        f"detector_pretrain_include_encoder={args.detector_pretrain_include_encoder}"
    )
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
    log_main(f"dec_n_points={args.dec_n_points}")
    log_main(f"lite_refpoint_refine={args.lite_refpoint_refine}")
    log_main(f"bbox_refine_mode={args.bbox_refine_mode}")
    log_main(f"query_init={args.query_init}")
    log_main(f"query_memory_detach={args.query_memory_detach}")
    log_main(f"query_init_gate={args.query_init_gate}")
    log_main(f"scale_routing={args.scale_routing}")
    log_main(f"scale_routing_mode={args.scale_routing_mode}")
    log_main(f"scale_routing_layers={args.scale_routing_layers}")
    log_main(f"p5_attention_bias={args.p5_attention_bias}")
    log_main(f"out_feature_indexes={args.out_feature_indexes}")
    log_main(f"projector_scale={args.projector_scale}")
    log_main(f"projector_source_indexes={args.projector_source_indexes}")
    log_main(f"projector_source_mode={args.projector_source_mode}")
    log_main(f"projector_c2f_blocks={args.projector_c2f_blocks}")
    log_main(f"projector_resample_share={args.projector_resample_share}")
    log_main(f"projector_distill_teacher={args.projector_distill_teacher}")
    log_main(f"projector_distill_coef={args.projector_distill_coef}")
    log_main(f"projector_distill_stop_epoch={args.projector_distill_stop_epoch}")
    log_main(f"projector_distill_level_weights={args.projector_distill_level_weights}")
    log_main(f"projector_type={args.projector_type}")
    log_main(f"projector_p5_mode={args.projector_p5_mode}")
    log_main(f"sdsr_rank_channels={args.sdsr_rank_channels}")
    log_main(f"sdsr_detail_channels={args.sdsr_detail_channels}")
    log_main(f"sdsr_use_local_reassembly={args.sdsr_use_local_reassembly}")
    log_main(f"sdsr_use_directional_guide={args.sdsr_use_directional_guide}")
    log_main(f"sdsr_use_phase_downsample={args.sdsr_use_phase_downsample}")
    log_main(f"sdsr_cross_scale_mode={args.sdsr_cross_scale_mode}")
    log_main(f"sdsr_cross_scale_rank={args.sdsr_cross_scale_rank}")
    log_main(f"detector_init_seed={args.detector_init_seed}")
    log_main(f"lr={args.lr}")
    log_main(f"lr_encoder={args.lr_encoder}")
    log_main(f"lr_drop={args.lr_drop}")
    log_main(f"warmup_epochs={args.warmup_epochs}")
    log_main(f"lr_scheduler={args.lr_scheduler}")
    log_main(f"lr_min_factor={args.lr_min_factor}")
    log_main(f"weight_decay={args.weight_decay}")
    log_main(f"lr_vit_layer_decay={args.lr_vit_layer_decay}")
    log_main(f"lr_component_decay={args.lr_component_decay}")
    log_main(f"backbone_refine_blocks={args.backbone_refine_blocks}")
    log_main(f"backbone_refine_lr_scale={args.backbone_refine_lr_scale}")
    log_main(f"backbone_refine_anchor_coef={args.backbone_refine_anchor_coef}")
    log_main(f"backbone_refine_anchor_stop_epoch={args.backbone_refine_anchor_stop_epoch}")
    log_main(f"online_refine_mode={args.online_refine_mode}")
    log_main(
        f"online_refine_epochs=[{args.online_refine_start_epoch}, "
        f"{args.online_refine_stop_epoch})"
    )
    log_main(f"online_refine_coef={args.online_refine_coef}")
    log_main(f"online_refine_layers={args.online_refine_layers}")
    log_main(f"online_refine_layer_weights={args.online_refine_layer_weights}")
    log_main(f"online_refine_background_weight={args.online_refine_background_weight}")
    log_main(f"online_refine_margin={args.online_refine_margin}")
    log_main(f"online_refine_min_box_tokens={args.online_refine_min_box_tokens}")
    log_main(f"multi_scale={args.multi_scale}")
    log_main(f"expanded_scales={args.expanded_scales}")
    log_main(f"square_resize_div_64={args.square_resize_div_64}")
    log_main(f"aug_preset={args.aug_preset}")
    log_main(f"eval_max_dets={args.eval_max_dets}")
    log_main(f"segmentation_head={args.segmentation_head}")
    log_main(f"segmentation_head_only={args.segmentation_head_only}")
    log_main(f"mask_downsample_ratio={args.mask_downsample_ratio}")
    log_main(f"mask_feature_levels={args.mask_feature_levels}")
    log_main(f"mask_point_sample_ratio={args.mask_point_sample_ratio}")
    log_main(f"mask_boundary_sample_ratio={args.mask_boundary_sample_ratio}")
    log_main(f"mask_ce_loss_coef={args.mask_ce_loss_coef}")
    log_main(f"mask_dice_loss_coef={args.mask_dice_loss_coef}")
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
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_refine_mode=args.bbox_refine_mode,
        query_init=args.query_init,
        query_memory_detach=args.query_memory_detach,
        query_init_gate=args.query_init_gate,
        scale_routing=args.scale_routing,
        scale_routing_mode=args.scale_routing_mode,
        scale_routing_layers=args.scale_routing_layers,
        p5_attention_bias=args.p5_attention_bias,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        projector_source_indexes=args.projector_source_indexes,
        projector_source_mode=args.projector_source_mode,
        projector_c2f_blocks=args.projector_c2f_blocks,
        projector_resample_share=args.projector_resample_share,
        projector_type=args.projector_type,
        projector_p5_mode=args.projector_p5_mode,
        sdsr_rank_channels=args.sdsr_rank_channels,
        sdsr_detail_channels=args.sdsr_detail_channels,
        sdsr_use_local_reassembly=args.sdsr_use_local_reassembly,
        sdsr_use_directional_guide=args.sdsr_use_directional_guide,
        sdsr_use_phase_downsample=args.sdsr_use_phase_downsample,
        sdsr_cross_scale_mode=args.sdsr_cross_scale_mode,
        sdsr_cross_scale_rank=args.sdsr_cross_scale_rank,
        positional_encoding_size=args.resolution // 16,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_drop=args.lr_drop,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        lr_min_factor=args.lr_min_factor,
        weight_decay=args.weight_decay,
        lr_vit_layer_decay=args.lr_vit_layer_decay,
        lr_component_decay=args.lr_component_decay,
        backbone_refine_blocks=tuple(args.backbone_refine_blocks),
        backbone_refine_lr_scale=args.backbone_refine_lr_scale,
        backbone_refine_anchor_coef=args.backbone_refine_anchor_coef,
        backbone_refine_anchor_stop_epoch=args.backbone_refine_anchor_stop_epoch,
        online_refine_mode=args.online_refine_mode,
        online_refine_start_epoch=args.online_refine_start_epoch,
        online_refine_stop_epoch=args.online_refine_stop_epoch,
        online_refine_coef=args.online_refine_coef,
        online_refine_layers=tuple(args.online_refine_layers),
        online_refine_layer_weights=tuple(args.online_refine_layer_weights),
        online_refine_background_weight=args.online_refine_background_weight,
        online_refine_margin=args.online_refine_margin,
        online_refine_min_box_tokens=args.online_refine_min_box_tokens,
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        aug_config=AUG_PRESETS[args.aug_preset],
        use_ema=args.use_ema,
        tensorboard=args.tensorboard,
        run_test=args.run_test,
        eval_max_dets=args.eval_max_dets,
        segmentation_head=args.segmentation_head,
        segmentation_head_only=args.segmentation_head_only,
        square_resize_div_64=args.square_resize_div_64,
        mask_point_sample_ratio=args.mask_point_sample_ratio,
        mask_boundary_sample_ratio=args.mask_boundary_sample_ratio,
        mask_ce_loss_coef=args.mask_ce_loss_coef,
        mask_dice_loss_coef=args.mask_dice_loss_coef,
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
        projector_distill_teacher=args.projector_distill_teacher,
        projector_distill_coef=args.projector_distill_coef,
        projector_distill_stop_epoch=args.projector_distill_stop_epoch,
        projector_distill_level_weights=tuple(args.projector_distill_level_weights),
        progress_bar=args.progress_bar,
    )


if __name__ == "__main__":
    main()
