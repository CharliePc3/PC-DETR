import os
os.environ.setdefault("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3")
os.environ.setdefault("DINOV3_WEIGHTS_DIR", "/data/cpc/root/project/RF-DETR-DINOv3/weights/dinov3")
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.datasets.aug_config import (
    AUG_AERIAL,
    AUG_AGGRESSIVE,
    AUG_CONSERVATIVE,
    AUG_CONFIG,
    AUG_INDUSTRIAL,
)
from rfdetr_dinov3 import RFDETRDINOv3

AUG_PRESETS = {
    "default": AUG_CONFIG,
    "conservative": AUG_CONSERVATIVE,
    "aggressive": AUG_AGGRESSIVE,
    "aerial": AUG_AERIAL,
    "industrial": AUG_INDUSTRIAL,
    "none": {},
}


def parse_args():
    parser = argparse.ArgumentParser("Train RF-DETR-DINOv3.")
    parser.add_argument("--dataset-dir", default="/data/cpc/root/dataset/PUMCH-AA/train_val/YOLO_format")
    parser.add_argument("--dataset-file", default="yolo", choices=("coco", "o365", "roboflow", "yolo"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "pumchaa_dinov3_smoke"))
    parser.add_argument("--encoder", default="dinov3_small")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--expanded-scales", action="store_true")
    parser.add_argument("--aug-preset", default="default", choices=tuple(AUG_PRESETS))
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-select", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
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
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    positional_encoding_size = args.resolution // 16

    print("Building RF-DETR-DINOv3 model...")
    print(f"encoder={args.encoder}")
    print(f"resolution={args.resolution}")
    print(f"dec_layers={args.dec_layers}")
    print(f"num_queries={args.num_queries}")
    print(f"num_select={args.num_select}")
    print(f"projector_scale={args.projector_scale}")
    print(f"lr={args.lr}")
    print(f"lr_encoder={args.lr_encoder}")
    print(f"weight_decay={args.weight_decay}")
    print(f"lr_vit_layer_decay={args.lr_vit_layer_decay}")
    print(f"lr_component_decay={args.lr_component_decay}")
    print(f"multi_scale={args.multi_scale}")
    print(f"expanded_scales={args.expanded_scales}")
    print(f"aug_preset={args.aug_preset}")
    print(f"eval_max_dets={args.eval_max_dets}")

    model = RFDETRDINOv3(
        encoder=args.encoder,
        pretrain_weights=None,
        num_classes=args.num_classes,
        resolution=args.resolution,
        dec_layers=args.dec_layers,
        num_queries=args.num_queries,
        num_select=args.num_select,
        projector_scale=args.projector_scale,
        positional_encoding_size=positional_encoding_size,
    )

    print("Starting training/evaluation...")
    model.train(
        dataset_file=args.dataset_file,
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        device=args.device,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        resume=args.resume,
        eval=args.eval_only,
        resolution=args.resolution,
        dec_layers=args.dec_layers,
        num_queries=args.num_queries,
        num_select=args.num_select,
        projector_scale=args.projector_scale,
        positional_encoding_size=positional_encoding_size,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        lr_vit_layer_decay=args.lr_vit_layer_decay,
        lr_component_decay=args.lr_component_decay,
        eval_max_dets=args.eval_max_dets,
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        aug_config=AUG_PRESETS[args.aug_preset],
        use_ema=args.use_ema,
        tensorboard=args.tensorboard,
        run_test=args.run_test,
        progress_bar=args.progress_bar,
    )


if __name__ == "__main__":
    main()
