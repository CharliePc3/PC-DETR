#!/usr/bin/env python3
"""Compare DINOv3 dense-token PCA maps with a shared per-image PCA basis."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import analyze_dinov3_tokens as diag

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PlainImageRecord:
    def __init__(self, image_id: int, file_name: str, path: Path):
        self.image_id = image_id
        self.file_name = file_name
        self.path = path
        self.bboxes_xywh = []
        self.width = 0
        self.height = 0


def load_plain_image_records(image_root: Path) -> list[PlainImageRecord]:
    paths = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No images found under {image_root}")
    return [PlainImageRecord(idx, str(path.relative_to(image_root)), path) for idx, path in enumerate(paths)]


def parse_variant(value: str) -> tuple[str, str | None]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Variant must be formatted as label:path, use label: for base weights.")
    label, path = value.split(":", 1)
    if not label:
        raise argparse.ArgumentTypeError("Variant label cannot be empty.")
    return label, path or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Shared-PCA comparison for DINOv3 token maps")
    parser.add_argument("--data-root", default="/data/cpc/root/dataset/COCO")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--image-root", default=None, help="Recursive image folder. Overrides COCO record loading.")
    parser.add_argument("--output-dir", default="output/token_analysis/pca_compare_shared")
    parser.add_argument("--encoder", default="dinov3_small", choices=tuple(f"dinov3_{k}" for k in diag.SIZE_TO_MODEL))
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--register-border", type=int, default=1)
    parser.add_argument("--register-fill", choices=("zero", "rand", "randn"), default="zero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def build_model(encoder: str, pretrained_encoder: str | None, device: torch.device) -> torch.nn.Module:
    args = argparse.Namespace(encoder=encoder, pretrained_encoder=pretrained_encoder)
    return diag.build_dinov3_encoder(args).to(device)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    if args.image_root:
        records = load_plain_image_records(Path(args.image_root))
        if args.shuffle:
            random.shuffle(records)
        records = records[: args.num_images]
    else:
        ann_file = Path(args.ann_file) if args.ann_file else diag.infer_ann_file(data_root, args.split)
        records = diag.load_coco_records(data_root, args.split, ann_file)
        if args.shuffle:
            random.shuffle(records)
        records = records[: args.num_images]
    if not records:
        raise RuntimeError("No annotated images found.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    models = [(label, build_model(args.encoder, path, device)) for label, path in args.variant]
    patch_size = int(models[0][1].patch_size)
    image_dir = data_root / args.split

    for idx, record in enumerate(records):
        image_path = getattr(record, "path", None)
        if image_path is None:
            image_path = image_dir / record.file_name
            if not image_path.exists():
                image_path = data_root / record.file_name
        image = Image.open(image_path).convert("RGB")
        image_tensor = diag.image_to_tensor(image, args.resolution, device)
        if record.bboxes_xywh:
            boxes_xyxy = diag.boxes_to_resized_xyxy(
                record.bboxes_xywh,
                record.width,
                record.height,
                args.resolution,
            )
        else:
            boxes_xyxy = torch.empty((0, 4), dtype=torch.float32)

        variant_tokens = []
        for label, model in models:
            registered_tensor, register_mask = diag.surround_tensor_with_registers(
                image_tensor,
                patch_size=patch_size,
                register_border=args.register_border,
                register_fill=args.register_fill,
            )
            features = model.forward_features(registered_tensor)
            tokens = F.normalize(features["x_norm_patchtokens"][0].float(), dim=-1)
            variant_tokens.append((label, tokens, registered_tensor, register_mask[0]))

        all_tokens = torch.cat([tokens for _, tokens, _, _ in variant_tokens], dim=0)
        metadata = diag.compute_pca_metadata(all_tokens)
        reg_h = variant_tokens[0][2].shape[-2] // patch_size
        reg_w = variant_tokens[0][2].shape[-1] // patch_size

        panels = [
            diag.draw_boxes(
                diag.draw_title(
                    image.convert("RGB").resize((args.resolution, args.resolution), Image.BICUBIC),
                    f"{record.image_id} {record.file_name}",
                ),
                boxes_xyxy,
            )
        ]
        for label, tokens, _, _ in variant_tokens:
            pca_image = diag.pca_tokens_to_image(tokens, reg_h, reg_w, args.resolution, metadata)
            panels.append(diag.draw_title(pca_image, f"{label}: shared PCA + register"))
        panels.append(
            diag.draw_title(
                diag.register_mask_to_image(variant_tokens[0][3], reg_h, reg_w, args.resolution),
                "register-token mask",
            )
        )

        cols = 2
        rows = (len(panels) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * args.resolution, rows * args.resolution), "black")
        for panel_idx, panel in enumerate(panels):
            x = (panel_idx % cols) * args.resolution
            y = (panel_idx // cols) * args.resolution
            canvas.paste(panel, (x, y))
        canvas.save(output_dir / f"{idx:04d}_{record.image_id}_shared_pca.jpg")
        print(f"[{idx + 1}/{len(records)}] saved {record.image_id}", flush=True)

    print(f"Saved shared-PCA comparisons to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
