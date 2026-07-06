#!/usr/bin/env python3
"""Frozen DINOv3 linear segmentation probe on ADE20K.

This is a lightweight dense-feature quality check, not a full semantic
segmentation training recipe. It freezes DINOv3, trains a 1x1 convolutional
classifier on patch features, and reports ADE20K validation mIoU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TVF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3")
os.environ.setdefault("DINOV3_WEIGHTS_DIR", str(PROJECT_ROOT / "weights" / "dinov3"))

from rfdetr.models.backbone.dinov3 import (  # noqa: E402
    SIZE_TO_MODEL,
    _load_dinov3_backbone_builder,
    _load_state_dict,
    _resolve_weights_path,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_INDEX = 255


@dataclass
class EvalResult:
    miou: float
    mean_acc: float
    pixel_acc: float
    valid_classes: int
    loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ADE20K frozen DINOv3 linear segmentation probe")
    parser.add_argument("--data-root", default="/data/cpc/root/dataset/ADE20K/ADEChallengeData2016")
    parser.add_argument("--output-dir", default="output/ade20k_linear_seg/dinov3_small")
    parser.add_argument("--encoder", default="dinov3_small", choices=tuple(f"dinov3_{k}" for k in SIZE_TO_MODEL))
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[11])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--visualize", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-head", default=None, help="Evaluate a saved linear head checkpoint and exit.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ade_label_to_train_id(mask: Image.Image) -> torch.Tensor:
    array = np.array(mask, dtype=np.int64)
    target = np.full_like(array, IGNORE_INDEX, dtype=np.int64)
    valid = (array >= 1) & (array <= 150)
    target[valid] = array[valid] - 1
    return torch.from_numpy(target)


class ADE20KDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        image_size: int,
        crop_size: int,
        train: bool,
        limit: int | None = None,
    ):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.crop_size = crop_size
        self.train = train
        image_dir = data_root / "images" / split
        ann_dir = data_root / "annotations" / split
        if not image_dir.exists() or not ann_dir.exists():
            raise FileNotFoundError(f"ADE20K split not found: {image_dir} / {ann_dir}")

        samples = []
        for image_path in sorted(image_dir.glob("*.jpg")):
            ann_path = ann_dir / f"{image_path.stem}.png"
            if ann_path.exists():
                samples.append((image_path, ann_path))
        if limit is not None:
            samples = samples[:limit]
        if not samples:
            raise RuntimeError(f"No ADE20K samples found for split={split}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.train:
            scale = random.uniform(0.5, 2.0)
            target_short = max(1, int(self.image_size * scale))
        else:
            target_short = self.image_size
        width, height = image.size
        if width < height:
            new_w = target_short
            new_h = int(round(height * target_short / width))
        else:
            new_h = target_short
            new_w = int(round(width * target_short / height))
        image = image.resize((new_w, new_h), Image.BICUBIC)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        return image, mask

    def _crop(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        crop = self.crop_size
        width, height = image.size
        pad_w = max(crop - width, 0)
        pad_h = max(crop - height, 0)
        if pad_w or pad_h:
            image = TVF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TVF.pad(mask, [0, 0, pad_w, pad_h], fill=0)
            width, height = image.size
        if self.train:
            left = random.randint(0, width - crop)
            top = random.randint(0, height - crop)
        else:
            left = (width - crop) // 2
            top = (height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop))
        mask = mask.crop((left, top, left + crop, top + crop))
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, ann_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(ann_path)
        image, mask = self._resize(image, mask)
        image, mask = self._crop(image, mask)
        if self.train and random.random() < 0.5:
            image = TVF.hflip(image)
            mask = TVF.hflip(mask)
        tensor = TVF.to_tensor(image)
        tensor = TVF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = ade_label_to_train_id(mask)
        return tensor, target, image_path.stem


class FrozenDinoLinearSeg(nn.Module):
    def __init__(self, encoder: nn.Module, layers: list[int], num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.layers = layers
        embed_dim = int(getattr(encoder, "embed_dim"))
        self.head = nn.Conv2d(embed_dim * len(layers), num_classes, kernel_size=1)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder.get_intermediate_layers(
                x,
                n=self.layers,
                reshape=True,
                return_class_token=False,
            )
            if isinstance(features, tuple):
                features = list(features)
            features = [feat.detach() for feat in features]
            dense = torch.cat(features, dim=1)
        return self.head(dense)


def build_dinov3_encoder(args: argparse.Namespace) -> nn.Module:
    size = args.encoder.split("_", 1)[1]
    model_name = SIZE_TO_MODEL[size]
    builder = _load_dinov3_backbone_builder(model_name)
    model = builder(pretrained=False)
    weights_path = _resolve_weights_path(model_name, args.pretrained_encoder)
    state_dict = _load_state_dict(weights_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def fast_hist(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = (target >= 0) & (target < num_classes)
    if valid.sum() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=pred.device)
    index = target[valid] * num_classes + pred[valid]
    hist = torch.bincount(index, minlength=num_classes * num_classes)
    return hist.reshape(num_classes, num_classes).double()


def compute_scores(hist: torch.Tensor) -> tuple[float, float, float, int]:
    diag = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - diag
    acc_den = hist.sum(1)
    valid_iou = union > 0
    valid_acc = acc_den > 0
    iou = diag[valid_iou] / union[valid_iou].clamp_min(1)
    acc = diag[valid_acc] / acc_den[valid_acc].clamp_min(1)
    pixel_acc = diag.sum() / hist.sum().clamp_min(1)
    return (
        float(iou.mean().item()) if iou.numel() else 0.0,
        float(acc.mean().item()) if acc.numel() else 0.0,
        float(pixel_acc.item()),
        int(valid_iou.sum().item()),
    )


def colorize_mask(mask: np.ndarray, num_classes: int = 150) -> Image.Image:
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < num_classes)
    out[valid] = palette[mask[valid]]
    return Image.fromarray(out)


@torch.no_grad()
def evaluate(
    model: FrozenDinoLinearSeg,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    output_dir: Path,
    visualize: int,
) -> EvalResult:
    model.eval()
    hist = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    total_loss = 0.0
    num_batches = 0
    vis_dir = output_dir / "visualizations"
    if visualize > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for images, targets, names in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        logits_up = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(logits_up, targets, ignore_index=IGNORE_INDEX)
        preds = logits_up.argmax(dim=1)
        hist += fast_hist(preds.reshape(-1), targets.reshape(-1), num_classes)
        total_loss += float(loss.item())
        num_batches += 1

        if saved < visualize:
            mean = torch.tensor(IMAGENET_MEAN, device=images.device)[:, None, None]
            std = torch.tensor(IMAGENET_STD, device=images.device)[:, None, None]
            for i in range(images.shape[0]):
                if saved >= visualize:
                    break
                image = (images[i] * std + mean).clamp(0, 1).cpu()
                image_pil = TVF.to_pil_image(image)
                gt = colorize_mask(targets[i].cpu().numpy(), num_classes)
                pred = colorize_mask(preds[i].cpu().numpy(), num_classes)
                canvas = Image.new("RGB", (image_pil.width * 3, image_pil.height))
                canvas.paste(image_pil, (0, 0))
                canvas.paste(gt, (image_pil.width, 0))
                canvas.paste(pred, (image_pil.width * 2, 0))
                canvas.save(vis_dir / f"{saved:04d}_{names[i]}.jpg", quality=92)
                saved += 1

    miou, mean_acc, pixel_acc, valid_classes = compute_scores(hist)
    return EvalResult(
        miou=miou,
        mean_acc=mean_acc,
        pixel_acc=pixel_acc,
        valid_classes=valid_classes,
        loss=total_loss / max(num_batches, 1),
    )


def train_one_epoch(
    model: FrozenDinoLinearSeg,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool,
) -> float:
    model.train()
    model.encoder.eval()
    total_loss = 0.0
    start = time.time()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    for step, (images, targets, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
            target_small = F.interpolate(
                targets.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()
            loss = F.cross_entropy(logits, target_small, ignore_index=IGNORE_INDEX)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        if step == 1 or step % 50 == 0:
            elapsed = time.time() - start
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={total_loss / step:.4f} time={elapsed:.1f}s",
                flush=True,
            )
    return total_loss / max(len(loader), 1)


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data_root = Path(args.data_root).expanduser().resolve()
    val_set = ADE20KDataset(
        data_root,
        split="validation",
        image_size=args.image_size,
        crop_size=args.crop_size,
        train=False,
        limit=args.val_limit,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    encoder = build_dinov3_encoder(args)
    model = FrozenDinoLinearSeg(encoder, layers=args.layers, num_classes=args.num_classes).to(device)
    if args.eval_head is not None:
        checkpoint = torch.load(Path(args.eval_head).expanduser(), map_location="cpu")
        state_dict = checkpoint["head"] if isinstance(checkpoint, dict) and "head" in checkpoint else checkpoint
        model.head.load_state_dict(state_dict, strict=True)
        print(
            f"data_root={data_root} val={len(val_set)} encoder={args.encoder} "
            f"layers={args.layers} checkpoint={args.pretrained_encoder} eval_head={args.eval_head}",
            flush=True,
        )
        result = evaluate(model, val_loader, device, args.num_classes, output_dir, visualize=args.visualize)
        best = {"epoch": checkpoint.get("best", {}).get("epoch") if isinstance(checkpoint, dict) else None, **asdict(result)}
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump({"best": best, "args": vars(args)}, handle, indent=2, sort_keys=True)
        print(
            f"eval-head mIoU={result.miou:.4f} mAcc={result.mean_acc:.4f} "
            f"pixAcc={result.pixel_acc:.4f} loss={result.loss:.4f}",
            flush=True,
        )
        print(f"saved to {output_dir}", flush=True)
        return

    train_set = ADE20KDataset(
        data_root,
        split="training",
        image_size=args.image_size,
        crop_size=args.crop_size,
        train=True,
        limit=args.train_limit,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"data_root={data_root} train={len(train_set)} val={len(val_set)} "
        f"encoder={args.encoder} layers={args.layers} checkpoint={args.pretrained_encoder}",
        flush=True,
    )

    best = None
    history_path = output_dir / "history.csv"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.amp)
        row = {"epoch": epoch, "train_loss": train_loss}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            result = evaluate(
                model,
                val_loader,
                device,
                args.num_classes,
                output_dir,
                visualize=args.visualize if epoch == args.epochs else 0,
            )
            row.update(asdict(result))
            print(
                f"eval epoch={epoch} mIoU={result.miou:.4f} "
                f"mAcc={result.mean_acc:.4f} pixAcc={result.pixel_acc:.4f} "
                f"loss={result.loss:.4f}",
                flush=True,
            )
            if best is None or result.miou > best["miou"]:
                best = {"epoch": epoch, **asdict(result)}
                torch.save({"head": model.head.state_dict(), "args": vars(args), "best": best}, output_dir / "best_head.pt")
        append_history(history_path, row)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"best": best, "args": vars(args)}, handle, indent=2, sort_keys=True)
    print(f"best={best}", flush=True)
    print(f"saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
