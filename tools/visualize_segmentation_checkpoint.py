#!/usr/bin/env python3
"""Render COCO ground truth and RF-DETR instance-mask predictions side by side."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torchvision.transforms import functional as TVF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3")
os.environ.setdefault("DINOV3_WEIGHTS_DIR", str(PROJECT_ROOT / "weights" / "dinov3"))

from rfdetr.models import PostProcess, build_model  # noqa: E402
from rfdetr.util.coco_classes import COCO_CLASSES  # noqa: E402


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize an RF-DETR segmentation checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-root",
        default="/data/cpc/root/dataset/COCO_RFDETR_TEST/overfit",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-predictions", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def color_for_label(label: int) -> tuple[int, int, int]:
    generator = np.random.default_rng(label * 7919 + 17)
    return tuple(int(value) for value in generator.integers(48, 240, size=3))


def blend_mask(canvas: np.ndarray, mask: np.ndarray, color, alpha: float = 0.42) -> None:
    if not mask.any():
        return
    canvas[mask] = (
        (1.0 - alpha) * canvas[mask] + alpha * np.asarray(color, dtype=np.float32)
    )


def render_instances(
    image: Image.Image,
    instances: list[dict],
) -> Image.Image:
    canvas = np.asarray(image, dtype=np.float32).copy()
    for instance in instances:
        blend_mask(canvas, instance["mask"], color_for_label(instance["label"]))
    output = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    for instance in instances:
        color = color_for_label(instance["label"])
        box = tuple(float(value) for value in instance["box"])
        draw.rectangle(box, outline=color, width=3)
        name = COCO_CLASSES.get(instance["label"], str(instance["label"]))
        score = instance.get("score")
        text = name if score is None else f"{name} {score:.2f}"
        draw.text((box[0] + 2, box[1] + 2), text, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
    return output


def ground_truth_instances(coco: COCO, image_id: int) -> list[dict]:
    instances = []
    for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False)):
        x, y, width, height = annotation["bbox"]
        instances.append(
            {
                "label": int(annotation["category_id"]),
                "box": (x, y, x + width, y + height),
                "mask": coco.annToMask(annotation).astype(bool),
            }
        )
    return instances


def main() -> None:
    args = parse_args()
    if args.num_images < 1 or args.max_predictions < 1:
        raise ValueError("image and prediction counts must be positive.")
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("score threshold must be in [0, 1].")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    model_args.device = args.device
    model_args.use_cdn = False
    model = build_model(model_args).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    postprocess = PostProcess(num_select=int(model_args.num_select))

    data_root = Path(args.data_root).expanduser().resolve()
    coco = COCO(str(data_root / "annotations" / "instances_val2017.json"))
    image_infos = [
        info
        for info in coco.loadImgs(coco.getImgIds())
        if coco.getAnnIds(imgIds=[info["id"]], iscrowd=False)
    ][: args.num_images]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for image_info in image_infos:
        image_path = data_root / "val2017" / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        tensor = TVF.normalize(TVF.to_tensor(image), MEAN, STD)
        tensor = TVF.resize(
            tensor,
            [int(model_args.resolution), int(model_args.resolution)],
            antialias=True,
        )[None].to(args.device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda"),
        ):
            raw_output = model(tensor)
        target_size = torch.tensor([[image.height, image.width]], device=args.device)
        prediction = postprocess(raw_output, target_size)[0]
        keep = torch.nonzero(
            prediction["scores"] >= args.score_threshold,
            as_tuple=False,
        ).flatten()[: args.max_predictions]
        predicted_instances = [
            {
                "label": int(prediction["labels"][index]),
                "score": float(prediction["scores"][index]),
                "box": prediction["boxes"][index].float().cpu().tolist(),
                "mask": prediction["masks"][index, 0].cpu().numpy().astype(bool),
            }
            for index in keep
        ]
        true_instances = ground_truth_instances(coco, int(image_info["id"]))
        ground_truth_panel = render_instances(image, true_instances)
        prediction_panel = render_instances(image, predicted_instances)
        canvas = Image.new("RGB", (2 * image.width, image.height))
        canvas.paste(ground_truth_panel, (0, 0))
        canvas.paste(prediction_panel, (image.width, 0))
        output_path = output_dir / f"{int(image_info['id']):012d}_gt_pred.jpg"
        canvas.save(output_path, quality=92)
        summaries.append(
            {
                "image_id": int(image_info["id"]),
                "file": output_path.name,
                "ground_truth_instances": len(true_instances),
                "predicted_instances": len(predicted_instances),
            }
        )
        print(json.dumps(summaries[-1]), flush=True)

    (output_dir / "index.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
