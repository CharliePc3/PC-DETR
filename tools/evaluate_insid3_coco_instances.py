#!/usr/bin/env python3
"""Evaluate two INSID3-to-COCO instance-segmentation protocols.

``full-image`` is a pure K-shot protocol: category supports from COCO train
vote on complete COCO val images and connected components become instances.

``detector-guided`` uses RF-DETR class/box/score predictions as instance
prompts. INSID3 runs on each expanded detector crop using instance-level train
supports, and the selected crop mask is pasted back into the original image.

Both modes keep support annotations in train and use val ground truth only in
COCOeval. They are research baselines, not the official INSID3 benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as TVF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3")
os.environ.setdefault(
    "DINOV3_WEIGHTS_DIR", str(PROJECT_ROOT / "weights" / "dinov3")
)

from rfdetr.models import PostProcess, build_model  # noqa: E402
from rfdetr.main import populate_args  # noqa: E402
from rfdetr.models.backbone.dinov3 import (  # noqa: E402
    SIZE_TO_MODEL,
    _load_dinov3_backbone_builder,
    _load_state_dict,
    _resolve_weights_path,
)
from rfdetr.models.insid3 import DinoV3InContextSegmenter  # noqa: E402
from rfdetr.models.insid3_coco import (  # noqa: E402
    DetectionGuidedInContextSegmenter,
    FullImageFewShotInstanceSegmenter,
    InContextSupport,
    InstanceMaskPrediction,
    expanded_integer_box,
    paste_crop_mask,
    predictions_to_coco_results,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SupportSpec:
    category_id: int
    image_id: int
    annotation_id: int | None
    crop_box: tuple[int, int, int, int] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate INSID3 full-image or detector-guided COCO instances"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("full-image", "detector-guided"),
    )
    parser.add_argument(
        "--data-root",
        default="/data/cpc/root/dataset/COCO_RFDETR_TEST/medium",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-size", default="small", choices=tuple(SIZE_TO_MODEL))
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["person", "car", "dog", "cat", "bicycle"],
        help="COCO category names to evaluate; pass the desired full list explicitly.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--svd-components", type=int, default=128)
    parser.add_argument("--cluster-similarity", type=float, default=0.6)
    parser.add_argument("--merge-threshold", type=float, default=0.2)
    parser.add_argument("--support-vote-threshold", type=float, default=0.5)
    parser.add_argument("--score-temperature", type=float, default=0.1)
    parser.add_argument("--min-component-patches", type=int, default=1)
    parser.add_argument("--max-instances-per-category", type=int, default=100)
    parser.add_argument("--support-min-fraction", type=float, default=0.01)
    parser.add_argument("--support-max-fraction", type=float, default=0.8)
    parser.add_argument("--support-box-expansion", type=float, default=0.15)
    parser.add_argument("--detector-checkpoint", default=None)
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--target-box-expansion", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size < 16 or args.image_size % 16:
        raise ValueError("--image-size must be a positive multiple of 16.")
    if args.shots < 1 or args.feature_batch_size < 1:
        raise ValueError("--shots and --feature-batch-size must be positive.")
    if args.max_images < 0 or args.max_detections < 1:
        raise ValueError("image/detection limits must be non-negative/positive.")
    if not 0 < args.support_min_fraction < args.support_max_fraction <= 1:
        raise ValueError("support fraction bounds must satisfy 0 < min < max <= 1.")
    if args.mode == "detector-guided" and not args.detector_checkpoint:
        raise ValueError("--detector-checkpoint is required in detector-guided mode.")


def build_encoder(args: argparse.Namespace) -> torch.nn.Module:
    model_name = SIZE_TO_MODEL[args.model_size]
    encoder = _load_dinov3_backbone_builder(model_name)(pretrained=False)
    weights_path = _resolve_weights_path(model_name, args.pretrained_encoder)
    encoder.load_state_dict(_load_state_dict(weights_path), strict=True)
    return encoder.eval()


def normalized_square_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BICUBIC
    )
    return TVF.normalize(TVF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)


def resized_mask(mask: np.ndarray, image_size: int) -> torch.Tensor:
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (image_size, image_size), Image.Resampling.NEAREST
    )
    return torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()) > 127


def category_union_mask(coco: COCO, image_id: int, category_id: int) -> np.ndarray:
    info = coco.loadImgs([image_id])[0]
    mask = np.zeros((int(info["height"]), int(info["width"])), dtype=np.uint8)
    annotation_ids = coco.getAnnIds(
        imgIds=[image_id], catIds=[category_id], iscrowd=False
    )
    for annotation in coco.loadAnns(annotation_ids):
        mask |= coco.annToMask(annotation).astype(np.uint8)
    return mask


def full_image_support_specs(
    coco: COCO,
    category_ids: Sequence[int],
    shots: int,
    min_fraction: float,
    max_fraction: float,
) -> list[SupportSpec]:
    specs: list[SupportSpec] = []
    for category_id in category_ids:
        candidates: list[tuple[float, int]] = []
        for image_id in coco.getImgIds(catIds=[category_id]):
            mask = category_union_mask(coco, image_id, category_id)
            fraction = float(mask.mean())
            if min_fraction <= fraction <= max_fraction:
                candidates.append((abs(fraction - 0.2), int(image_id)))
        candidates.sort()
        if len(candidates) < shots:
            raise RuntimeError(
                f"Category {category_id} has {len(candidates)} suitable full-image "
                f"supports, fewer than --shots={shots}."
            )
        specs.extend(
            SupportSpec(category_id, image_id, None, None)
            for _, image_id in candidates[:shots]
        )
    return specs


def annotation_xyxy(annotation: dict) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in annotation["bbox"])
    return x, y, x + width, y + height


def crop_support_specs(
    coco: COCO,
    category_ids: Sequence[int],
    shots: int,
    min_fraction: float,
    max_fraction: float,
    expansion: float,
) -> list[SupportSpec]:
    specs: list[SupportSpec] = []
    for category_id in category_ids:
        candidates: list[tuple[float, int, int, tuple[int, int, int, int]]] = []
        annotation_ids = coco.getAnnIds(catIds=[category_id], iscrowd=False)
        for annotation in coco.loadAnns(annotation_ids):
            image_id = int(annotation["image_id"])
            info = coco.loadImgs([image_id])[0]
            crop_box = expanded_integer_box(
                annotation_xyxy(annotation),
                (int(info["height"]), int(info["width"])),
                expansion=expansion,
            )
            mask = coco.annToMask(annotation).astype(np.uint8)
            left, top, right, bottom = crop_box
            crop_mask = mask[top:bottom, left:right]
            fraction = float(crop_mask.mean()) if crop_mask.size else 0.0
            if min_fraction <= fraction <= max_fraction:
                candidates.append(
                    (
                        -float(annotation.get("area", crop_mask.sum())),
                        int(annotation["id"]),
                        image_id,
                        crop_box,
                    )
                )
        candidates.sort()
        if len(candidates) < shots:
            raise RuntimeError(
                f"Category {category_id} has {len(candidates)} suitable crop "
                f"supports, fewer than --shots={shots}."
            )
        specs.extend(
            SupportSpec(category_id, image_id, annotation_id, crop_box)
            for _, annotation_id, image_id, crop_box in candidates[:shots]
        )
    return specs


def materialize_support_inputs(
    specs: Sequence[SupportSpec],
    coco: COCO,
    data_root: Path,
    image_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for spec in specs:
        info = coco.loadImgs([spec.image_id])[0]
        image = Image.open(data_root / "train2017" / info["file_name"]).convert("RGB")
        if spec.annotation_id is None:
            mask = category_union_mask(coco, spec.image_id, spec.category_id)
        else:
            annotation = coco.loadAnns([spec.annotation_id])[0]
            mask = coco.annToMask(annotation).astype(np.uint8)
        if spec.crop_box is not None:
            left, top, right, bottom = spec.crop_box
            image = image.crop((left, top, right, bottom))
            mask = mask[top:bottom, left:right]
        images.append(normalized_square_tensor(image, image_size))
        masks.append(resized_mask(mask, image_size))
    return images, masks


@torch.no_grad()
def extract_support_bank(
    segmenter: DinoV3InContextSegmenter,
    specs: Sequence[SupportSpec],
    image_tensors: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> list[InContextSupport]:
    supports: list[InContextSupport] = []
    for start in range(0, len(specs), batch_size):
        stop = min(start + batch_size, len(specs))
        features = segmenter.extract_features(
            torch.stack(list(image_tensors[start:stop])).to(device)
        )
        for local_index, feature in enumerate(features):
            index = start + local_index
            spec = specs[index]
            supports.append(
                InContextSupport(
                    category_id=spec.category_id,
                    features=feature,
                    mask=masks[index].to(device),
                    image_id=spec.image_id,
                    annotation_id=spec.annotation_id,
                )
            )
    return supports


@torch.no_grad()
def extract_feature_maps(
    segmenter: DinoV3InContextSegmenter,
    image_tensors: Sequence[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    """Extract an arbitrary tensor list in bounded batches."""

    features: list[torch.Tensor] = []
    for start in range(0, len(image_tensors), batch_size):
        stop = min(start + batch_size, len(image_tensors))
        batch_features = segmenter.extract_features(
            torch.stack(list(image_tensors[start:stop])).to(device)
        )
        features.extend(batch_features.unbind(0))
    return features


def selected_categories(coco: COCO, names: Sequence[str] | None) -> list[int]:
    categories = coco.loadCats(coco.getCatIds())
    if names is None:
        return sorted(int(category["id"]) for category in categories)
    by_name = {str(category["name"]): int(category["id"]) for category in categories}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"Unknown COCO categories: {missing}")
    return [by_name[name] for name in names]


def checkpoint_kwargs(saved_args: object) -> dict[str, object]:
    if isinstance(saved_args, argparse.Namespace):
        return copy.deepcopy(vars(saved_args))
    if isinstance(saved_args, dict):
        return copy.deepcopy(saved_args)
    if hasattr(saved_args, "__dict__"):
        return copy.deepcopy(vars(saved_args))
    raise TypeError("checkpoint args must be a namespace-like object.")


def load_detector(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, PostProcess, argparse.Namespace]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in checkpoint or "model" not in checkpoint:
        raise KeyError("detector checkpoint must contain 'args' and 'model'.")
    # Populate current defaults around older checkpoint args. The project adds
    # experimental configuration fields frequently, while the state dict is
    # still fully strict once the matching architecture has been rebuilt.
    detector_args = populate_args(**checkpoint_kwargs(checkpoint["args"]))
    detector_args.device = str(device)
    model = build_model(detector_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    postprocess = PostProcess(num_select=int(detector_args.num_select))
    return model, postprocess, detector_args


@torch.no_grad()
def detector_predictions(
    model: torch.nn.Module,
    postprocess: PostProcess,
    detector_args: argparse.Namespace,
    image: Image.Image,
    device: torch.device,
    threshold: float,
    max_detections: int,
    contiguous_to_category: Sequence[int],
    allowed_category_ids: set[int] | None = None,
) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    original_width, original_height = image.size
    tensor = normalized_square_tensor(image, int(detector_args.resolution)).to(device)
    outputs = model(tensor[None])
    result = postprocess(
        outputs,
        torch.tensor([[original_height, original_width]], device=device),
    )[0]
    order = result["scores"].argsort(descending=True)
    detections = []
    for index in order:
        score = float(result["scores"][index])
        if score < threshold or len(detections) >= max_detections:
            break
        contiguous_label = int(result["labels"][index])
        if not 0 <= contiguous_label < len(contiguous_to_category):
            continue
        category_id = int(contiguous_to_category[contiguous_label])
        if allowed_category_ids is not None and category_id not in allowed_category_ids:
            continue
        box = tuple(float(value) for value in result["boxes"][index])
        detections.append((category_id, score, box))
    return detections


def evaluate_results(
    coco: COCO,
    results: list[dict[str, object]],
    image_ids: Sequence[int],
    category_ids: Sequence[int],
) -> list[float]:
    if not results:
        return [0.0] * 12
    detections = coco.loadRes(results)
    evaluator = COCOeval(coco, detections, "segm")
    evaluator.params.imgIds = list(image_ids)
    evaluator.params.catIds = list(category_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats.tolist()


def run_full_image(
    args: argparse.Namespace,
    segmenter: DinoV3InContextSegmenter,
    supports: Sequence[InContextSupport],
    positional_basis: torch.Tensor,
    val_coco: COCO,
    data_root: Path,
    image_ids: Sequence[int],
    device: torch.device,
) -> list[dict[str, object]]:
    predictor = FullImageFewShotInstanceSegmenter(
        segmenter,
        num_debias_components=args.svd_components,
        cluster_similarity_threshold=args.cluster_similarity,
        merge_threshold=args.merge_threshold,
        vote_threshold=args.support_vote_threshold,
        score_temperature=args.score_temperature,
        min_component_patches=args.min_component_patches,
    )
    results: list[dict[str, object]] = []
    for index, image_id in enumerate(image_ids):
        info = val_coco.loadImgs([image_id])[0]
        image = Image.open(data_root / "val2017" / info["file_name"]).convert("RGB")
        features = segmenter.extract_features(
            normalized_square_tensor(image, args.image_size)[None].to(device)
        )[0]
        predictions = predictor.predict(
            supports,
            features,
            (int(info["height"]), int(info["width"])),
            positional_basis=positional_basis,
            max_instances_per_category=args.max_instances_per_category,
        )
        results.extend(predictions_to_coco_results(image_id, predictions))
        print(
            f"[{index + 1}/{len(image_ids)}] image={image_id} "
            f"instances={len(predictions)}",
            flush=True,
        )
    return results


def run_detector_guided(
    args: argparse.Namespace,
    segmenter: DinoV3InContextSegmenter,
    supports: Sequence[InContextSupport],
    positional_basis: torch.Tensor,
    val_coco: COCO,
    data_root: Path,
    image_ids: Sequence[int],
    category_ids: Sequence[int],
    device: torch.device,
) -> list[dict[str, object]]:
    detector, postprocess, detector_args = load_detector(
        Path(args.detector_checkpoint).expanduser().resolve(), device
    )
    predictor = DetectionGuidedInContextSegmenter(
        segmenter,
        num_debias_components=args.svd_components,
        cluster_similarity_threshold=args.cluster_similarity,
        merge_threshold=args.merge_threshold,
        vote_threshold=args.support_vote_threshold,
        score_temperature=args.score_temperature,
        min_component_patches=args.min_component_patches,
    )
    supported = set(category_ids)
    contiguous_to_category = sorted(int(value) for value in val_coco.getCatIds())
    results: list[dict[str, object]] = []
    for image_index, image_id in enumerate(image_ids):
        info = val_coco.loadImgs([image_id])[0]
        image = Image.open(data_root / "val2017" / info["file_name"]).convert("RGB")
        image_size = (int(info["height"]), int(info["width"]))
        detections = detector_predictions(
            detector,
            postprocess,
            detector_args,
            image,
            device,
            args.detector_threshold,
            args.max_detections,
            contiguous_to_category,
            supported,
        )
        crop_records = []
        crop_tensors = []
        for category_id, score, box in detections:
            try:
                crop_box = expanded_integer_box(
                    box, image_size, expansion=args.target_box_expansion
                )
            except ValueError:
                continue
            crop_records.append((category_id, score, crop_box))
            crop_tensors.append(
                normalized_square_tensor(image.crop(crop_box), args.image_size)
            )
        crop_features = extract_feature_maps(
            segmenter,
            crop_tensors,
            device=device,
            batch_size=args.feature_batch_size,
        )
        predictions: list[InstanceMaskPrediction] = []
        for (category_id, score, crop_box), features in zip(
            crop_records, crop_features
        ):
            crop_prediction = predictor.predict_crop(
                category_id,
                score,
                supports,
                features,
                (crop_box[3] - crop_box[1], crop_box[2] - crop_box[0]),
                positional_basis=positional_basis,
            )
            if crop_prediction is None:
                continue
            crop_prediction.mask = paste_crop_mask(
                crop_prediction.mask, crop_box, image_size
            )
            predictions.append(crop_prediction)
        results.extend(predictions_to_coco_results(image_id, predictions))
        print(
            f"[{image_index + 1}/{len(image_ids)}] image={image_id} "
            f"detections={len(detections)} masks={len(predictions)}",
            flush=True,
        )
    return results


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_coco = COCO(
        str(data_root / "annotations" / "instances_train2017.json")
    )
    val_coco = COCO(str(data_root / "annotations" / "instances_val2017.json"))
    category_ids = selected_categories(train_coco, args.categories)

    encoder = build_encoder(args).to(device)
    segmenter = DinoV3InContextSegmenter(encoder).to(device)
    positional_basis = segmenter.build_positional_basis(
        (args.image_size, args.image_size),
        args.svd_components,
        device=device,
    )
    if args.svd_components > positional_basis.shape[1]:
        raise ValueError(
            f"Requested {args.svd_components} components, but the encoder "
            f"supports at most {positional_basis.shape[1]}."
        )

    if args.mode == "full-image":
        specs = full_image_support_specs(
            train_coco,
            category_ids,
            args.shots,
            args.support_min_fraction,
            args.support_max_fraction,
        )
    else:
        specs = crop_support_specs(
            train_coco,
            category_ids,
            args.shots,
            args.support_min_fraction,
            args.support_max_fraction,
            args.support_box_expansion,
        )
    support_images, support_masks = materialize_support_inputs(
        specs, train_coco, data_root, args.image_size
    )
    supports = extract_support_bank(
        segmenter,
        specs,
        support_images,
        support_masks,
        device=device,
        batch_size=args.feature_batch_size,
    )

    image_ids = sorted(int(image_id) for image_id in val_coco.getImgIds())
    if args.max_images:
        image_ids = image_ids[: args.max_images]
    if args.mode == "full-image":
        results = run_full_image(
            args,
            segmenter,
            supports,
            positional_basis,
            val_coco,
            data_root,
            image_ids,
            device,
        )
    else:
        results = run_detector_guided(
            args,
            segmenter,
            supports,
            positional_basis,
            val_coco,
            data_root,
            image_ids,
            category_ids,
            device,
        )

    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(results, ensure_ascii=False), encoding="utf-8"
    )
    stats = evaluate_results(val_coco, results, image_ids, category_ids)
    summary = {
        "scope": (
            "Project-level INSID3-to-COCO instance protocol; support annotations "
            "come only from train and target masks are used only by COCOeval."
        ),
        "mode": args.mode,
        "args": vars(args),
        "category_ids": category_ids,
        "image_ids": image_ids,
        "supports": [asdict(spec) for spec in specs],
        "num_predictions": len(results),
        "coco_segm_stats": stats,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
