from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from PIL import Image


SPLIT_MAP = {
    "train": "train",
    "val": "valid",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert mini-ImageNet pickle caches to RF-DETR Roboflow COCO format.")
    parser.add_argument("--input-dir", type=Path, default=Path("/data/cpc/root/dataset/miniimagenet"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/cpc/root/dataset/miniimagenet_rfdetr"))
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_split(input_dir: Path, split: str) -> dict:
    path = input_dir / f"mini-imagenet-cache-{split}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing mini-ImageNet cache: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def collect_categories(input_dir: Path) -> list[str]:
    names: set[str] = set()
    for split in SPLIT_MAP:
        data = load_split(input_dir, split)
        names.update(data["class_dict"].keys())
    return sorted(names)


def save_image(array, path: Path, image_format: str, jpeg_quality: int) -> None:
    image = Image.fromarray(array)
    if image_format == "jpg":
        image.save(path, quality=jpeg_quality, subsampling=0)
    else:
        image.save(path)


def convert_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    output_split: str,
    class_to_category_id: dict[str, int],
    categories: list[dict],
    image_format: str,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[int, int]:
    data = load_split(input_dir, split)
    images_array = data["image_data"]
    class_dict = data["class_dict"]

    split_dir = output_dir / output_split
    split_dir.mkdir(parents=True, exist_ok=True)
    ann_path = split_dir / "_annotations.coco.json"
    if ann_path.exists() and not overwrite:
        raise FileExistsError(f"{ann_path} already exists. Pass --overwrite to regenerate.")

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    image_id = 1
    height = int(images_array.shape[1])
    width = int(images_array.shape[2])

    for class_name in sorted(class_dict):
        category_id = class_to_category_id[class_name]
        for source_index in class_dict[class_name]:
            filename = f"{split}_{category_id:03d}_{int(source_index):05d}.{image_format}"
            image_path = split_dir / filename
            if overwrite or not image_path.exists():
                save_image(images_array[source_index], image_path, image_format, jpeg_quality)

            coco_images.append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": width,
                    "height": height,
                }
            )
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [0, 0, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "segmentation": [[0, 0, width, 0, width, height, 0, height]],
                }
            )
            image_id += 1
            annotation_id += 1

    coco = {
        "info": {
            "description": "mini-ImageNet converted to RF-DETR-compatible COCO format",
            "source_format": "mini-imagenet pickle cache",
            "bbox_policy": "single full-image box per classification image",
        },
        "licenses": [],
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    with ann_path.open("w") as handle:
        json.dump(coco, handle)

    return len(coco_images), len(coco_annotations)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_names = collect_categories(args.input_dir)
    class_to_category_id = {name: idx + 1 for idx, name in enumerate(class_names)}
    categories = [
        {
            "id": class_to_category_id[name],
            "name": name,
            "supercategory": "miniimagenet",
        }
        for name in class_names
    ]

    for split, output_split in SPLIT_MAP.items():
        num_images, num_annotations = convert_split(
            args.input_dir,
            args.output_dir,
            split,
            output_split,
            class_to_category_id,
            categories,
            args.image_format,
            args.jpeg_quality,
            args.overwrite,
        )
        print(f"{output_split}: {num_images} images, {num_annotations} annotations")

    metadata_path = args.output_dir / "class_mapping.json"
    with metadata_path.open("w") as handle:
        json.dump(
            {
                "classes": class_names,
                "class_to_category_id": class_to_category_id,
                "category_id_base": 1,
                "bbox_policy": "single full-image box per classification image",
            },
            handle,
            indent=2,
        )
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
