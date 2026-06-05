import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from rfdetr.util.logger import get_logger

logger = get_logger()


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DINOV3_REPO_DIR = os.environ.get("DINOV3_REPO_DIR")
DINOV3_WEIGHTS_DIRS = [
    Path(p).expanduser()
    for p in (
        os.environ.get("DINOV3_WEIGHTS_DIR"),
        PROJECT_ROOT / "weights" / "dinov3",
        PROJECT_ROOT / "ckpts",
        Path("/data/cpc/root/storage/dinov3\u6743\u91cd"),
    )
    if p is not None
]

SIZE_TO_MODEL = {
    "small": "dinov3_vits16",
    "splus": "dinov3_vits16plus",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}


def _load_dinov3_backbone_builder(model_name: str):
    try:
        from dinov3.hub import backbones

        return getattr(backbones, model_name)
    except ImportError as import_error:
        if DINOV3_REPO_DIR is None:
            raise ImportError(
                "DINOv3 package is not importable. Install DINOv3 as a Python dependency, "
                "vendor the required DINOv3 code into this project, or set DINOV3_REPO_DIR "
                "to a local DINOv3 source checkout."
            ) from import_error

    repo_path = Path(DINOV3_REPO_DIR).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo_path}")
    repo_dir = str(repo_path)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from dinov3.hub import backbones

    return getattr(backbones, model_name)


def _resolve_weights_path(model_name: str, pretrained_encoder: str | None) -> Path:
    if pretrained_encoder:
        weights_path = Path(pretrained_encoder).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"DINOv3 pretrained encoder not found: {weights_path}")
        return weights_path

    for weights_dir in DINOV3_WEIGHTS_DIRS:
        candidates = sorted(weights_dir.glob(f"{model_name}_pretrain_*.pth"))
        if candidates:
            return candidates[0].resolve()

    search_dirs = ", ".join(str(p) for p in DINOV3_WEIGHTS_DIRS)
    raise FileNotFoundError(
        f"No DINOv3 weights found for {model_name}. Searched: {search_dirs}. "
        "Set pretrained_encoder or DINOV3_WEIGHTS_DIR to the correct checkpoint location."
    )


def _load_state_dict(weights_path: Path):
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "teacher"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    return checkpoint


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
        size: str = "base",
        out_feature_indexes: list[int] | None = None,
        pretrained_encoder: str | None = None,
        load_pretrained: bool = True,
    ):
        super().__init__()
        if out_feature_indexes is None:
            out_feature_indexes = [2, 5, 8, 11]
        if size not in SIZE_TO_MODEL:
            raise ValueError(f"Unsupported DINOv3 size '{size}'. Available: {sorted(SIZE_TO_MODEL)}")

        self.out_feature_indexes = out_feature_indexes
        model_name = SIZE_TO_MODEL[size]

        builder = _load_dinov3_backbone_builder(model_name)
        logger.info("Building DINOv3 backbone: %s", model_name)
        self.encoder = builder(pretrained=False)

        if load_pretrained:
            weights_path = _resolve_weights_path(model_name, pretrained_encoder)
            logger.info("Loading DINOv3 pretrained weights: %s", weights_path)
            state_dict = _load_state_dict(weights_path)
            self.encoder.load_state_dict(state_dict, strict=True)

        self.patch_size = self.encoder.patch_size
        self._out_feature_channels = [self.encoder.embed_dim for _ in out_feature_indexes]

    def forward(self, x):
        height, width = x.shape[-2:]
        assert height % self.patch_size == 0 and width % self.patch_size == 0, (
            f"DINOv3 requires input height/width to be divisible by patch size {self.patch_size}, "
            f"but got {tuple(x.shape)}"
        )
        return list(
            self.encoder.get_intermediate_layers(
                x,
                n=self.out_feature_indexes,
                reshape=True,
                return_class_token=False,
            )
        )
