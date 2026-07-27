"""Adapter exposing the UniRefiner method as a composable training objective."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision.transforms import functional as TVF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class UniRefinerObjective:
    """Reuse UniRefiner filtering/register losses inside the unified trainer."""

    def __init__(self, args, student, teacher, device: torch.device) -> None:
        root = Path(args.unirefiner_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from unirefiner.method import UniRefinerMethod
        from unirefiner.models.wrappers.dinov3 import wrap_dinov3

        self.method = UniRefinerMethod()
        self.student = wrap_dinov3(student)
        self.teacher = wrap_dinov3(teacher)
        self.device = device
        self.weight = float(args.lambda_unirefiner)
        self.cast_dtype = None
        if device.type == "cuda" and args.amp:
            self.cast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

        self.runtime = SimpleNamespace(
            reg_factor=args.uni_reg_factor,
            register_fill=args.uni_register_fill,
            num_proposals=args.uni_num_proposals,
            fp_gp_sigma=args.uni_fp_gp_sigma,
            fp_gp_cosine_threshold=args.uni_fp_gp_cosine_threshold,
            adaptive_spurious_detector_cosine_threshold=args.uni_adaptive_register_threshold,
            disable_attention_hijack_filter=args.uni_disable_attention_hijack_filter,
            attention_hijack_layer_start=args.uni_attention_hijack_layer_start,
            attention_hijack_layer_end=args.uni_attention_hijack_layer_end,
            attention_hijack_sigma=args.uni_attention_hijack_sigma,
            disable_student_teacher_matching=args.uni_disable_student_teacher_matching,
            uniformity_strength=args.uni_uniformity_strength,
            spatial_consistency_start_stage=args.uni_scd_start_stage,
            spatial_consistency_weight=args.uni_scd_weight,
            channel_mask_channels=[],
            distributed=False,
            rank=0,
            train_stage=0.0,
            global_step=0,
            vis_pca_interval=0,
            vis_pca_test_image=None,
            vis_pca_save_dir=None,
            wandb_run=None,
        )
        self.background = self._load_background(
            Path(args.unirefiner_background),
            args.resolution,
        ).to(device=device)

    @staticmethod
    def _load_background(path: Path, resolution: int) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"UniRefiner background image not found: {path}")
        image = Image.open(path).convert("RGB").resize(
            (resolution, resolution),
            Image.BICUBIC,
        )
        tensor = TVF.normalize(TVF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return tensor.unsqueeze(0)

    def __call__(
        self,
        images: torch.Tensor,
        *,
        train_stage: float,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self.runtime.train_stage = float(train_stage)
        self.runtime.global_step = int(global_step)
        background = self.background.expand(images.shape[0], -1, -1, -1)
        losses, _batch_size = self.method(
            (images, background),
            self.student,
            self.teacher,
            self.device,
            self.cast_dtype,
            False,
            self.runtime,
        )
        raw_loss = losses["loss_final"]
        stats = {
            "loss_unirefiner": float(raw_loss.detach().item()),
            "uni_nce": float(losses["nce_loss"].detach().item()),
            "uni_scd": float(losses["loss_scd"].detach().item()),
            "uni_align_reg": float(losses["align_reg"].detach().item()),
            "uni_uniform_reg": float(losses["uniform_reg"].detach().item()),
            "uni_useful_ratio": float(losses["useful_ratio"].detach().item()),
        }
        return self.weight * raw_loss, stats
