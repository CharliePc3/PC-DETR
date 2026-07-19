# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from peft import PeftModel

from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.dinov3 import DinoV3Backbone
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.semantic_reassembly_projector import ScaleDecoupledReassemblyProjector
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import NestedTensor

logger = get_logger()

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""

    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,
        drop_path=0.0,
        out_channels=256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple[int, int] = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_encoder_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 0,
        register_border_tokens: int = 0,
        register_fill: str = "randn",
        register_noise_std: float = 1.0,
        feature_adapter: str = "none",
        feature_adapter_init_scale: float = 1.0,
        projector_type: str = "multiscale",
        sdsr_detail_channels: int = 32,
        sdsr_use_local_reassembly: bool = True,
        sdsr_use_directional_guide: bool = True,
        sdsr_use_phase_downsample: bool = True,
    ):
        super().__init__()
        self.name = name
        name_parts = name.split("_")
        if len(name_parts) != 2 or name_parts[0] != "dinov3":
            raise ValueError("RF-DETR-DINOv3 only supports encoder names of the form dinov3_<size>.")
        self.encoder = DinoV3Backbone(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            pretrained_encoder=pretrained_encoder,
            load_pretrained=load_encoder_weights,
            register_border_tokens=register_border_tokens,
            register_fill=register_fill,
            register_noise_std=register_noise_std,
            feature_adapter=feature_adapter,
            feature_adapter_init_scale=feature_adapter_init_scale,
        )
        # build encoder + projector as backbone module
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
        assert sorted(self.projector_scale) == self.projector_scale, (
            "only support projector scale P3/P4/P5/P6 in ascending order."
        )
        self.projector_type = projector_type
        if projector_type == "multiscale":
            level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
            scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]
            self.projector = MultiScaleProjector(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                scale_factors=scale_factors,
                layer_norm=layer_norm,
                rms_norm=rms_norm,
            )
        elif projector_type == "sdsr":
            self.projector = ScaleDecoupledReassemblyProjector(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                detail_channels=sdsr_detail_channels,
                use_local_reassembly=sdsr_use_local_reassembly,
                use_directional_guide=sdsr_use_directional_guide,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights")
            self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor):
        """ """
        # (H, W, B, C)
        feats = self.encoder(tensor_list.tensors)
        if self.projector_type == "sdsr":
            feats = self.projector(feats, image=tensor_list.tensors, mask=tensor_list.mask)
        else:
            feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        if self.projector_type == "sdsr":
            feats = self.projector(feats, image=tensors)
        else:
            feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_vit_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_vit_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name or "patch_embed" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name, weight_decay_rate=1.0):
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate
