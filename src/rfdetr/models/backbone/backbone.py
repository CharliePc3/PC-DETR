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
from rfdetr.models.backbone.semantic_reassembly_projector import (
    ScaleDecoupledReassemblyProjector,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v2 import (
    ScaleDecoupledReassemblyProjectorV2,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v3 import (
    ScaleDecoupledReassemblyProjectorV3,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v4 import (
    ScaleDecoupledReassemblyProjectorV4,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v5 import (
    ScaleDecoupledReassemblyProjectorV5,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v6 import (
    ScaleDecoupledReassemblyProjectorV6,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v7 import (
    ScaleDecoupledReassemblyProjectorV7,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v8 import (
    ScaleDecoupledReassemblyProjectorV8,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v9 import (
    ScaleDecoupledReassemblyProjectorV9,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v10 import (
    ScaleDecoupledReassemblyProjectorV10,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v11 import (
    ScaleDecoupledReassemblyProjectorV11,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v12 import (
    ScaleDecoupledReassemblyProjectorV12,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v13 import (
    ScaleDecoupledReassemblyProjectorV13,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v14 import (
    ScaleDecoupledReassemblyProjectorV14,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v15 import (
    ScaleDecoupledReassemblyProjectorV15,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v16 import (
    ScaleDecoupledReassemblyProjectorV16,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v17 import (
    ScaleDecoupledReassemblyProjectorV17,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v18 import (
    ScaleDecoupledReassemblyProjectorV18,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v19 import (
    ScaleDecoupledReassemblyProjectorV19,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v20 import (
    ScaleDecoupledReassemblyProjectorV20,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v21 import (
    ScaleDecoupledReassemblyProjectorV21,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v22 import (
    ScaleDecoupledReassemblyProjectorV22,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    ScaleDecoupledReassemblyProjectorV23,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v24 import (
    ScaleDecoupledReassemblyProjectorV24,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v25 import (
    ScaleDecoupledReassemblyProjectorV25,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v26 import (
    ScaleDecoupledReassemblyProjectorV26,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v27 import (
    ScaleDecoupledReassemblyProjectorV27,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v28 import (
    ScaleDecoupledReassemblyProjectorV28,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v29 import (
    ScaleDecoupledReassemblyProjectorV29,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v30 import (
    ScaleDecoupledReassemblyProjectorV30,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v31 import (
    ScaleDecoupledReassemblyProjectorV31,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v32 import (
    ScaleDecoupledReassemblyProjectorV32,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v33 import (
    ScaleDecoupledReassemblyProjectorV33,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v34 import (
    ScaleDecoupledReassemblyProjectorV34,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v35 import (
    ScaleDecoupledReassemblyProjectorV35,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v36 import (
    ScaleDecoupledReassemblyProjectorV36,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v37 import (
    ScaleDecoupledReassemblyProjectorV37R1,
    ScaleDecoupledReassemblyProjectorV37R2,
    ScaleDecoupledReassemblyProjectorV37R3,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v38 import (
    ScaleDecoupledReassemblyProjectorV38Both,
    ScaleDecoupledReassemblyProjectorV38P3,
    ScaleDecoupledReassemblyProjectorV38P4,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v39 import (
    ScaleDecoupledReassemblyProjectorV39P4Static,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v40 import (
    ScaleDecoupledReassemblyProjectorV40P4BoundedDynamic,
    ScaleDecoupledReassemblyProjectorV40P4Fixed,
    ScaleDecoupledReassemblyProjectorV40P4Learnable,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v41 import (
    ScaleDecoupledReassemblyProjectorV41P3Shallow,
    ScaleDecoupledReassemblyProjectorV41P3Uniform,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v42 import (
    ScaleDecoupledReassemblyProjectorV42P5Deep,
    ScaleDecoupledReassemblyProjectorV42P5Uniform,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v43 import (
    ScaleDecoupledReassemblyProjectorV43Spatial010,
    ScaleDecoupledReassemblyProjectorV43Spatial020,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v44 import (
    ScaleDecoupledReassemblyProjectorV44C3k2,
    ScaleDecoupledReassemblyProjectorV44RepC3,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v45 import (
    ScaleDecoupledReassemblyProjectorV45SpatialCentered020,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v46 import (
    ScaleDecoupledReassemblyProjectorV46AnnealedCentered020,
)
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
        projector_p5_mode: str = "full",
        projector_source_indexes: dict[str, list[int]] | None = None,
        projector_source_mode: str = "mask",
        projector_c2f_blocks: dict[str, int] | None = None,
        projector_resample_share: str = "none",
        sdsr_rank_channels: int = 64,
        sdsr_detail_channels: int = 32,
        sdsr_use_local_reassembly: bool = True,
        sdsr_use_directional_guide: bool = True,
        sdsr_use_phase_downsample: bool = True,
        sdsr_cross_scale_mode: str = "none",
        sdsr_cross_scale_rank: int = 32,
    ):
        super().__init__()
        self.name = name
        name_parts = name.split("_")
        if len(name_parts) != 2 or name_parts[0] != "dinov3":
            raise ValueError(
                "RF-DETR-DINOv3 only supports encoder names of the form dinov3_<size>."
            )
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
        assert (
            sorted(self.projector_scale) == self.projector_scale
        ), "only support projector scale P3/P4/P5/P6 in ascending order."
        self.projector_type = projector_type
        if projector_type == "multiscale":
            level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
            scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]
            source_positions_by_scale = None
            if projector_source_indexes is not None:
                feature_position = {
                    block_index: position
                    for position, block_index in enumerate(out_feature_indexes)
                }
                source_positions_by_scale = [
                    [feature_position[index] for index in projector_source_indexes[level]]
                    for level in self.projector_scale
                ]
            c2f_blocks_by_scale = None
            if projector_c2f_blocks is not None:
                c2f_blocks_by_scale = [
                    projector_c2f_blocks[level] for level in self.projector_scale
                ]
            self.projector = MultiScaleProjector(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                scale_factors=scale_factors,
                p5_mode=projector_p5_mode,
                layer_norm=layer_norm,
                rms_norm=rms_norm,
                source_indices_by_scale=source_positions_by_scale,
                source_selection_mode=projector_source_mode,
                c2f_blocks_by_scale=c2f_blocks_by_scale,
                resample_share_mode=projector_resample_share,
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
        elif projector_type == "sdsr_v2":
            self.projector = ScaleDecoupledReassemblyProjectorV2(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                detail_channels=sdsr_detail_channels,
                use_local_reassembly=sdsr_use_local_reassembly,
                use_directional_guide=sdsr_use_directional_guide,
            )
        elif projector_type == "sdsr_v3":
            self.projector = ScaleDecoupledReassemblyProjectorV3(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                detail_channels=sdsr_detail_channels,
                use_local_reassembly=sdsr_use_local_reassembly,
                use_directional_guide=sdsr_use_directional_guide,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v4":
            self.projector = ScaleDecoupledReassemblyProjectorV4(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                detail_channels=sdsr_detail_channels,
                use_local_reassembly=sdsr_use_local_reassembly,
                use_directional_guide=sdsr_use_directional_guide,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v5":
            self.projector = ScaleDecoupledReassemblyProjectorV5(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                detail_channels=sdsr_detail_channels,
                use_local_reassembly=sdsr_use_local_reassembly,
                use_directional_guide=sdsr_use_directional_guide,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v6":
            self.projector = ScaleDecoupledReassemblyProjectorV6(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                detail_channels=sdsr_detail_channels,
                use_local_reassembly=sdsr_use_local_reassembly,
                use_directional_guide=sdsr_use_directional_guide,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v7":
            self.projector = ScaleDecoupledReassemblyProjectorV7(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v8":
            self.projector = ScaleDecoupledReassemblyProjectorV8(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v9":
            self.projector = ScaleDecoupledReassemblyProjectorV9(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v10":
            self.projector = ScaleDecoupledReassemblyProjectorV10(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v11":
            self.projector = ScaleDecoupledReassemblyProjectorV11(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v12":
            self.projector = ScaleDecoupledReassemblyProjectorV12(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v13":
            self.projector = ScaleDecoupledReassemblyProjectorV13(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v14":
            self.projector = ScaleDecoupledReassemblyProjectorV14(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v15":
            self.projector = ScaleDecoupledReassemblyProjectorV15(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v16":
            self.projector = ScaleDecoupledReassemblyProjectorV16(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v17":
            self.projector = ScaleDecoupledReassemblyProjectorV17(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v18":
            self.projector = ScaleDecoupledReassemblyProjectorV18(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v19":
            self.projector = ScaleDecoupledReassemblyProjectorV19(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v20":
            self.projector = ScaleDecoupledReassemblyProjectorV20(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v21":
            self.projector = ScaleDecoupledReassemblyProjectorV21(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v22":
            self.projector = ScaleDecoupledReassemblyProjectorV22(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v23":
            self.projector = ScaleDecoupledReassemblyProjectorV23(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v24":
            self.projector = ScaleDecoupledReassemblyProjectorV24(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v25":
            self.projector = ScaleDecoupledReassemblyProjectorV25(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v26":
            self.projector = ScaleDecoupledReassemblyProjectorV26(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v27":
            self.projector = ScaleDecoupledReassemblyProjectorV27(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v28":
            self.projector = ScaleDecoupledReassemblyProjectorV28(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v29":
            self.projector = ScaleDecoupledReassemblyProjectorV29(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v30":
            self.projector = ScaleDecoupledReassemblyProjectorV30(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                detail_channels=sdsr_detail_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v31":
            self.projector = ScaleDecoupledReassemblyProjectorV31(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v32":
            self.projector = ScaleDecoupledReassemblyProjectorV32(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v33":
            self.projector = ScaleDecoupledReassemblyProjectorV33(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v34":
            self.projector = ScaleDecoupledReassemblyProjectorV34(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v35":
            self.projector = ScaleDecoupledReassemblyProjectorV35(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v36":
            self.projector = ScaleDecoupledReassemblyProjectorV36(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type in {"sdsr_v37_r1", "sdsr_v37_r2", "sdsr_v37_r3"}:
            projector_class = {
                "sdsr_v37_r1": ScaleDecoupledReassemblyProjectorV37R1,
                "sdsr_v37_r2": ScaleDecoupledReassemblyProjectorV37R2,
                "sdsr_v37_r3": ScaleDecoupledReassemblyProjectorV37R3,
            }[projector_type]
            self.projector = projector_class(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type in {
            "sdsr_v38_p3",
            "sdsr_v38_p4",
            "sdsr_v38_both",
        }:
            projector_class = {
                "sdsr_v38_p3": ScaleDecoupledReassemblyProjectorV38P3,
                "sdsr_v38_p4": ScaleDecoupledReassemblyProjectorV38P4,
                "sdsr_v38_both": ScaleDecoupledReassemblyProjectorV38Both,
            }[projector_type]
            self.projector = projector_class(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type in {
            "sdsr_v41_p3_uniform",
            "sdsr_v41_p3_shallow",
        }:
            projector_class = {
                "sdsr_v41_p3_uniform": (
                    ScaleDecoupledReassemblyProjectorV41P3Uniform
                ),
                "sdsr_v41_p3_shallow": (
                    ScaleDecoupledReassemblyProjectorV41P3Shallow
                ),
            }[projector_type]
            self.projector = projector_class(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type in {
            "sdsr_v42_p5_uniform",
            "sdsr_v42_p5_deep",
        }:
            projector_class = {
                "sdsr_v42_p5_uniform": (
                    ScaleDecoupledReassemblyProjectorV42P5Uniform
                ),
                "sdsr_v42_p5_deep": ScaleDecoupledReassemblyProjectorV42P5Deep,
            }[projector_type]
            self.projector = projector_class(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type in {
            "sdsr_v43_spatial010",
            "sdsr_v43_spatial020",
        }:
            projector_class = {
                "sdsr_v43_spatial010": (
                    ScaleDecoupledReassemblyProjectorV43Spatial010
                ),
                "sdsr_v43_spatial020": (
                    ScaleDecoupledReassemblyProjectorV43Spatial020
                ),
            }[projector_type]
            self.projector = projector_class(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type in {
            "sdsr_v44_repc3",
            "sdsr_v44_c3k2",
        }:
            projector_class = {
                "sdsr_v44_repc3": ScaleDecoupledReassemblyProjectorV44RepC3,
                "sdsr_v44_c3k2": ScaleDecoupledReassemblyProjectorV44C3k2,
            }[projector_type]
            self.projector = projector_class(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v45_spatial_centered020":
            self.projector = ScaleDecoupledReassemblyProjectorV45SpatialCentered020(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v46_annealed_centered020":
            self.projector = ScaleDecoupledReassemblyProjectorV46AnnealedCentered020(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type == "sdsr_v39_p4_static":
            self.projector = ScaleDecoupledReassemblyProjectorV39P4Static(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
                use_phase_downsample=sdsr_use_phase_downsample,
            )
        elif projector_type in {
            "sdsr_v40_p4_fixed",
            "sdsr_v40_p4_learnable",
            "sdsr_v40_p4_bounded_dynamic",
        }:
            projector_class = {
                "sdsr_v40_p4_fixed": ScaleDecoupledReassemblyProjectorV40P4Fixed,
                "sdsr_v40_p4_learnable": (
                    ScaleDecoupledReassemblyProjectorV40P4Learnable
                ),
                "sdsr_v40_p4_bounded_dynamic": (
                    ScaleDecoupledReassemblyProjectorV40P4BoundedDynamic
                ),
            }[projector_type]
            self.projector = projector_class(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                levels=self.projector_scale,
                rank_channels=sdsr_rank_channels,
                cross_scale_mode=sdsr_cross_scale_mode,
                cross_scale_rank=sdsr_cross_scale_rank,
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
        if self.projector_type in {
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
            "sdsr_v37_r1",
            "sdsr_v37_r2",
            "sdsr_v37_r3",
            "sdsr_v38_p3",
            "sdsr_v38_p4",
            "sdsr_v38_both",
            "sdsr_v39_p4_static",
            "sdsr_v40_p4_fixed",
            "sdsr_v40_p4_learnable",
            "sdsr_v40_p4_bounded_dynamic",
            "sdsr_v41_p3_uniform",
            "sdsr_v41_p3_shallow",
            "sdsr_v42_p5_uniform",
            "sdsr_v42_p5_deep",
            "sdsr_v43_spatial010",
            "sdsr_v43_spatial020",
            "sdsr_v44_repc3",
            "sdsr_v44_c3k2",
            "sdsr_v45_spatial_centered020",
            "sdsr_v46_annealed_centered020",
        }:
            feats = self.projector(
                feats, image=tensor_list.tensors, mask=tensor_list.mask
            )
        else:
            feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[
                0
            ]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        if self.projector_type in {
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
            "sdsr_v37_r1",
            "sdsr_v37_r2",
            "sdsr_v37_r3",
            "sdsr_v38_p3",
            "sdsr_v38_p4",
            "sdsr_v38_both",
            "sdsr_v39_p4_static",
            "sdsr_v40_p4_fixed",
            "sdsr_v40_p4_learnable",
            "sdsr_v40_p4_bounded_dynamic",
            "sdsr_v41_p3_uniform",
            "sdsr_v41_p3_shallow",
            "sdsr_v42_p5_uniform",
            "sdsr_v42_p5_deep",
            "sdsr_v43_spatial010",
            "sdsr_v43_spatial020",
            "sdsr_v44_repc3",
            "sdsr_v44_c3k2",
            "sdsr_v45_spatial_centered020",
            "sdsr_v46_annealed_centered020",
        }:
            feats = self.projector(feats, image=tensors)
        else:
            feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(
                torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            )
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
                if is_refined_backbone_parameter(
                    n, getattr(args, "backbone_refine_blocks", ())
                ):
                    lr *= float(getattr(args, "backbone_refine_lr_scale", 1.0))
                wd = args.weight_decay * get_vit_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def is_refined_backbone_parameter(name: str, block_indexes) -> bool:
    block_indexes = tuple(block_indexes or ())
    if not block_indexes:
        return False
    if ".encoder.encoder.norm." in name:
        return True
    return any(f".encoder.encoder.blocks.{index}." in name for index in block_indexes)


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
