# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import math
import os
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class BaseConfig(BaseModel):
    """
    Base configuration class that validates input parameters against the defined model schema.
    If any unknown fields are provided, a ValueError is raised listing the unknown and available parameters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", validate_assignment=True
    )

    @model_validator(mode="before")
    @classmethod
    def catch_typo_kwargs(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        allowed_params = set(cls.model_fields.keys())
        provided_params = set(values)
        unknown_params = provided_params - allowed_params
        if unknown_params:
            unknown_params_list = ", ".join(
                f"'{param}'" for param in sorted(unknown_params)
            )
            allowed_params_list = ", ".join(sorted(allowed_params))
            raise ValueError(
                f"Unknown parameter(s): {unknown_params_list}. Available parameter(s): {allowed_params_list}."
            )
        return values

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in type(self).model_fields:
            super().__setattr__(name, value)
            return
        raise ValueError(f"Unknown attribute: '{name}'.")


class ModelConfig(BaseConfig):
    encoder: Literal["dinov3_small", "dinov3_splus", "dinov3_base", "dinov3_large"]
    out_feature_indexes: List[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
    projector_source_indexes: Optional[Dict[str, List[int]]] = None
    projector_source_mode: Literal["mask", "prune"] = "mask"
    projector_c2f_blocks: Optional[Dict[str, int]] = None
    projector_resample_share: Literal["none", "p3", "p5", "p3_p5"] = "none"
    projector_p4_depth_prior: Optional[Tuple[float, ...]] = None
    projector_type: Literal[
        "multiscale",
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
    ] = "multiscale"
    projector_p5_mode: Literal[
        "full",
        "pool",
        "dwconv",
        "group2",
        "group2_mix",
        "group2_mix128",
        "group2_fullmix",
        "group2_first_full",
        "group2_first_full_dw",
        "group2_first2_full",
        "group2_last_full",
        "group4",
        "group8",
        "fusion",
        "fusion_wide",
        "fusion_refine",
        "fusion_residual",
    ] = "full"
    sdsr_rank_channels: int = 64
    sdsr_detail_channels: int = 32
    sdsr_use_local_reassembly: bool = True
    sdsr_use_directional_guide: bool = True
    sdsr_use_phase_downsample: bool = True
    sdsr_cross_scale_mode: Literal["none", "topdown", "bidirectional"] = "none"
    sdsr_cross_scale_rank: int = 32
    detector_init_seed: Optional[int] = None
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    dec_level_n_points: Optional[Tuple[int, ...]] = None
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    bbox_refine_mode: Literal["shared", "layerwise", "residual"] = "shared"
    query_init: Literal["learned", "hybrid_topk"] = "learned"
    query_memory_detach: bool = True
    query_init_gate: float = 0.0
    scale_routing: bool = False
    scale_routing_mode: Literal["legacy", "cell"] = "legacy"
    scale_routing_layers: Optional[List[int]] = None
    p5_attention_bias: float = 0.0
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    pretrain_exclude_keys: Optional[List[str]] = None
    pretrain_keys_modify_to_load: Optional[List[str]] = None
    pretrained_encoder: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    positional_encoding_size: int
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    use_cdn: bool = False
    dn_number: int = 100
    dn_total_query_budget: int = 0
    dn_label_noise_scale: float = 0.5
    dn_box_noise_scale: float = 1.0
    dn_negative: bool = True
    dn_loss_coef: float = 1.0
    dn_neg_loss_coef: float = 1.0
    register_border_tokens: int = 0
    register_fill: Literal["randn", "rand", "zero"] = "randn"
    register_noise_std: float = 1.0
    feature_adapter: Literal["none", "residual_ln_1x1", "ln_1x1"] = "none"
    feature_adapter_init_scale: float = 1.0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    mask_feature_levels: int = 1
    license: str = "Apache-2.0"

    @model_validator(mode="after")
    def validate_model_options(self):
        if self.mask_feature_levels < 1:
            raise ValueError("mask_feature_levels must be positive.")
        if self.mask_feature_levels > len(self.projector_scale):
            raise ValueError(
                "mask_feature_levels cannot exceed the configured projector levels."
            )
        if self.projector_source_indexes is not None:
            if self.projector_type != "multiscale":
                raise ValueError(
                    "projector_source_indexes requires projector_type='multiscale'."
                )
            if set(self.projector_source_indexes) != set(self.projector_scale):
                raise ValueError(
                    "projector_source_indexes must provide exactly one entry for "
                    "every configured projector scale."
                )
            available_indexes = set(self.out_feature_indexes)
            for scale, indexes in self.projector_source_indexes.items():
                if not indexes:
                    raise ValueError(f"{scale} must retain at least one source index.")
                if len(indexes) != len(set(indexes)):
                    raise ValueError(f"{scale} source indexes must not contain duplicates.")
                unknown_indexes = set(indexes) - available_indexes
                if unknown_indexes:
                    raise ValueError(
                        f"{scale} source indexes {sorted(unknown_indexes)} are not present "
                        f"in out_feature_indexes={self.out_feature_indexes}."
                    )
        if self.projector_c2f_blocks is not None:
            if self.projector_type != "multiscale":
                raise ValueError(
                    "projector_c2f_blocks requires projector_type='multiscale'."
                )
            if set(self.projector_c2f_blocks) != set(self.projector_scale):
                raise ValueError(
                    "projector_c2f_blocks must provide exactly one entry for "
                    "every configured projector scale."
                )
            if any(blocks < 0 for blocks in self.projector_c2f_blocks.values()):
                raise ValueError("projector_c2f_blocks values must be non-negative.")
        if self.projector_resample_share != "none":
            if self.projector_type != "multiscale":
                raise ValueError(
                    "projector_resample_share requires projector_type='multiscale'."
                )
            if self.projector_resample_share in {"p3", "p3_p5"} and "P3" not in self.projector_scale:
                raise ValueError("P3 resampling sharing requires P3 in projector_scale.")
            if self.projector_resample_share in {"p5", "p3_p5"} and "P5" not in self.projector_scale:
                raise ValueError("P5 resampling sharing requires P5 in projector_scale.")
        if self.projector_p4_depth_prior is not None:
            if self.projector_type != "multiscale":
                raise ValueError(
                    "projector_p4_depth_prior requires projector_type='multiscale'."
                )
            if "P4" not in self.projector_scale:
                raise ValueError("projector_p4_depth_prior requires P4 in projector_scale.")
            if len(self.projector_p4_depth_prior) != len(self.out_feature_indexes):
                raise ValueError(
                    "projector_p4_depth_prior must provide one weight per out_feature_index."
                )
            if any(
                not math.isfinite(weight) or weight <= 0
                for weight in self.projector_p4_depth_prior
            ):
                raise ValueError(
                    "projector_p4_depth_prior weights must be finite and positive."
                )
        if self.p5_attention_bias != 0.0:
            if not self.projector_scale or self.projector_scale[-1] != "P5":
                raise ValueError(
                    "p5_attention_bias requires P5 as the last projector scale."
                )
        if self.projector_p5_mode != "full":
            if self.projector_type != "multiscale":
                raise ValueError(
                    "A lightweight P5 mode requires projector_type='multiscale'."
                )
            if "P5" not in self.projector_scale:
                raise ValueError(
                    "A lightweight P5 mode requires P5 in projector_scale."
                )
            p5_index = self.projector_scale.index("P5")
            if p5_index == 0 or self.projector_scale[p5_index - 1] != "P4":
                raise ValueError(
                    "A lightweight P5 mode requires P4 immediately before P5."
                )
        if self.lite_refpoint_refine and self.bbox_refine_mode != "shared":
            raise ValueError("bbox_refine_mode requires lite_refpoint_refine=False.")
        if not self.scale_routing:
            if (
                self.scale_routing_mode != "legacy"
                or self.scale_routing_layers is not None
            ):
                raise ValueError(
                    "scale_routing_mode/layers require scale_routing=True."
                )
            return self
        if self.scale_routing_layers is not None:
            if len(set(self.scale_routing_layers)) != len(self.scale_routing_layers):
                raise ValueError("scale_routing_layers must not contain duplicates.")
            if any(
                layer < 0 or layer >= self.dec_layers
                for layer in self.scale_routing_layers
            ):
                raise ValueError(
                    f"scale_routing_layers must be in [0, {self.dec_layers - 1}]."
                )
        return self

    @field_validator("pretrain_weights", "pretrained_encoder", mode="after")
    @classmethod
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Expand user paths (e.g., '~' or paths with separators) but leave simple filenames
        (like 'rf-detr-base.pth') unchanged so they can match hosted model keys.
        """
        if v is None:
            return v
        return os.path.realpath(os.path.expanduser(v))


class RFDETRBaseConfig(ModelConfig):
    """
    The configuration for an RF-DETR Base model.
    """

    encoder: Literal["dinov3_small", "dinov3_splus", "dinov3_base", "dinov3_large"] = (
        "dinov3_small"
    )
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = None
    resolution: int = 560
    positional_encoding_size: int = 35


class RFDETRLargeDeprecatedConfig(RFDETRBaseConfig):
    """
    Deprecated alias retained for RF-DETR API compatibility.
    """

    encoder: Literal["dinov3_base"] = "dinov3_base"
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P5"]
    pretrain_weights: Optional[str] = None


class RFDETRNanoConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Nano model.
    """

    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = None


class RFDETRSmallConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Small model.
    """

    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = None


class RFDETRMediumConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Medium model.
    """

    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = None


class RFDETRLargeConfig(ModelConfig):
    encoder: Literal["dinov3_splus", "dinov3_base", "dinov3_large"] = "dinov3_splus"
    hidden_dim: int = 256
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_windows: int = 2
    patch_size: int = 16
    projector_scale: List[Literal["P4",]] = ["P4"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_classes: int = 90
    positional_encoding_size: int = 704 // 16
    pretrain_weights: Optional[str] = None
    resolution: int = 704


class RFDETRSegPreviewConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 36
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-preview.pt"
    num_classes: int = 90


class RFDETRSegNanoConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 1
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 312
    positional_encoding_size: int = 312 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: Optional[str] = "rf-detr-seg-nano.pt"
    num_classes: int = 90


class RFDETRSegSmallConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 384
    positional_encoding_size: int = 384 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: Optional[str] = "rf-detr-seg-small.pt"
    num_classes: int = 90


class RFDETRSegMediumConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 2
    dec_layers: int = 5
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 432 // 12
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-medium.pt"
    num_classes: int = 90


class RFDETRSegLargeConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 2
    dec_layers: int = 5
    patch_size: int = 12
    resolution: int = 504
    positional_encoding_size: int = 504 // 12
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-large.pt"
    num_classes: int = 90


class RFDETRSegXLargeConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 2
    dec_layers: int = 6
    patch_size: int = 12
    resolution: int = 624
    positional_encoding_size: int = 624 // 12
    num_queries: int = 300
    num_select: int = 300
    pretrain_weights: Optional[str] = "rf-detr-seg-xlarge.pt"
    num_classes: int = 90


class RFDETRSeg2XLargeConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    num_windows: int = 2
    dec_layers: int = 6
    patch_size: int = 12
    resolution: int = 768
    positional_encoding_size: int = 768 // 12
    num_queries: int = 300
    num_select: int = 300
    pretrain_weights: Optional[str] = "rf-detr-seg-xxlarge.pt"
    num_classes: int = 90


class TrainConfig(BaseModel):
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    batch_size: int = 4
    device: Literal["auto", "cpu", "cuda", "mps"] = DEVICE
    grad_accum_steps: int = 4
    epochs: int = 100
    resume: Optional[str] = None
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = 10
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    backbone_refine_blocks: Tuple[int, ...] = ()
    backbone_refine_lr_scale: float = 1.0
    backbone_refine_anchor_coef: float = 0.0
    backbone_refine_anchor_stop_epoch: int = 20
    online_refine_mode: Literal[
        "none", "object_token", "object_local", "feature_teacher"
    ] = "none"
    online_refine_start_epoch: int = 0
    online_refine_stop_epoch: int = 6
    online_refine_coef: float = 0.0
    online_refine_layers: Tuple[int, ...] = (8, 11)
    online_refine_layer_weights: Tuple[float, ...] = (0.35, 0.65)
    online_refine_background_weight: float = 0.1
    online_refine_margin: float = 0.2
    online_refine_min_box_tokens: int = 1
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    use_cdn: bool = False
    dn_number: int = 100
    dn_total_query_budget: int = 0
    dn_label_noise_scale: float = 0.5
    dn_box_noise_scale: float = 1.0
    dn_negative: bool = True
    dn_loss_coef: float = 1.0
    dn_neg_loss_coef: float = 1.0
    use_budgeted_sa: bool = False
    sa_start_epoch: int = 0
    sa_stop_epoch: int = 0
    sa_total_budgets: Tuple[int, int, int] = (6, 7, 9)
    sa_area_thresholds: Tuple[float, float] = (32**2, 96**2)
    matcher_quality_mode: Literal[
        "none", "aux_group_gate", "cls_ignore", "aux_group_gate_cls_ignore"
    ] = "none"
    matcher_quality_start_epoch: int = 2
    matcher_quality_ramp_epoch: int = 8
    matcher_quality_thresholds: Tuple[float, float, float] = (0.1, 0.2, 0.3)
    matcher_quality_log: bool = False
    use_dense_o2o: bool = False
    dense_o2o_mode: Literal["image", "enhanced"] = "image"
    dense_o2o_start_epoch: int = 2
    dense_o2o_image_stop_epoch: int = 12
    dense_o2o_copyblend_stop_epoch: int = 21
    dense_o2o_mosaic_prob: float = 0.5
    dense_o2o_mixup_prob: float = 0.5
    dense_o2o_copyblend_prob: float = 0.5
    dense_o2o_copyblend_area_threshold: float = 100.0
    dense_o2o_copyblend_num_objects: int = 3
    dense_o2o_copyblend_expand_ratios: Tuple[float, float] = (0.1, 0.25)
    num_select: int = 300
    dataset_file: Literal["coco", "o365", "roboflow", "yolo"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: str
    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    multi_scale_stop_epoch: int = -1
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    feature_adapter: Literal["none", "residual_ln_1x1", "ln_1x1"] = "none"
    feature_adapter_init_scale: float = 1.0
    projector_source_indexes: Optional[Dict[str, List[int]]] = None
    projector_source_mode: Literal["mask", "prune"] = "mask"
    projector_c2f_blocks: Optional[Dict[str, int]] = None
    projector_resample_share: Literal["none", "p3", "p5", "p3_p5"] = "none"
    projector_p4_depth_prior: Optional[Tuple[float, ...]] = None
    projector_distill_teacher: Optional[str] = None
    projector_distill_coef: float = 0.0
    projector_distill_stop_epoch: int = 20
    projector_distill_level_weights: Tuple[float, ...] = (0.5, 1.0, 0.5)
    projector_type: Literal[
        "multiscale",
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
    ] = "multiscale"
    projector_p5_mode: Literal[
        "full",
        "pool",
        "dwconv",
        "group2",
        "group2_mix",
        "group2_mix128",
        "group2_fullmix",
        "group2_first_full",
        "group2_first_full_dw",
        "group2_first2_full",
        "group2_last_full",
        "group4",
        "group8",
        "fusion",
        "fusion_wide",
        "fusion_refine",
        "fusion_residual",
    ] = "full"
    p5_attention_bias: float = 0.0
    sdsr_rank_channels: int = 64
    sdsr_detail_channels: int = 32
    sdsr_use_local_reassembly: bool = True
    sdsr_use_directional_guide: bool = True
    sdsr_use_phase_downsample: bool = True
    sdsr_cross_scale_mode: Literal["none", "topdown", "bidirectional"] = "none"
    sdsr_cross_scale_rank: int = 32
    progress_bar: bool = (
        False  # Enable tqdm progress bars during training and evaluation epochs.
    )
    tensorboard: bool = True
    wandb: bool = False
    mlflow: bool = False
    clearml: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: List[str] = None
    run_test: bool = True
    segmentation_head: bool = False
    segmentation_head_only: bool = False
    eval_max_dets: int = 100
    aug_config: Optional[Dict[str, Any]] = None

    @field_validator("dataset_dir", "output_dir", mode="after")
    @classmethod
    def expand_paths(cls, v: str) -> str:
        """
        Expand user paths (e.g., '~' or paths with separators) but leave simple filenames
        (like 'rf-detr-base.pth') unchanged so they can match hosted model keys.
        """
        if v is None:
            return v
        return os.path.realpath(os.path.expanduser(v))


class SegmentationTrainConfig(TrainConfig):
    num_select: Optional[int] = None
    mask_point_sample_ratio: int = 16
    mask_boundary_sample_ratio: float = 0.0
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 5.0
    segmentation_head: bool = True
