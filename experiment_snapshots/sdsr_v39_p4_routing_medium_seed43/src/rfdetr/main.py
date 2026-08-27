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
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
cleaned main file
"""

import argparse
import copy
import datetime
import json
import math
import multiprocessing
import os
import random
import shutil
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable, DefaultDict, List

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler

import rfdetr.util.misc as utils
from rfdetr.assets.model_weights import (
    ModelWeights,
    download_pretrain_weights,
    validate_pretrain_weights,
)
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import build_backbone_parameter_anchor, evaluate, train_one_epoch
from rfdetr.models import PostProcess, build_criterion_and_postprocessors, build_model
from rfdetr.util.benchmark import benchmark
from rfdetr.util.drop_scheduler import drop_scheduler
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import get_rank, get_world_size, is_main_process, save_on_master
from rfdetr.util.utils import BestMetricHolder, ModelEma, clean_state_dict
from rfdetr.utilities.decorators import _DeprecatedDict

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")

logger = get_logger()


# THE FOLLOWING MODEL ASSETS ARE COVERED BY THE APACHE 2.0 LICENSE
# Legacy dictionary for backward compatibility - DEPRECATED
# Use ModelWeights enum from rfdetr.assets.model_weights instead
OPEN_SOURCE_MODELS = _DeprecatedDict(
    {asset.filename: asset.url for asset in ModelWeights},
    deprecated_name="OPEN_SOURCE_MODELS",
    replacement="`ModelWeights` enum from `rfdetr.assets.model_weights`",
)


def _run_on_train_end_callbacks(callbacks: DefaultDict[str, List[Callable]]) -> None:
    """Run registered training-end callbacks for cleanup."""
    for callback in callbacks["on_train_end"]:
        callback()


def _log_detailed_metrics(tag: str, stats: dict) -> None:
    """Log per-class and overall evaluation metrics in a table format."""
    if not is_main_process() or not isinstance(stats, dict):
        return

    results_json = stats.get("results_json", {})
    class_map = (
        results_json.get("class_map", []) if isinstance(results_json, dict) else []
    )
    if not class_map:
        return

    overall = None
    per_class_rows = []
    for row in class_map:
        if row.get("class") == "all":
            overall = row
        else:
            per_class_rows.append(row)

    def fmt(value):
        try:
            return f"{float(value):.4f}"
        except Exception:
            return "nan"

    logger.info("-" * 96)
    logger.info("[%s] Evaluation Metrics", tag)
    logger.info("-" * 96)
    logger.info(
        "%-16s%12s%12s%12s%12s%12s%12s",
        "Class",
        "AP50:95",
        "AP50",
        "AP75",
        "Precision",
        "Recall",
        "F1-Score",
    )
    logger.info("-" * 96)

    for row in per_class_rows:
        logger.info(
            "%-16s%12s%12s%12s%12s%12s%12s",
            str(row.get("class", "unknown")),
            fmt(row.get("map@50:95")),
            fmt(row.get("map@50")),
            fmt(row.get("map@75")),
            fmt(row.get("precision")),
            fmt(row.get("recall")),
            fmt(row.get("f1_score")),
        )

    logger.info("-" * 96)
    if overall is not None:
        logger.info(
            "%-16s%12s%12s%12s%12s%12s%12s",
            "all",
            fmt(overall.get("map@50:95")),
            fmt(overall.get("map@50")),
            fmt(overall.get("map@75")),
            fmt(overall.get("precision")),
            fmt(overall.get("recall")),
            fmt(overall.get("f1_score")),
        )
        conf = overall.get("confidence_threshold", None)
        if conf is not None:
            logger.info("Best confidence threshold: %.2f", float(conf))
    logger.info("-" * 96)


class Model:
    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self.args = args
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)
        if args.pretrain_weights is not None:
            logger.info("Loading pretrain weights")

            # Validate MD5 hash before loading (non-strict, just warns)
            validate_pretrain_weights(args.pretrain_weights, strict=False)

            try:
                checkpoint = torch.load(
                    args.pretrain_weights, map_location="cpu", weights_only=False
                )
            except Exception as e:
                logger.error(f"Failed to load pretrain weights: {e}")
                # re-download weights if they are corrupted
                logger.info("Failed to load pretrain weights, re-downloading")
                download_pretrain_weights(args.pretrain_weights, redownload=True)
                checkpoint = torch.load(
                    args.pretrain_weights, map_location="cpu", weights_only=False
                )

            # Extract class_names from checkpoint if available
            if "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
                self.args.class_names = checkpoint["args"].class_names
                self.class_names = checkpoint["args"].class_names

            checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
            if checkpoint_num_classes != args.num_classes + 1:
                logger.warning(
                    f"Reinitializing detection head with {checkpoint_num_classes - 1} classes based on pretrained weights,"
                    f" configured for {args.num_classes}."
                )
                self.reinitialize_detection_head(checkpoint_num_classes)
            # add support to exclude_keys
            # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
            if args.pretrain_exclude_keys is not None:
                assert isinstance(args.pretrain_exclude_keys, list)
                removed_keys = []
                for exclude_key in args.pretrain_exclude_keys:
                    if exclude_key.endswith("*"):
                        prefix = exclude_key[:-1]
                        keys_to_remove = [
                            key for key in checkpoint["model"] if key.startswith(prefix)
                        ]
                    else:
                        keys_to_remove = (
                            [exclude_key] if exclude_key in checkpoint["model"] else []
                        )
                    for key in keys_to_remove:
                        checkpoint["model"].pop(key)
                    removed_keys.extend(keys_to_remove)
                if removed_keys:
                    logger.info(
                        "Excluded %d keys from pretrain checkpoint. First excluded keys: %s",
                        len(removed_keys),
                        removed_keys[:5],
                    )
            if args.pretrain_keys_modify_to_load is not None:
                from rfdetr.util.obj365_to_coco_model import (
                    get_coco_pretrain_from_obj365,
                )

                assert isinstance(args.pretrain_keys_modify_to_load, list)
                for modify_key_to_load in args.pretrain_keys_modify_to_load:
                    try:
                        checkpoint["model"][modify_key_to_load] = (
                            get_coco_pretrain_from_obj365(
                                model_without_ddp.state_dict()[modify_key_to_load],
                                checkpoint["model"][modify_key_to_load],
                            )
                        )
                    except:
                        logger.error(
                            f"Failed to load {modify_key_to_load}, deleting from checkpoint"
                        )
                        checkpoint["model"].pop(modify_key_to_load)

            # we may want to resume training with a smaller number of groups for group detr
            num_desired_queries = args.num_queries * args.group_detr
            query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
            for name, state in checkpoint["model"].items():
                if any(name.endswith(x) for x in query_param_names):
                    checkpoint["model"][name] = state[:num_desired_queries]

            incompatible = self.model.load_state_dict(checkpoint["model"], strict=False)
            logger.info(
                "Loaded pretrain checkpoint with %d missing keys and %d unexpected keys.",
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
            if incompatible.missing_keys:
                logger.info("First missing keys: %s", incompatible.missing_keys[:10])
            if incompatible.unexpected_keys:
                logger.info(
                    "First unexpected keys: %s", incompatible.unexpected_keys[:10]
                )

        if args.backbone_lora:
            logger.info("Applying LORA to backbone")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                use_dora=True,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",  # covers OWL-ViT
                    "qkv",  # covers open_clip ie Siglip2
                    "query",
                    "key",
                    "value",
                    "cls_token",
                    "storage_tokens",
                ],
            )
            self.model.backbone[0].encoder = get_peft_model(
                self.model.backbone[0].encoder, lora_config
            )
        self.model = self.model.to(self.device)
        self.postprocess = PostProcess(num_select=args.num_select)
        self.stop_early = False

    def reinitialize_detection_head(self, num_classes):
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self):
        self.stop_early = True
        logger.info("Early stopping requested, will complete current epoch and stop")

    def train(self, callbacks: DefaultDict[str, List[Callable]], **kwargs):
        currently_supported_callbacks = [
            "on_fit_epoch_end",
            "on_train_batch_start",
            "on_train_end",
        ]
        for key in callbacks.keys():
            if key not in currently_supported_callbacks:
                raise ValueError(
                    f"Callback {key} is not currently supported, please file an issue if you need it!\n"
                    f"Currently supported callbacks: {currently_supported_callbacks}"
                )
        args = populate_args(**kwargs)
        if getattr(args, "class_names") is not None:
            self.args.class_names = args.class_names
            self.args.num_classes = args.num_classes

        utils.init_distributed_mode(args)
        logger.info("git:\n  {}\n".format(utils.get_sha()))
        logger.info(str(args))
        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        criterion, postprocess = build_criterion_and_postprocessors(args)
        model = self.model
        for attr in (
            "use_cdn",
            "dn_number",
            "dn_label_noise_scale",
            "dn_box_noise_scale",
            "dn_negative",
        ):
            if hasattr(model, attr):
                setattr(model, attr, getattr(args, attr))
        model.to(device)

        if getattr(args, "segmentation_head_only", False):
            if not args.segmentation_head or not hasattr(model, "segmentation_head"):
                raise ValueError(
                    "segmentation_head_only requires a model with segmentation_head enabled."
                )
            model.requires_grad_(False)
            model.segmentation_head.requires_grad_(True)
            logger.info(
                "Mask-head-only stage enabled: froze the detector and left only "
                "segmentation_head trainable."
            )

        backbone_parameter_anchor = None
        if float(getattr(args, "backbone_refine_anchor_coef", 0.0)) > 0:
            backbone_parameter_anchor = build_backbone_parameter_anchor(
                model, args.backbone_refine_blocks
            )
            anchor_parameters = sum(
                parameter.numel()
                for _name, parameter, _reference in backbone_parameter_anchor
            )
            logger.info(
                "Enabled backbone L2-SP anchor: blocks=%s params=%d coef=%g stop_epoch=%d",
                args.backbone_refine_blocks,
                anchor_parameters,
                args.backbone_refine_anchor_coef,
                args.backbone_refine_anchor_stop_epoch,
            )

        backbone_refine_teacher = None
        if getattr(args, "online_refine_mode", "none") == "feature_teacher":
            backbone_refine_teacher = copy.deepcopy(model.backbone[0].encoder)
            backbone_refine_teacher.requires_grad_(False)
            backbone_refine_teacher.eval().to(device)
            logger.info(
                "Enabled short feature teacher: layers=%s coef=%g epochs=[%d,%d) "
                "background_weight=%g",
                args.online_refine_layers,
                args.online_refine_coef,
                args.online_refine_start_epoch,
                args.online_refine_stop_epoch,
                args.online_refine_background_weight,
            )
        elif getattr(args, "online_refine_mode", "none") in (
            "object_token",
            "object_local",
        ):
            logger.info(
                "Enabled object-conditioned token refinement: layers=%s coef=%g "
                "epochs=[%d,%d) margin=%g",
                args.online_refine_layers,
                args.online_refine_coef,
                args.online_refine_start_epoch,
                args.online_refine_stop_epoch,
                args.online_refine_margin,
            )

        projector_distill_teacher = None
        if getattr(args, "projector_distill_teacher", None):
            teacher_checkpoint = torch.load(
                args.projector_distill_teacher,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            teacher_args = copy.deepcopy(args)
            checkpoint_args = teacher_checkpoint.get("args")
            architecture_fields = (
                "encoder",
                "out_feature_indexes",
                "projector_scale",
                "projector_source_indexes",
                "projector_source_mode",
                "projector_c2f_blocks",
                "projector_resample_share",
                "projector_type",
                "projector_p5_mode",
                "resolution",
                "positional_encoding_size",
                "dec_layers",
                "num_queries",
                "group_detr",
                "dec_n_points",
                "lite_refpoint_refine",
                "bbox_refine_mode",
                "scale_routing",
                "scale_routing_mode",
                "scale_routing_layers",
                "p5_attention_bias",
                "register_border_tokens",
            )
            if checkpoint_args is None:
                raise ValueError("Projector distillation checkpoint has no saved args.")
            for field in architecture_fields:
                if hasattr(checkpoint_args, field):
                    setattr(teacher_args, field, getattr(checkpoint_args, field))
            projector_distill_teacher = build_model(teacher_args)
            projector_distill_teacher.load_state_dict(
                teacher_checkpoint["model"], strict=True
            )
            projector_distill_teacher.requires_grad_(False)
            projector_distill_teacher.eval().to(device)
            del teacher_checkpoint
            logger.info(
                "Enabled projector distillation: teacher=%s coef=%.4f stop_epoch=%d weights=%s",
                args.projector_distill_teacher,
                args.projector_distill_coef,
                args.projector_distill_stop_epoch,
                args.projector_distill_level_weights,
            )

        model_without_ddp = model
        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True
            )
            model_without_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Number of trainable parameters: %d (%.2f M)",
            n_parameters,
            n_parameters / 1e6,
        )
        param_dicts = get_param_dict(args, model_without_ddp)

        param_dicts = [p for p in param_dicts if p["params"].requires_grad]

        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
        # Choose the learning rate scheduler based on the new argument

        dataset_train = build_dataset(
            image_set="train", args=args, resolution=args.resolution
        )
        dataset_val = build_dataset(
            image_set="val", args=args, resolution=args.resolution
        )
        run_test = getattr(args, "run_test", True)
        if run_test:
            dataset_test = build_dataset(
                image_set="test" if args.dataset_file == "roboflow" else "val",
                args=args,
                resolution=args.resolution,
            )
        logger.info(
            f"Dataset loaded: {len(dataset_train)} training samples, {len(dataset_val)} validation samples"
        )

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            if args.run_test:
                sampler_test = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            if args.run_test:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        effective_batch_size = args.batch_size * args.grad_accum_steps
        min_batches = kwargs.get("min_batches", 5)

        num_workers = args.num_workers
        # Hotfix for https://github.com/roboflow/rf-detr/issues/428
        # On platforms using 'spawn' (Windows, macOS), multiprocessing requires the entry point
        # to be protected by `if __name__ == '__main__':`. If it's missing, we force
        # num_workers=0 to prevent a RuntimeError that crashes the process.
        if (
            num_workers > 0
            and multiprocessing.get_start_method(allow_none=True) == "spawn"
        ):
            import __main__

            if not hasattr(__main__, "__file__") or not __main__.__name__ == "__main__":
                warnings.warn(
                    "Setting num_workers to 0 because the script is not wrapped in "
                    "`if __name__ == '__main__':`. This is required for multiprocessing with the 'spawn' start method.",
                    RuntimeWarning,
                )
                num_workers = 0

        if len(dataset_train) < effective_batch_size * min_batches:
            logger.info(
                f"Training with uniform sampler because dataset is too small: {len(dataset_train)} < {effective_batch_size * min_batches}"
            )
            sampler = torch.utils.data.RandomSampler(
                dataset_train,
                replacement=True,
                num_samples=effective_batch_size * min_batches,
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_size=effective_batch_size,
                collate_fn=utils.collate_fn,
                num_workers=num_workers,
                sampler=sampler,
            )
        else:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, effective_batch_size, drop_last=True
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn,
                num_workers=num_workers,
            )

        # There is exactly one optimizer and scheduler step per DataLoader
        # iteration. Deriving this from the dataset size drifts when drop_last
        # removes a partial batch (5000 / 16 is 312 steps, not 313).
        num_training_steps_per_epoch_lr = len(data_loader_train)
        total_training_steps_lr = num_training_steps_per_epoch_lr * args.epochs
        warmup_steps_lr = num_training_steps_per_epoch_lr * args.warmup_epochs

        def lr_lambda(current_step: int):
            if current_step < warmup_steps_lr:
                return float(current_step) / float(max(1, warmup_steps_lr))
            if args.lr_scheduler == "cosine":
                progress = float(current_step - warmup_steps_lr) / float(
                    max(1, total_training_steps_lr - warmup_steps_lr)
                )
                return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
            if args.lr_scheduler == "step":
                return (
                    1.0
                    if current_step < args.lr_drop * num_training_steps_per_epoch_lr
                    else 0.1
                )
            raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        logger.info(
            "LR scheduler: type=%s, steps_per_epoch=%d, drop_epoch=%d",
            args.lr_scheduler,
            num_training_steps_per_epoch_lr,
            args.lr_drop,
        )
        if args.lr_scheduler == "step" and args.lr_drop >= args.epochs:
            logger.warning(
                "lr_drop=%d is not reached during %d training epochs; the learning rate will remain constant.",
                args.lr_drop,
                args.epochs,
            )

        data_loader_val = DataLoader(
            dataset_val,
            args.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=num_workers,
        )
        base_ds = get_coco_api_from_dataset(dataset_val)
        if args.run_test:
            data_loader_test = DataLoader(
                dataset_test,
                args.batch_size,
                sampler=sampler_test,
                drop_last=False,
                collate_fn=utils.collate_fn,
                num_workers=num_workers,
            )
            base_ds_test = get_coco_api_from_dataset(dataset_test)
        if args.use_ema:
            self.ema_m = ModelEma(
                model_without_ddp, decay=args.ema_decay, tau=args.ema_tau
            )
        else:
            self.ema_m = None

        output_dir = Path(args.output_dir)

        if is_main_process():
            logger.info("Get benchmark")
            if args.do_benchmark:
                benchmark_model = copy.deepcopy(model_without_ddp)
                bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
                logger.info(json.dumps(bm, indent=2))
                del benchmark_model

        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
            if args.use_ema:
                if "ema_model" in checkpoint:
                    self.ema_m.module.load_state_dict(
                        clean_state_dict(checkpoint["ema_model"])
                    )
                else:
                    del self.ema_m
                    self.ema_m = ModelEma(model, decay=args.ema_decay, tau=args.ema_tau)
            if (
                not args.eval
                and "optimizer" in checkpoint
                and "lr_scheduler" in checkpoint
                and "epoch" in checkpoint
            ):
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1

        if args.start_epoch >= args.epochs:
            logger.info(
                "Checkpoint epoch (%d) has already reached or exceeded the target epochs (%d). "
                "Training is already complete.",
                args.start_epoch - 1,
                args.epochs,
            )
            _run_on_train_end_callbacks(callbacks)
            return

        if args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocess, data_loader_val, base_ds, device, args
            )
            if args.output_dir:
                if not args.segmentation_head:
                    save_on_master(
                        coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
                    )
                else:
                    save_on_master(
                        coco_evaluator.coco_eval["segm"].eval, output_dir / "eval.pth"
                    )
            return

        # for drop
        total_batch_size = effective_batch_size * get_world_size()
        num_training_steps_per_epoch = (
            len(dataset_train) + total_batch_size - 1
        ) // total_batch_size
        schedules = {}
        if args.dropout > 0:
            schedules["do"] = drop_scheduler(
                args.dropout,
                args.epochs,
                num_training_steps_per_epoch,
                args.cutoff_epoch,
                args.drop_mode,
                args.drop_schedule,
            )
            logger.info(
                "Min DO = %.7f, Max DO = %.7f",
                min(schedules["do"]),
                max(schedules["do"]),
            )

        if args.drop_path > 0:
            schedules["dp"] = drop_scheduler(
                args.drop_path,
                args.epochs,
                num_training_steps_per_epoch,
                args.cutoff_epoch,
                args.drop_mode,
                args.drop_schedule,
            )
            logger.info(
                "Min DP = %.7f, Max DP = %.7f",
                min(schedules["dp"]),
                max(schedules["dp"]),
            )

        if (
            args.output_dir
            and is_main_process()
            and getattr(args, "save_dataset_grids", False)
        ):
            from rfdetr.datasets.save_grids import DatasetGridSaver

            DatasetGridSaver(
                data_loader_train, output_dir, max_batches=3, dataset_type="train"
            ).save_grid()
            DatasetGridSaver(
                data_loader_val, output_dir, max_batches=3, dataset_type="val"
            ).save_grid()
        logger.info("Start training")
        start_time = time.time()
        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        best_map_5095 = 0
        best_map_50 = 0
        best_map_ema_5095 = 0
        best_map_ema_50 = 0

        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed:
                sampler_train.set_epoch(epoch)

            model.train()
            if getattr(args, "segmentation_head_only", False):
                # `model.train()` recursively enables stochastic/stateful layers in
                # the frozen detector. Keep that feature extractor deterministic
                # while allowing the mask head itself to train.
                core_model = model.module if hasattr(model, "module") else model
                core_model.eval()
                core_model.segmentation_head.train()
            criterion.train()
            if hasattr(criterion, "set_epoch"):
                criterion.set_epoch(epoch)
            train_stats = train_one_epoch(
                model,
                criterion,
                lr_scheduler,
                data_loader_train,
                optimizer,
                device,
                epoch,
                effective_batch_size,
                args.clip_max_norm,
                ema_m=self.ema_m,
                schedules=schedules,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers,
                args=args,
                callbacks=callbacks,
                projector_distill_teacher=projector_distill_teacher,
                backbone_parameter_anchor=backbone_parameter_anchor,
                backbone_refine_teacher=backbone_refine_teacher,
            )
            train_epoch_time = time.time() - epoch_start_time
            train_epoch_time_str = str(
                datetime.timedelta(seconds=int(train_epoch_time))
            )
            if args.output_dir:
                checkpoint_paths = [output_dir / "checkpoint.pth"]
                # extra checkpoint before LR drop and every `checkpoint_interval` epochs
                if (epoch + 1) % args.lr_drop == 0 or (
                    epoch + 1
                ) % args.checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    }
                    if args.use_ema:
                        weights.update(
                            {
                                "ema_model": self.ema_m.module.state_dict(),
                            }
                        )
                    if not args.dont_save_weights:
                        # create checkpoint dir
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                        save_on_master(weights, checkpoint_path)

            with torch.no_grad():
                test_stats, coco_evaluator = evaluate(
                    model,
                    criterion,
                    postprocess,
                    data_loader_val,
                    base_ds,
                    device,
                    args=args,
                    header="Test",
                )
            _log_detailed_metrics("val", test_stats)
            if not args.segmentation_head:
                map_regular = test_stats["coco_eval_bbox"][0]
            else:
                map_regular = test_stats["coco_eval_masks"][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                best_map_5095 = max(best_map_5095, map_regular)
                if not args.segmentation_head:
                    map50 = test_stats["coco_eval_bbox"][1]
                else:
                    map50 = test_stats["coco_eval_masks"][1]
                best_map_50 = max(best_map_50, map50)
                checkpoint_path = output_dir / "checkpoint_best_regular.pth"
                if not args.dont_save_weights:
                    save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            if args.use_ema:
                ema_test_stats, _ = evaluate(
                    self.ema_m.module,
                    criterion,
                    postprocess,
                    data_loader_val,
                    base_ds,
                    device,
                    args=args,
                    header="Test-ema",
                )
                _log_detailed_metrics("val_ema", ema_test_stats)
                log_stats.update(
                    {f"ema_test_{k}": v for k, v in ema_test_stats.items()}
                )
                if not args.segmentation_head:
                    map_ema = ema_test_stats["coco_eval_bbox"][0]
                else:
                    map_ema = ema_test_stats["coco_eval_masks"][0]
                best_map_ema_5095 = max(best_map_ema_5095, map_ema)
                _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
                if _isbest:
                    if not args.segmentation_head:
                        map_ema_50 = ema_test_stats["coco_eval_bbox"][1]
                    else:
                        map_ema_50 = ema_test_stats["coco_eval_masks"][1]
                    best_map_ema_50 = max(best_map_ema_50, map_ema_50)
                    checkpoint_path = output_dir / "checkpoint_best_ema.pth"
                    if not args.dont_save_weights:
                        save_on_master(
                            {
                                "model": self.ema_m.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "epoch": epoch,
                                "args": args,
                            },
                            checkpoint_path,
                        )
            log_stats.update(best_map_holder.summary())

            # epoch parameters
            ep_paras = {"epoch": epoch, "n_parameters": n_parameters}
            log_stats.update(ep_paras)
            try:
                log_stats.update({"now_time": str(datetime.datetime.now())})
            except:
                pass
            log_stats["train_epoch_time"] = train_epoch_time_str
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats["epoch_time"] = epoch_time_str
            if args.output_dir and is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            if not args.segmentation_head:
                                torch.save(
                                    coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name,
                                )
                            else:
                                torch.save(
                                    coco_evaluator.coco_eval["segm"].eval,
                                    output_dir / "eval" / name,
                                )

            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)

            if self.stop_early:
                logger.info(f"Early stopping requested, stopping at epoch {epoch}")
                break

        best_is_ema = best_map_ema_5095 > best_map_5095

        if is_main_process():
            if best_is_ema:
                best_checkpoint = output_dir / "checkpoint_best_ema.pth"
            else:
                best_checkpoint = output_dir / "checkpoint_best_regular.pth"

            if best_checkpoint.exists():
                shutil.copy2(best_checkpoint, output_dir / "checkpoint_best_total.pth")
                utils.strip_checkpoint(output_dir / "checkpoint_best_total.pth")

            best_map_5095 = max(best_map_5095, best_map_ema_5095)
            if best_is_ema:
                results = ema_test_stats["results_json"]
            else:
                results = test_stats["results_json"]

            class_map = results["class_map"]
            results["class_map"] = {"valid": class_map}
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

            # Save mask results for valid split if available
            best_stats = ema_test_stats if best_is_ema else test_stats
            if "results_json_masks" in best_stats:
                mask_full = best_stats["results_json_masks"]
                mask_output = {k: v for k, v in mask_full.items() if k != "class_map"}
                mask_output["class_map"] = {"valid": mask_full["class_map"]}
                with open(output_dir / "results_mask.json", "w") as f:
                    json.dump(mask_output, f)
                logger.info(
                    "Mask results saved to %s", output_dir / "results_mask.json"
                )

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            best_summary = best_map_holder.summary()
            if args.use_ema:
                best_ap5095 = best_summary.get("all_best_res", 0.0)
                best_epoch = best_summary.get("all_best_ep", -1)
                best_source = (
                    "ema"
                    if best_summary.get("ema_best_res", float("-inf"))
                    >= best_summary.get("regular_best_res", float("-inf"))
                    else "regular"
                )
            else:
                best_ap5095 = best_summary.get("best_res", 0.0)
                best_epoch = best_summary.get("best_ep", -1)
                best_source = "regular"

            logger.info(
                "Best AP50:95 over training: %.4f (epoch=%s, source=%s)",
                float(best_ap5095),
                best_epoch,
                best_source,
            )
            logger.info("Training time %s", total_time_str)
            logger.info("Results saved to %s", output_dir / "results.json")
        if best_is_ema:
            self.model = self.ema_m.module
        model.eval()

        if args.distributed:
            torch.distributed.barrier()

        if args.run_test:
            best_state_dict = torch.load(
                output_dir / "checkpoint_best_total.pth",
                map_location="cpu",
                weights_only=False,
            )["model"]
            model_without_ddp.load_state_dict(best_state_dict)
            model.eval()

            test_stats, _ = evaluate(
                model,
                criterion,
                postprocess,
                data_loader_test,
                base_ds_test,
                device,
                args=args,
            )
            logger.info(f"Test results: {test_stats}")
            with open(output_dir / "results.json", "r") as f:
                results = json.load(f)
            test_metrics = test_stats["results_json"]["class_map"]
            results["class_map"]["test"] = test_metrics
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

            # Save mask results if they exist (read-modify-write to preserve valid split data)
            if "results_json_masks" in test_stats:
                test_mask_results = test_stats["results_json_masks"]
                test_mask_class_map = test_mask_results["class_map"]
                results_mask_path = output_dir / "results_mask.json"
                if results_mask_path.exists():
                    with open(results_mask_path, "r") as f:
                        results_mask = json.load(f)
                else:
                    # Initialize with top-level scalar metrics (e.g., map, precision, recall, f1_score)
                    # and an empty class_map, mirroring the structure from the validation phase.
                    results_mask = {
                        k: v for k, v in test_mask_results.items() if k != "class_map"
                    }
                    results_mask["class_map"] = {}
                results_mask["class_map"]["test"] = test_mask_class_map
                with open(results_mask_path, "w") as f:
                    json.dump(results_mask, f)
                logger.info("Mask results saved to %s", results_mask_path)

        _run_on_train_end_callbacks(callbacks)

    def export(
        self,
        output_dir="output",
        infer_dir=None,
        simplify=False,
        backbone_only=False,
        opset_version=17,
        verbose=True,
        force=False,
        shape=None,
        batch_size=1,
        **kwargs,
    ):
        """Export the trained model to ONNX format"""
        logger.info("Exporting model to ONNX format")
        try:
            from rfdetr.deploy.export import (
                export_onnx,
                make_infer_image,
                onnx_simplify,
            )
        except ImportError:
            logger.error(
                "It seems some dependencies for ONNX export are missing. Please run `pip install rfdetr[onnxexport]` and try again."
            )
            raise

        device = self.device
        model = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        if shape is None:
            shape = (self.resolution, self.resolution)
        else:
            if shape[0] % 14 != 0 or shape[1] % 14 != 0:
                raise ValueError("Shape must be divisible by 14")

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(
            device
        )
        input_names = ["input"]
        if backbone_only:
            output_names = ["features"]
        elif self.args.segmentation_head:
            output_names = ["dets", "labels", "masks"]
        else:
            output_names = ["dets", "labels"]

        dynamic_axes = None
        model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                logger.debug(f"PyTorch inference output shape: {features.shape}")
            elif self.args.segmentation_head:
                outputs = model(input_tensors)
                dets = outputs["pred_boxes"]
                labels = outputs["pred_logits"]
                masks = outputs["pred_masks"]
                if isinstance(masks, torch.Tensor):
                    logger.debug(
                        f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, "
                        f"Masks: {masks.shape}"
                    )
                else:
                    # masks is a dict with spatial_features, query_features, bias
                    logger.debug(
                        f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}"
                    )
                    logger.debug(
                        "Mask spatial_features: "
                        f"{masks['spatial_features'].shape}, "
                        f"query_features: {masks['query_features'].shape}, "
                        f"bias: {masks['bias'].shape}"
                    )
            else:
                outputs = model(input_tensors)
                dets = outputs["pred_boxes"]
                labels = outputs["pred_logits"]
                logger.debug(
                    f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}"
                )
        model.cpu()
        input_tensors = input_tensors.cpu()

        # Export to ONNX
        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version,
        )

        logger.info(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=force,
            )
            logger.info(f"Successfully simplified ONNX model to: {sim_output_file}")

        logger.info("ONNX export completed successfully")
        self.model = self.model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "LWDETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = vars(args)  # Convert Namespace to dictionary

    if args.subcommand == "distill":
        distill(**config)
    elif args.subcommand is None:
        main(**config)
    elif args.subcommand == "export_model":
        filter_keys = [
            "num_classes",
            "grad_accum_steps",
            "lr",
            "lr_encoder",
            "weight_decay",
            "epochs",
            "lr_drop",
            "clip_max_norm",
            "lr_vit_layer_decay",
            "lr_component_decay",
            "dropout",
            "drop_path",
            "drop_mode",
            "drop_schedule",
            "cutoff_epoch",
            "pretrained_encoder",
            "pretrain_weights",
            "pretrain_exclude_keys",
            "pretrain_keys_modify_to_load",
            "freeze_florence",
            "freeze_aimv2",
            "decoder_norm",
            "set_cost_class",
            "set_cost_bbox",
            "set_cost_giou",
            "cls_loss_coef",
            "bbox_loss_coef",
            "giou_loss_coef",
            "focal_alpha",
            "aux_loss",
            "sum_group_losses",
            "use_varifocal_loss",
            "use_position_supervised_loss",
            "ia_bce_loss",
            "use_budgeted_sa",
            "sa_start_epoch",
            "sa_stop_epoch",
            "sa_total_budgets",
            "sa_area_thresholds",
            "matcher_quality_mode",
            "matcher_quality_start_epoch",
            "matcher_quality_ramp_epoch",
            "matcher_quality_thresholds",
            "matcher_quality_log",
            "use_dense_o2o",
            "dense_o2o_mode",
            "dense_o2o_start_epoch",
            "dense_o2o_image_stop_epoch",
            "dense_o2o_copyblend_stop_epoch",
            "dense_o2o_mosaic_prob",
            "dense_o2o_mixup_prob",
            "dense_o2o_copyblend_prob",
            "dense_o2o_copyblend_area_threshold",
            "dense_o2o_copyblend_num_objects",
            "dense_o2o_copyblend_expand_ratios",
            "dataset_file",
            "coco_path",
            "dataset_dir",
            "square_resize_div_64",
            "output_dir",
            "checkpoint_interval",
            "seed",
            "resume",
            "start_epoch",
            "eval",
            "use_ema",
            "ema_decay",
            "ema_tau",
            "num_workers",
            "device",
            "world_size",
            "dist_url",
            "sync_bn",
            "fp16_eval",
            "infer_dir",
            "verbose",
            "opset_version",
            "dry_run",
            "shape",
        ]
        for key in filter_keys:
            config.pop(key, None)  # Use pop with None to avoid KeyError

        from deploy.export import main as export_main

        if args.batch_size != 1:
            config["batch_size"] = 1
            logger.info(
                f"Only batch_size 1 is supported for onnx export, \
                 but got batchsize = {args.batch_size}. batch_size is forcibly set to 1."
            )
        export_main(**config)


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument(
        "--print_freq",
        default=10,
        type=int,
        help="log frequency (in steps) during train/eval",
    )
    parser.add_argument("--amp", default=False, type=bool)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_encoder", default=1.5e-4, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--lr_drop", default=11, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument("--lr_vit_layer_decay", default=0.8, type=float)
    parser.add_argument("--lr_component_decay", default=1.0, type=float)
    parser.add_argument("--backbone_refine_blocks", type=int, nargs="*", default=[])
    parser.add_argument("--backbone_refine_lr_scale", type=float, default=1.0)
    parser.add_argument("--backbone_refine_anchor_coef", type=float, default=0.0)
    parser.add_argument("--backbone_refine_anchor_stop_epoch", type=int, default=20)
    parser.add_argument(
        "--online_refine_mode",
        choices=("none", "object_token", "object_local", "feature_teacher"),
        default="none",
    )
    parser.add_argument("--online_refine_start_epoch", type=int, default=0)
    parser.add_argument("--online_refine_stop_epoch", type=int, default=6)
    parser.add_argument("--online_refine_coef", type=float, default=0.0)
    parser.add_argument("--online_refine_layers", type=int, nargs="+", default=[8, 11])
    parser.add_argument(
        "--online_refine_layer_weights", type=float, nargs="+", default=[0.35, 0.65]
    )
    parser.add_argument("--online_refine_background_weight", type=float, default=0.1)
    parser.add_argument("--online_refine_margin", type=float, default=0.2)
    parser.add_argument("--online_refine_min_box_tokens", type=int, default=1)
    parser.add_argument(
        "--do_benchmark", action="store_true", help="benchmark the model"
    )

    # drop args
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument(
        "--dropout", type=float, default=0, help="Drop path rate (default: 0.0)"
    )
    parser.add_argument(
        "--drop_path", type=float, default=0, help="Drop path rate (default: 0.0)"
    )

    # early / late dropout and stochastic depth settings
    parser.add_argument(
        "--drop_mode",
        type=str,
        default="standard",
        choices=["standard", "early", "late"],
        help="drop mode",
    )
    parser.add_argument(
        "--drop_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="drop schedule for early dropout / s.d. only",
    )
    parser.add_argument(
        "--cutoff_epoch",
        type=int,
        default=0,
        help="if drop_mode is early / late, this is the epoch where dropout ends / starts",
    )

    # Model parameters
    parser.add_argument(
        "--pretrained_encoder",
        type=str,
        default=None,
        help="Path to the pretrained encoder.",
    )
    parser.add_argument(
        "--pretrain_weights",
        type=str,
        default=None,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--pretrain_exclude_keys",
        type=str,
        default=None,
        nargs="+",
        help="Keys you do not want to load.",
    )
    parser.add_argument(
        "--pretrain_keys_modify_to_load",
        type=str,
        default=None,
        nargs="+",
        help="Keys you want to modify to load. Only used when loading objects365 pre-trained weights.",
    )

    # * Backbone
    parser.add_argument(
        "--encoder",
        default="dinov3_small",
        type=str,
        help="Name of the DINOv3 encoder to use",
    )
    parser.add_argument(
        "--vit_encoder_num_layers",
        default=12,
        type=int,
        help="Number of layers used in ViT encoder",
    )
    parser.add_argument("--window_block_indexes", default=None, type=int, nargs="+")
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--out_feature_indexes",
        default=[-1],
        type=int,
        nargs="+",
        help="only for vit now",
    )
    parser.add_argument("--freeze_encoder", action="store_true", dest="freeze_encoder")
    parser.add_argument("--layer_norm", action="store_true", dest="layer_norm")
    parser.add_argument("--rms_norm", action="store_true", dest="rms_norm")
    parser.add_argument("--backbone_lora", action="store_true", dest="backbone_lora")
    parser.add_argument(
        "--force_no_pretrain", action="store_true", dest="force_no_pretrain"
    )

    # * Transformer
    parser.add_argument(
        "--dec_layers",
        default=3,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--sa_nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's self-attentions",
    )
    parser.add_argument(
        "--ca_nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's cross-attentions",
    )
    parser.add_argument(
        "--num_queries", default=300, type=int, help="Number of query slots"
    )
    parser.add_argument(
        "--group_detr",
        default=13,
        type=int,
        help="Number of groups to speed up detr training",
    )
    parser.add_argument("--two_stage", action="store_true")
    parser.add_argument(
        "--projector_scale",
        default="P4",
        type=str,
        nargs="+",
        choices=("P3", "P4", "P5", "P6"),
    )
    parser.add_argument(
        "--projector_type",
        default="multiscale",
        choices=(
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
        ),
    )
    parser.add_argument(
        "--projector_p5_mode",
        default="full",
        choices=(
            "full",
            "pool",
            "dwconv",
            "group2",
            "group2_mix",
            "group2_mix128",
            "group2_fullmix",
            "group2_first_full",
            "group2_first2_full",
            "group2_last_full",
            "group4",
            "group8",
            "fusion",
            "fusion_wide",
            "fusion_refine",
            "fusion_residual",
        ),
        help="Build P5 independently (full) or derive it cheaply from P4.",
    )
    parser.add_argument("--sdsr_rank_channels", default=64, type=int)
    parser.add_argument("--sdsr_detail_channels", default=32, type=int)
    parser.add_argument(
        "--sdsr_no_local_reassembly",
        dest="sdsr_use_local_reassembly",
        action="store_false",
    )
    parser.add_argument(
        "--sdsr_no_directional_guide",
        dest="sdsr_use_directional_guide",
        action="store_false",
    )
    parser.add_argument(
        "--sdsr_no_phase_downsample",
        dest="sdsr_use_phase_downsample",
        action="store_false",
    )
    parser.add_argument(
        "--sdsr_cross_scale_mode",
        default="none",
        choices=("none", "topdown", "bidirectional"),
    )
    parser.add_argument("--sdsr_cross_scale_rank", default=32, type=int)
    parser.add_argument("--detector_init_seed", default=None, type=int)
    parser.set_defaults(
        sdsr_use_local_reassembly=True,
        sdsr_use_directional_guide=True,
        sdsr_use_phase_downsample=True,
    )
    parser.add_argument(
        "--lite_refpoint_refine",
        action="store_true",
        help="lite refpoint refine mode for speed-up",
    )
    parser.add_argument(
        "--bbox_refine_mode",
        default="shared",
        choices=("shared", "layerwise", "residual"),
    )
    parser.add_argument(
        "--num_select",
        default=100,
        type=int,
        help="the number of predictions selected for evaluation",
    )
    parser.add_argument(
        "--dec_n_points", default=4, type=int, help="the number of sampling points"
    )
    parser.add_argument("--scale_routing", action="store_true")
    parser.add_argument(
        "--scale_routing_mode", default="legacy", choices=("legacy", "cell")
    )
    parser.add_argument("--scale_routing_layers", default=None, type=int, nargs="+")
    parser.add_argument(
        "--p5_attention_bias",
        default=0.0,
        type=float,
        help="Initial deformable-attention logit bias for the last P5 level.",
    )
    parser.add_argument("--decoder_norm", default="LN", type=str)
    parser.add_argument("--bbox_reparam", action="store_true")
    parser.add_argument("--freeze_batch_norm", action="store_true")
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument(
        "--sum_group_losses",
        action="store_true",
        help="To sum losses across groups or mean losses.",
    )
    parser.add_argument("--use_varifocal_loss", action="store_true")
    parser.add_argument("--use_position_supervised_loss", action="store_true")
    parser.add_argument("--ia_bce_loss", action="store_true")
    parser.add_argument("--use_cdn", action="store_true")
    parser.add_argument("--dn_number", default=100, type=int)
    parser.add_argument("--dn_label_noise_scale", default=0.5, type=float)
    parser.add_argument("--dn_box_noise_scale", default=1.0, type=float)
    parser.add_argument("--no_dn_negative", dest="dn_negative", action="store_false")
    parser.set_defaults(dn_negative=True)
    parser.add_argument("--dn_loss_coef", default=1.0, type=float)
    parser.add_argument("--dn_neg_loss_coef", default=1.0, type=float)
    parser.add_argument("--use_budgeted_sa", action="store_true")
    parser.add_argument("--sa_start_epoch", default=0, type=int)
    parser.add_argument("--sa_stop_epoch", default=0, type=int)
    parser.add_argument("--sa_total_budgets", default=(6, 7, 9), type=int, nargs=3)
    parser.add_argument(
        "--sa_area_thresholds", default=(32**2, 96**2), type=float, nargs=2
    )
    parser.add_argument(
        "--matcher_quality_mode",
        default="none",
        choices=("none", "aux_group_gate", "cls_ignore", "aux_group_gate_cls_ignore"),
    )
    parser.add_argument("--matcher_quality_start_epoch", default=2, type=int)
    parser.add_argument("--matcher_quality_ramp_epoch", default=8, type=int)
    parser.add_argument(
        "--matcher_quality_thresholds", default=(0.1, 0.2, 0.3), type=float, nargs=3
    )
    parser.add_argument("--matcher_quality_log", action="store_true")
    parser.add_argument("--use_dense_o2o", action="store_true")
    parser.add_argument(
        "--dense_o2o_mode", default="image", choices=("image", "enhanced")
    )
    parser.add_argument("--dense_o2o_start_epoch", default=2, type=int)
    parser.add_argument("--dense_o2o_image_stop_epoch", default=12, type=int)
    parser.add_argument("--dense_o2o_copyblend_stop_epoch", default=21, type=int)
    parser.add_argument("--dense_o2o_mosaic_prob", default=0.5, type=float)
    parser.add_argument("--dense_o2o_mixup_prob", default=0.5, type=float)
    parser.add_argument("--dense_o2o_copyblend_prob", default=0.5, type=float)
    parser.add_argument(
        "--dense_o2o_copyblend_area_threshold", default=100.0, type=float
    )
    parser.add_argument("--dense_o2o_copyblend_num_objects", default=3, type=int)
    parser.add_argument(
        "--dense_o2o_copyblend_expand_ratios", default=(0.1, 0.25), type=float, nargs=2
    )
    parser.add_argument(
        "--feature_adapter",
        default="none",
        choices=("none", "residual_ln_1x1", "ln_1x1"),
        help="Optional lightweight adapter applied to DINOv3 feature maps before the projector.",
    )
    parser.add_argument("--feature_adapter_init_scale", default=1.0, type=float)
    parser.set_defaults(
        segmentation_head=False,
        mask_downsample_ratio=4,
        mask_feature_levels=1,
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--square_resize_div_64", action="store_true")

    parser.add_argument(
        "--output_dir", default="output", help="path where to save, empty for no saving"
    )
    parser.add_argument("--dont_save_weights", action="store_true")
    parser.add_argument(
        "--checkpoint_interval",
        default=10,
        type=int,
        help="epoch interval to save checkpoint",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", default=0.9997, type=float)
    parser.add_argument("--ema_tau", default=0, type=float)

    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--sync_bn",
        default=True,
        type=bool,
        help="setup synchronized BatchNorm for distributed training",
    )

    # fp16
    parser.add_argument(
        "--fp16_eval",
        default=False,
        action="store_true",
        help="evaluate in fp16 precision.",
    )

    # custom args
    parser.add_argument(
        "--encoder_only", action="store_true", help="Export and benchmark encoder only"
    )
    parser.add_argument(
        "--backbone_only",
        action="store_true",
        help="Export and benchmark backbone only",
    )
    parser.add_argument("--resolution", type=int, default=640, help="input resolution")
    parser.add_argument("--use_cls_token", action="store_true", help="use cls token")
    parser.add_argument("--multi_scale", action="store_true", help="use multi scale")
    parser.add_argument(
        "--expanded_scales", action="store_true", help="use expanded scales"
    )
    parser.add_argument(
        "--do_random_resize_via_padding",
        action="store_true",
        help="use random resize via padding",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=1,
        type=float,
        help="Number of warmup epochs for linear warmup before cosine annealing",
    )
    # Add scheduler type argument: 'step' or 'cosine'
    parser.add_argument(
        "--lr_scheduler",
        default="step",
        choices=["step", "cosine"],
        help="Type of learning rate scheduler to use: 'step' (default) or 'cosine'",
    )
    parser.add_argument(
        "--lr_min_factor",
        default=0.0,
        type=float,
        help="Minimum learning rate factor (as a fraction of initial lr) at the end of cosine annealing",
    )
    # Early stopping parameters
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping based on mAP improvement",
    )
    parser.add_argument(
        "--early_stopping_patience",
        default=10,
        type=int,
        help="Number of epochs with no improvement after which training will be stopped",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        default=0.001,
        type=float,
        help="Minimum change in mAP to qualify as an improvement",
    )
    parser.add_argument(
        "--early_stopping_use_ema",
        action="store_true",
        help="Use EMA model metrics for early stopping",
    )
    # subparsers
    subparsers = parser.add_subparsers(
        title="sub-commands",
        dest="subcommand",
        description="valid subcommands",
        help="additional help",
    )

    # subparser for export model
    parser_export = subparsers.add_parser("export_model", help="LWDETR model export")
    parser_export.add_argument("--infer_dir", type=str, default=None)
    parser_export.add_argument(
        "--verbose", type=ast.literal_eval, default=False, nargs="?", const=True
    )
    parser_export.add_argument("--opset_version", type=int, default=17)
    parser_export.add_argument(
        "--simplify", action="store_true", help="Simplify onnx model"
    )
    parser_export.add_argument(
        "--tensorrt",
        "--trtexec",
        "--trt",
        action="store_true",
        help="build tensorrt engine",
    )
    parser_export.add_argument(
        "--dry-run", "--test", "-t", action="store_true", help="just print command"
    )
    parser_export.add_argument(
        "--profile",
        action="store_true",
        help="Run nsys profiling during TensorRT export",
    )
    parser_export.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(640, 640),
        help="input shape (width, height)",
    )
    return parser


def populate_args(
    # Basic training parameters
    num_classes=2,
    grad_accum_steps=1,
    print_freq=10,
    amp=False,
    lr=1e-4,
    lr_encoder=1.5e-4,
    batch_size=2,
    weight_decay=1e-4,
    epochs=12,
    lr_drop=11,
    clip_max_norm=0.1,
    lr_vit_layer_decay=0.8,
    lr_component_decay=1.0,
    backbone_refine_blocks=(),
    backbone_refine_lr_scale=1.0,
    backbone_refine_anchor_coef=0.0,
    backbone_refine_anchor_stop_epoch=20,
    online_refine_mode="none",
    online_refine_start_epoch=0,
    online_refine_stop_epoch=6,
    online_refine_coef=0.0,
    online_refine_layers=(8, 11),
    online_refine_layer_weights=(0.35, 0.65),
    online_refine_background_weight=0.1,
    online_refine_margin=0.2,
    online_refine_min_box_tokens=1,
    do_benchmark=False,
    # Drop parameters
    dropout=0,
    drop_path=0,
    drop_mode="standard",
    drop_schedule="constant",
    cutoff_epoch=0,
    # Model parameters
    pretrained_encoder=None,
    pretrain_weights=None,
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
    pretrained_distiller=None,
    # Backbone parameters
    encoder="dinov3_small",
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    position_embedding="sine",
    out_feature_indexes=[-1],
    freeze_encoder=False,
    layer_norm=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    # Transformer parameters
    dec_layers=3,
    dim_feedforward=2048,
    hidden_dim=256,
    sa_nheads=8,
    ca_nheads=8,
    num_queries=300,
    group_detr=13,
    two_stage=False,
    projector_scale="P4",
    projector_type="multiscale",
    projector_p5_mode="full",
    sdsr_rank_channels=64,
    sdsr_detail_channels=32,
    sdsr_use_local_reassembly=True,
    sdsr_use_directional_guide=True,
    sdsr_use_phase_downsample=True,
    sdsr_cross_scale_mode="none",
    sdsr_cross_scale_rank=32,
    detector_init_seed=None,
    lite_refpoint_refine=False,
    bbox_refine_mode="shared",
    num_select=100,
    dec_n_points=4,
    decoder_norm="LN",
    bbox_reparam=False,
    freeze_batch_norm=False,
    # Matcher parameters
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    # Loss coefficients
    cls_loss_coef=2,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    focal_alpha=0.25,
    aux_loss=True,
    sum_group_losses=False,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    ia_bce_loss=False,
    use_cdn=False,
    dn_number=100,
    dn_label_noise_scale=0.5,
    dn_box_noise_scale=1.0,
    dn_negative=True,
    dn_loss_coef=1.0,
    dn_neg_loss_coef=1.0,
    use_budgeted_sa=False,
    sa_start_epoch=0,
    sa_stop_epoch=0,
    sa_total_budgets=(6, 7, 9),
    sa_area_thresholds=(32**2, 96**2),
    matcher_quality_mode="none",
    matcher_quality_start_epoch=2,
    matcher_quality_ramp_epoch=8,
    matcher_quality_thresholds=(0.1, 0.2, 0.3),
    matcher_quality_log=False,
    use_dense_o2o=False,
    dense_o2o_mode="image",
    dense_o2o_start_epoch=2,
    dense_o2o_image_stop_epoch=12,
    dense_o2o_copyblend_stop_epoch=21,
    dense_o2o_mosaic_prob=0.5,
    dense_o2o_mixup_prob=0.5,
    dense_o2o_copyblend_prob=0.5,
    dense_o2o_copyblend_area_threshold=100.0,
    dense_o2o_copyblend_num_objects=3,
    dense_o2o_copyblend_expand_ratios=(0.1, 0.25),
    register_border_tokens=0,
    register_fill="randn",
    register_noise_std=1.0,
    feature_adapter="none",
    feature_adapter_init_scale=1.0,
    # Dataset parameters
    dataset_file="coco",
    coco_path=None,
    dataset_dir=None,
    square_resize_div_64=False,
    aug_config=None,
    segmentation_head=False,
    segmentation_head_only=False,
    mask_downsample_ratio=4,
    mask_feature_levels=1,
    # Output parameters
    output_dir="output",
    dont_save_weights=False,
    checkpoint_interval=10,
    seed=42,
    resume="",
    start_epoch=0,
    eval=False,
    use_ema=False,
    ema_decay=0.9997,
    ema_tau=0,
    num_workers=2,
    # Distributed training parameters
    device="cuda",
    world_size=1,
    dist_url="env://",
    sync_bn=True,
    # FP16
    fp16_eval=False,
    # Custom args
    encoder_only=False,
    backbone_only=False,
    resolution=640,
    use_cls_token=False,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
    warmup_epochs=1,
    lr_scheduler="step",
    lr_min_factor=0.0,
    # Early stopping parameters
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=False,
    gradient_checkpointing=False,
    # Additional
    subcommand=None,
    **extra_kwargs,  # To handle any unexpected arguments
):
    args = argparse.Namespace(
        num_classes=num_classes,
        grad_accum_steps=grad_accum_steps,
        print_freq=print_freq,
        amp=amp,
        lr=lr,
        lr_encoder=lr_encoder,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_drop=lr_drop,
        clip_max_norm=clip_max_norm,
        lr_vit_layer_decay=lr_vit_layer_decay,
        lr_component_decay=lr_component_decay,
        backbone_refine_blocks=backbone_refine_blocks,
        backbone_refine_lr_scale=backbone_refine_lr_scale,
        backbone_refine_anchor_coef=backbone_refine_anchor_coef,
        backbone_refine_anchor_stop_epoch=backbone_refine_anchor_stop_epoch,
        online_refine_mode=online_refine_mode,
        online_refine_start_epoch=online_refine_start_epoch,
        online_refine_stop_epoch=online_refine_stop_epoch,
        online_refine_coef=online_refine_coef,
        online_refine_layers=online_refine_layers,
        online_refine_layer_weights=online_refine_layer_weights,
        online_refine_background_weight=online_refine_background_weight,
        online_refine_margin=online_refine_margin,
        online_refine_min_box_tokens=online_refine_min_box_tokens,
        do_benchmark=do_benchmark,
        dropout=dropout,
        drop_path=drop_path,
        drop_mode=drop_mode,
        drop_schedule=drop_schedule,
        cutoff_epoch=cutoff_epoch,
        pretrained_encoder=pretrained_encoder,
        pretrain_weights=pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        pretrain_keys_modify_to_load=pretrain_keys_modify_to_load,
        pretrained_distiller=pretrained_distiller,
        encoder=encoder,
        vit_encoder_num_layers=vit_encoder_num_layers,
        window_block_indexes=window_block_indexes,
        position_embedding=position_embedding,
        out_feature_indexes=out_feature_indexes,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        force_no_pretrain=force_no_pretrain,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        projector_scale=projector_scale,
        projector_type=projector_type,
        projector_p5_mode=projector_p5_mode,
        sdsr_rank_channels=sdsr_rank_channels,
        sdsr_detail_channels=sdsr_detail_channels,
        sdsr_use_local_reassembly=sdsr_use_local_reassembly,
        sdsr_use_directional_guide=sdsr_use_directional_guide,
        sdsr_use_phase_downsample=sdsr_use_phase_downsample,
        sdsr_cross_scale_mode=sdsr_cross_scale_mode,
        sdsr_cross_scale_rank=sdsr_cross_scale_rank,
        detector_init_seed=detector_init_seed,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_refine_mode=bbox_refine_mode,
        num_select=num_select,
        dec_n_points=dec_n_points,
        decoder_norm=decoder_norm,
        bbox_reparam=bbox_reparam,
        freeze_batch_norm=freeze_batch_norm,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        focal_alpha=focal_alpha,
        aux_loss=aux_loss,
        sum_group_losses=sum_group_losses,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
        ia_bce_loss=ia_bce_loss,
        use_cdn=use_cdn,
        dn_number=dn_number,
        dn_label_noise_scale=dn_label_noise_scale,
        dn_box_noise_scale=dn_box_noise_scale,
        dn_negative=dn_negative,
        dn_loss_coef=dn_loss_coef,
        dn_neg_loss_coef=dn_neg_loss_coef,
        use_budgeted_sa=use_budgeted_sa,
        sa_start_epoch=sa_start_epoch,
        sa_stop_epoch=sa_stop_epoch,
        sa_total_budgets=sa_total_budgets,
        sa_area_thresholds=sa_area_thresholds,
        matcher_quality_mode=matcher_quality_mode,
        matcher_quality_start_epoch=matcher_quality_start_epoch,
        matcher_quality_ramp_epoch=matcher_quality_ramp_epoch,
        matcher_quality_thresholds=matcher_quality_thresholds,
        matcher_quality_log=matcher_quality_log,
        use_dense_o2o=use_dense_o2o,
        dense_o2o_mode=dense_o2o_mode,
        dense_o2o_start_epoch=dense_o2o_start_epoch,
        dense_o2o_image_stop_epoch=dense_o2o_image_stop_epoch,
        dense_o2o_copyblend_stop_epoch=dense_o2o_copyblend_stop_epoch,
        dense_o2o_mosaic_prob=dense_o2o_mosaic_prob,
        dense_o2o_mixup_prob=dense_o2o_mixup_prob,
        dense_o2o_copyblend_prob=dense_o2o_copyblend_prob,
        dense_o2o_copyblend_area_threshold=dense_o2o_copyblend_area_threshold,
        dense_o2o_copyblend_num_objects=dense_o2o_copyblend_num_objects,
        dense_o2o_copyblend_expand_ratios=dense_o2o_copyblend_expand_ratios,
        register_border_tokens=register_border_tokens,
        register_fill=register_fill,
        register_noise_std=register_noise_std,
        feature_adapter=feature_adapter,
        feature_adapter_init_scale=feature_adapter_init_scale,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        aug_config=aug_config,
        segmentation_head=segmentation_head,
        segmentation_head_only=segmentation_head_only,
        mask_downsample_ratio=mask_downsample_ratio,
        mask_feature_levels=mask_feature_levels,
        output_dir=output_dir,
        dont_save_weights=dont_save_weights,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        resume=resume,
        start_epoch=start_epoch,
        eval=eval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        num_workers=num_workers,
        device=device,
        world_size=world_size,
        dist_url=dist_url,
        sync_bn=sync_bn,
        fp16_eval=fp16_eval,
        encoder_only=encoder_only,
        backbone_only=backbone_only,
        resolution=resolution,
        use_cls_token=use_cls_token,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        do_random_resize_via_padding=do_random_resize_via_padding,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        gradient_checkpointing=gradient_checkpointing,
        **extra_kwargs,
    )
    return args
