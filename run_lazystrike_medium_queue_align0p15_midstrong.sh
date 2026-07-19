#!/usr/bin/env bash
set -euo pipefail

cd /data/cpc/root/project/RF-DETR-DINOv3
export PYTHONPATH=/data/cpc/root/project/RF-DETR-DINOv3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

PY=/home/cpc/.conda/envs/rfdetr-dinov3/bin/python
DATA_ROOT=/data/cpc/root/dataset/COCO_RFDETR_TEST

"$PY" run_coco_subset.py \
  --subset medium \
  --data-root "$DATA_ROOT" \
  --epochs 24 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --num-workers 2 \
  --device cuda \
  --seed 42 \
  --group-detr 6 \
  --projector-scale P3 P4 P5 \
  --multi-scale \
  --expanded-scales \
  --resolution 640 \
  --dec-layers 4 \
  --num-queries 300 \
  --num-select 300 \
  --use-cdn \
  --dn-number 50 \
  --dn-label-noise-scale 0.5 \
  --dn-box-noise-scale 0.6 \
  --dn-loss-coef 0.5 \
  --pretrained-encoder /data/cpc/root/project/RF-DETR-DINOv3/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_v2_align0p15/checkpoints/model_final.pt \
  --output-dir /data/cpc/root/project/RF-DETR-DINOv3/output/coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_lazystrike_v2_align0p15_e24 \
  > /data/cpc/root/project/RF-DETR-DINOv3/medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_lazystrike_v2_align0p15_e24_nohup.log 2>&1

"$PY" run_coco_subset.py \
  --subset medium \
  --data-root "$DATA_ROOT" \
  --epochs 24 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --num-workers 2 \
  --device cuda \
  --seed 42 \
  --group-detr 6 \
  --projector-scale P3 P4 P5 \
  --multi-scale \
  --expanded-scales \
  --resolution 640 \
  --dec-layers 4 \
  --num-queries 300 \
  --num-select 300 \
  --use-cdn \
  --dn-number 50 \
  --dn-label-noise-scale 0.5 \
  --dn-box-noise-scale 0.6 \
  --dn-loss-coef 0.5 \
  --pretrained-encoder /data/cpc/root/project/RF-DETR-DINOv3/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_v1v2_midstrong/checkpoints/model_final.pt \
  --output-dir /data/cpc/root/project/RF-DETR-DINOv3/output/coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_lazystrike_v1v2_midstrong_e24 \
  > /data/cpc/root/project/RF-DETR-DINOv3/medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_lazystrike_v1v2_midstrong_e24_nohup.log 2>&1
