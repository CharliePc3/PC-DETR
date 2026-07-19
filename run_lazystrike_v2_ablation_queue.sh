#!/usr/bin/env bash
set -euo pipefail
cd /data/cpc/root/project/RF-DETR-DINOv3
while pgrep -f "output/lazystrike_refine/dinov3_small_coco5k_lazystrike_v2_distill1p5" >/dev/null; do
  sleep 60
done
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data/cpc/root/project/RF-DETR-DINOv3 /home/cpc/.conda/envs/rfdetr-dinov3/bin/python tools/train_dinov3_lazystrike_refine.py \
  --data-root /data/cpc/root/dataset/COCO \
  --split train2017 \
  --output-dir output/lazystrike_refine/dinov3_small_coco5k_lazystrike_v2_cover0p10 \
  --encoder dinov3_small \
  --resolution 640 \
  --num-images 5000 \
  --epochs 2 \
  --batch-size 2 \
  --workers 4 \
  --lr 5e-6 \
  --weight-decay 0.05 \
  --train-last-blocks 2 \
  --max-boxes-per-image 12 \
  --min-box-patches 1 \
  --lazy-topk 1 \
  --lazy-target-weight 0.10 \
  --cover-margin 0.10 \
  --cover-temperature 0.10 \
  --lambda-align 0.25 \
  --lambda-cover 0.10 \
  --lambda-consistency 0.25 \
  --lambda-distill 1.0 \
  --amp \
  --device cuda \
  --seed 42 \
  --shuffle \
  --log-every 50 \
  --save-every-epoch
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data/cpc/root/project/RF-DETR-DINOv3 /home/cpc/.conda/envs/rfdetr-dinov3/bin/python tools/train_dinov3_lazystrike_refine.py \
  --data-root /data/cpc/root/dataset/COCO \
  --split train2017 \
  --output-dir output/lazystrike_refine/dinov3_small_coco5k_lazystrike_v2_align0p20 \
  --encoder dinov3_small \
  --resolution 640 \
  --num-images 5000 \
  --epochs 2 \
  --batch-size 2 \
  --workers 4 \
  --lr 5e-6 \
  --weight-decay 0.05 \
  --train-last-blocks 2 \
  --max-boxes-per-image 12 \
  --min-box-patches 1 \
  --lazy-topk 1 \
  --lazy-target-weight 0.10 \
  --cover-margin 0.10 \
  --cover-temperature 0.10 \
  --lambda-align 0.20 \
  --lambda-cover 0.15 \
  --lambda-consistency 0.25 \
  --lambda-distill 1.0 \
  --amp \
  --device cuda \
  --seed 42 \
  --shuffle \
  --log-every 50 \
  --save-every-epoch
