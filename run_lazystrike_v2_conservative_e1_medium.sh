#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-0}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
REFINE_NAME="dinov3_small_coco5k_lazystrike_v2_conservative_e1_repro"
DETECT_NAME="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_lazystrike_v2_conservative_e1_repro_e24"
QUEUE_LOG="${ROOT}/lazystrike_v2_conservative_e1_then_medium_queue.log"
PID_FILE="${ROOT}/lazystrike_v2_conservative_e1_then_medium_queue.pid"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLCONFIGDIR="/tmp/matplotlib_lazystrike_v2_e1"
export PYTHONPATH="${ROOT}"
printf '%s\n' "$$" > "${PID_FILE}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${QUEUE_LOG}"
}

if [[ -n "${WAIT_FOR_PID}" ]]; then
  log_status "WAIT for PID ${WAIT_FOR_PID} before using GPU ${GPU_ID}"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep 60
  done
fi

log_status "START one-epoch v2_conservative reproduction on GPU ${GPU_ID}"
"${PYTHON}" -u tools/train_dinov3_lazystrike_refine.py \
  --data-root /data/cpc/root/dataset/COCO \
  --split train2017 \
  --output-dir "${ROOT}/output/lazystrike_refine/${REFINE_NAME}" \
  --encoder dinov3_small \
  --resolution 640 \
  --num-images 5000 \
  --epochs 1 \
  --batch-size 2 \
  --workers 4 \
  --lr 5e-6 \
  --weight-decay 0.05 \
  --train-last-blocks 2 \
  --max-boxes-per-image 12 \
  --min-box-patches 1 \
  --lazy-topk 1 \
  --lazy-target-weight 0.1 \
  --cover-margin 0.1 \
  --cover-temperature 0.1 \
  --lambda-align 0.25 \
  --lambda-cover 0.15 \
  --lambda-consistency 0.25 \
  --lambda-distill 1.0 \
  --device cuda \
  --seed 42 \
  --shuffle \
  --log-every 50 \
  --save-every-epoch \
  > "${ROOT}/train_dinov3_small_lazystrike_coco5k_v2_conservative_e1_repro_nohup.log" 2>&1
log_status "DONE one-epoch refinement"

log_status "START aligned medium detection with one-epoch checkpoint"
"${PYTHON}" -u run_coco_subset.py \
  --subset medium \
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
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
  --pretrained-encoder "${ROOT}/output/lazystrike_refine/${REFINE_NAME}/checkpoints/model_epoch1.pt" \
  --output-dir "${ROOT}/output/${DETECT_NAME}" \
  > "${ROOT}/medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_lazystrike_v2_conservative_e1_repro_e24_nohup.log" 2>&1
log_status "DONE aligned medium detection"
log_status "QUEUE FINISHED"
