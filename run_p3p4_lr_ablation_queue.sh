#!/usr/bin/env bash
set -euo pipefail

cd /data/cpc/root/project/RF-DETR-DINOv3
export MPLCONFIGDIR=/tmp/matplotlib

PYTHON=/home/cpc/.conda/envs/rfdetr-dinov3/bin/python
GPU=${GPU:-3}
RUN_A=${RUN_A:-1}
RUN_B=${RUN_B:-1}

COMMON_ARGS=(
  run_coco_subset.py
  --subset full
  --epochs 24
  --batch-size 8
  --grad-accum-steps 2
  --num-workers 2
  --device cuda
  --world-size 1
  --resolution 576
  --lr-vit-layer-decay 0.8
  --lr-component-decay 0.7
  --lr-drop 20
  --multi-scale
  --expanded-scales
  --use-ema
  --projector-scale P3 P4
  --out-feature-indexes 2 5 8 11
)

run_experiment() {
  local name=$1
  shift
  local output_dir="/data/cpc/root/project/RF-DETR-DINOv3/output/${name}"
  local log_file="/data/cpc/root/project/RF-DETR-DINOv3/${name}_nohup.log"

  echo "[$(date '+%F %T')] START ${name}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u \
    "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    "$@" 2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE ${name}" | tee -a "${log_file}"
}

run_benchmark() {
  local name=$1
  local output_dir="/data/cpc/root/project/RF-DETR-DINOv3/output/${name}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/benchmark_detector.py \
    --checkpoint "${output_dir}/checkpoint_best_total.pth" \
    --output "${output_dir}/benchmark_576_b1_rtx5090.json" \
    2>&1 | tee -a "/data/cpc/root/project/RF-DETR-DINOv3/${name}_benchmark.log"
}

DEIM_LR_NAME=coco_full_dinov3_b0_p3p4_lr5e4_enc2p5e5_idx25811_ms_exp_ema_step20_e24
RF_LR_NAME=coco_full_dinov3_b0_p3p4_rflr1e4_enc1p5e4_idx25811_ms_exp_ema_step20_e24

if [[ "${RUN_A}" == "1" ]]; then
  run_experiment "${DEIM_LR_NAME}" \
    --lr 5e-4 \
    --lr-encoder 2.5e-5 \
    --warmup-epochs 0.15
  run_benchmark "${DEIM_LR_NAME}"
fi

if [[ "${RUN_B}" == "1" ]]; then
  run_experiment "${RF_LR_NAME}" \
    --lr 1e-4 \
    --lr-encoder 1.5e-4 \
    --warmup-epochs 0.0
  run_benchmark "${RF_LR_NAME}"
fi
