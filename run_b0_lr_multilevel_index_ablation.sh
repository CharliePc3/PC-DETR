#!/usr/bin/env bash
set -euo pipefail

cd /data/cpc/root/project/RF-DETR-DINOv3
export MPLCONFIGDIR=/tmp/matplotlib

PYTHON=/home/cpc/.conda/envs/rfdetr-dinov3/bin/python
GPUS=${GPUS:-1,2}
MASTER_PORT=${MASTER_PORT:-29631}

COMMON_ARGS=(
  run_coco_subset.py
  --subset full
  --epochs 24
  --batch-size 8
  --grad-accum-steps 1
  --num-workers 2
  --resolution 576
  --lr 5e-4
  --lr-encoder 2.5e-5
  --lr-vit-layer-decay 0.8
  --lr-component-decay 0.7
  --lr-drop 20
  --warmup-epochs 0.15
  --multi-scale
  --expanded-scales
  --use-ema
)

run_experiment() {
  local name=$1
  shift
  local output_dir="/data/cpc/root/project/RF-DETR-DINOv3/output/${name}"
  local log_file="/data/cpc/root/project/RF-DETR-DINOv3/${name}_nohup.log"

  echo "[$(date '+%F %T')] START ${name}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port="${MASTER_PORT}" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    "$@" 2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE ${name}" | tee -a "${log_file}"
}

# E0 is the optimized-training baseline. E1 changes only the decoder feature
# levels; E2/E3 change only the DINOv3 intermediate block indexes.
run_experiment \
  coco_full_dinov3_b0_lr5e4_enc2p5e5_p4_idx25811_ms_exp_ema_step20_e24 \
  --projector-scale P4 \
  --out-feature-indexes 2 5 8 11

run_experiment \
  coco_full_dinov3_b0_lr5e4_enc2p5e5_p3p4p5_idx25811_ms_exp_ema_step20_e24 \
  --projector-scale P3 P4 P5 \
  --out-feature-indexes 2 5 8 11

run_experiment \
  coco_full_dinov3_b0_lr5e4_enc2p5e5_p4_idx14711_ms_exp_ema_step20_e24 \
  --projector-scale P4 \
  --out-feature-indexes 1 4 7 11

run_experiment \
  coco_full_dinov3_b0_lr5e4_enc2p5e5_p4_idx36911_ms_exp_ema_step20_e24 \
  --projector-scale P4 \
  --out-feature-indexes 3 6 9 11
