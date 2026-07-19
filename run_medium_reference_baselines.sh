#!/usr/bin/env bash

set -uo pipefail

GPU_ID="${1:-0}"
DEIM_ROOT="/data/cpc/root/project/DEIMv2"
RFDETR_ROOT="/data/cpc/root/project/RF-DETR/fast"
DEIM_PYTHON="/home/cpc/.conda/envs/deim/bin/python"
RFDETR_PYTHON="/home/cpc/.conda/envs/rfdetr-ori/bin/python"
QUEUE_LOG="/data/cpc/root/project/RF-DETR-DINOv3/medium_reference_baselines_queue.log"
QUEUE_PID_FILE="/data/cpc/root/project/RF-DETR-DINOv3/medium_reference_baselines_queue.pid"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=1
printf '%s\n' "$$" > "${QUEUE_PID_FILE}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${QUEUE_LOG}"
}

if [[ "${WAIT_FOR_GPU_IDLE}" == "1" && -n "${WAIT_FOR_PID}" ]]; then
  log_status "WAIT for existing PID ${WAIT_FOR_PID} before using GPU ${GPU_ID}"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep 60
  done
fi

if [[ "${WAIT_FOR_GPU_IDLE}" == "1" ]]; then
log_status "WAIT for GPU ${GPU_ID} memory to fall below 1024 MiB"
while true; do
  used_memory="$(
    nvidia-smi \
      --id="${GPU_ID}" \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
  )"
  if [[ "${used_memory}" =~ ^[0-9]+$ ]] && (( used_memory < 1024 )); then
    break
  fi
  sleep 60
done
log_status "GPU ${GPU_ID} is idle; starting reference queue"
else
  log_status "Immediate launch requested on GPU ${GPU_ID}; skipping idle-memory wait"
fi

log_status "START DEIMv2-L medium 5k/2k, batch 16, 24e on GPU ${GPU_ID}"
if (
  cd "${DEIM_ROOT}"
  MPLCONFIGDIR=/tmp/mpl_deimv2_medium24 \
  XDG_CACHE_HOME=/tmp/deimv2_medium24_cache \
  "${DEIM_PYTHON}" -u run_coco_medium_24e.py
) > "${DEIM_ROOT}/medium_deimv2_l_tb16_e24_nohup.log" 2>&1; then
  log_status "DONE DEIMv2-L medium"
else
  status=$?
  log_status "FAILED DEIMv2-L medium with exit code ${status}; continuing to RF-DETR"
fi

log_status "START RF-DETR Medium official checkpoint evaluation on COCO medium val"
if (
  cd "${RFDETR_ROOT}"
  MPLCONFIGDIR=/tmp/mpl_rfdetr_medium24 \
  XDG_CACHE_HOME=/tmp/rfdetr_medium24_cache \
  "${RFDETR_PYTHON}" -u run_coco_medium_train.py \
    --eval-only \
    --output-dir "${RFDETR_ROOT}/runs/eval_coco_medium_rfdetr_medium_maxdets100"
) > "${RFDETR_ROOT}/eval_coco_medium_rfdetr_medium_maxdets100_nohup.log" 2>&1; then
  log_status "DONE RF-DETR Medium evaluation"
else
  status=$?
  log_status "FAILED RF-DETR Medium evaluation with exit code ${status}"
fi

log_status "QUEUE FINISHED"
