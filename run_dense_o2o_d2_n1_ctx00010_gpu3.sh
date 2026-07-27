#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-3}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
VARIANT="d2_cb_n1_ctx00010"
OUTPUT_NAME="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_d2_cb1_p05_stop21_ctx00010_seed42_e24"
OUTPUT_DIR="${ROOT}/output/${OUTPUT_NAME}"
TRAIN_LOG="${ROOT}/${OUTPUT_NAME}_nohup.log"
QUEUE_LOG="${ROOT}/dense_o2o_d2_n1_ctx00010_gpu${GPU_ID}_queue_nohup.log"
PID_FILE="${ROOT}/dense_o2o_d2_n1_ctx00010_gpu${GPU_ID}_queue.pid"
LOCK_FILE="${ROOT}/dense_o2o_d2_n1_ctx00010_gpu${GPU_ID}_queue.lock"

cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}" /tmp/matplotlib_dense_o2o_d2_n1_ctx
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/matplotlib_dense_o2o_d2_n1_ctx"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another N1+Context experiment holds %s\n' \
    "$(date '+%F %T')" "${LOCK_FILE}" >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${QUEUE_LOG}"
}

on_exit() {
  status=$?
  if [[ ${status} -eq 0 ]]; then
    log_status "EXPERIMENT FINISHED successfully"
  else
    log_status "EXPERIMENT FAILED with exit code ${status}"
  fi
  rm -f "${PID_FILE}"
}
trap on_exit EXIT

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
  if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
    log_status "GPU ${GPU_ID} ready: ${used} MiB used (threshold ${MAX_USED_MIB} MiB)"
    break
  fi
  log_status "WAIT GPU ${GPU_ID}: ${used:-unknown} MiB used (threshold ${MAX_USED_MIB} MiB)"
  sleep "${POLL_SECONDS}"
done

if [[ -s "${OUTPUT_DIR}/log.txt" ]] && [[ $(wc -l < "${OUTPUT_DIR}/log.txt") -ge 24 ]]; then
  log_status "SKIP ${VARIANT}: 24 epoch records already exist"
  exit 0
fi

resume_args=()
if [[ -s "${OUTPUT_DIR}/checkpoint.pth" ]]; then
  resume_args=(--resume "${OUTPUT_DIR}/checkpoint.pth")
  log_status "RESUME ${VARIANT} from ${OUTPUT_DIR}/checkpoint.pth"
fi

log_status "START ${VARIANT}, seed=42"
start_seconds=$(date +%s)
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
  --out-feature-indexes 2 5 8 11 \
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
  --use-dense-o2o \
  --dense-o2o-mode enhanced \
  --dense-o2o-start-epoch 2 \
  --dense-o2o-image-stop-epoch 12 \
  --dense-o2o-copyblend-stop-epoch 21 \
  --dense-o2o-mosaic-prob 0.5 \
  --dense-o2o-mixup-prob 0.5 \
  --dense-o2o-copyblend-prob 0.5 \
  --dense-o2o-copyblend-area-threshold 100 \
  --dense-o2o-copyblend-num-objects 1 \
  --dense-o2o-copyblend-expand-ratios 0.0 0.1 \
  --output-dir "${OUTPUT_DIR}" \
  "${resume_args[@]}" \
  > "${TRAIN_LOG}" 2>&1
runtime_seconds=$(($(date +%s) - start_seconds))

result=$(jq -s -r '
  map(select(.test_coco_eval_bbox != null))
  | max_by(.test_coco_eval_bbox[0])
  | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
     .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
     .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
  | @tsv
' "${OUTPUT_DIR}/log.txt")
log_status "DONE ${VARIANT}, runtime=${runtime_seconds}s"
log_status "RESULT epoch/AP/AP50/AP75/APs/APm/APl: ${result}"
