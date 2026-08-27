#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
VARIANT="${VARIANT:?VARIANT is required}"
SHARE_MODE="${SHARE_MODE:?SHARE_MODE is required}"
P5_MODE="${P5_MODE:-group2_first_full}"
DISTILL_TEACHER="${DISTILL_TEACHER:-}"
DISTILL_COEF="${DISTILL_COEF:-0.0}"
DISTILL_STOP_EPOCH="${DISTILL_STOP_EPOCH:-20}"
SEED="${SEED:-43}"
EPOCHS="${EPOCHS:-24}"
MAX_USED_MIB="${MAX_USED_MIB:-8000}"
POLL_SECONDS="${POLL_SECONDS:-600}"
SUMMARY_DIR="${ROOT}/output/resample_sharing_stage3"
STATUS_LOG="${ROOT}/resample_sharing_stage3_status.log"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" >> "${STATUS_LOG}"
}

epoch_records() {
  if [[ -s "$1/log.txt" ]]; then
    wc -l < "$1/log.txt"
  else
    printf '0\n'
  fi
}

best_row() {
  jq -s -r '
    map(select(.test_coco_eval_bbox != null))
    | max_by(.test_coco_eval_bbox[0])
    | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
       .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
       .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
    | @tsv
  ' "$1/log.txt"
}

wait_for_gpu() {
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      -i "${GPU_ID}" | tr -d ' ')
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
      return
    fi
    log_status "WAIT ${VARIANT} seed${SEED} GPU${GPU_ID}: ${used:-unknown} MiB"
    sleep "${POLL_SECONDS}"
  done
}

NAME="coco_medium_stage3_a1_prune_c2f_3-3-3_share_${SHARE_MODE}_${VARIANT}_seed${SEED}_e${EPOCHS}"
OUTPUT_DIR="${ROOT}/output/${NAME}"
TRAIN_LOG="${ROOT}/${NAME}_nohup.log"
RESULT_FILE="${SUMMARY_DIR}/${VARIANT}_seed${SEED}.tsv"
COMPLETED=$(epoch_records "${OUTPUT_DIR}")

if (( COMPLETED < EPOCHS )); then
  RESUME_ARGS=()
  DISTILL_ARGS=()
  if [[ -n "${DISTILL_TEACHER}" ]]; then
    DISTILL_ARGS=(
      --projector-distill-teacher "${DISTILL_TEACHER}"
      --projector-distill-coef "${DISTILL_COEF}"
      --projector-distill-stop-epoch "${DISTILL_STOP_EPOCH}"
      --projector-distill-level-weights 0.5 1.0 0.5
    )
  fi
  if (( COMPLETED > 0 )) && [[ -s "${OUTPUT_DIR}/checkpoint.pth" ]]; then
    RESUME_ARGS=(--resume "${OUTPUT_DIR}/checkpoint.pth")
    log_status "RESUME ${VARIANT} seed${SEED} from ${COMPLETED}/${EPOCHS}"
  fi
  wait_for_gpu
  log_status "START ${VARIANT} seed${SEED} GPU${GPU_ID}: share=${SHARE_MODE} p5=${P5_MODE} distill=${DISTILL_COEF}"
  env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${ROOT}/src" \
    MPLCONFIGDIR="/tmp/mpl_resample_stage3_gpu${GPU_ID}" \
    XDG_CACHE_HOME="/tmp/resample_stage3_gpu${GPU_ID}_cache" \
    ALBUMENTATIONS_DISABLE_VERSION_CHECK=1 \
    "${PYTHON}" -u run_coco_subset.py \
      --subset medium \
      --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
      --epochs "${EPOCHS}" \
      --batch-size 8 \
      --grad-accum-steps 2 \
      --num-workers 2 \
      --resolution 640 \
      --dec-layers 4 \
      --num-queries 300 \
      --num-select 300 \
      --group-detr 6 \
      --dec-n-points 2 \
      --no-lite-refpoint-refine \
      --bbox-refine-mode shared \
      --scale-routing \
      --scale-routing-mode legacy \
      --p5-attention-bias 0.0 \
      --out-feature-indexes 2 5 8 11 \
      --projector-scale P3 P4 P5 \
      --projector-p3-indexes 2 5 11 \
      --projector-p4-indexes 2 5 8 11 \
      --projector-p5-indexes 2 8 11 \
      --projector-source-mode prune \
      --projector-c2f-blocks 3 3 3 \
      --projector-resample-share "${SHARE_MODE}" \
      --projector-type multiscale \
      --projector-p5-mode "${P5_MODE}" \
      "${DISTILL_ARGS[@]}" \
      --multi-scale \
      --expanded-scales \
      --aug-preset default \
      --lr 0.0001 \
      --lr-encoder 0.00015 \
      --lr-drop 20 \
      --weight-decay 0.0001 \
      --lr-vit-layer-decay 0.8 \
      --lr-component-decay 0.7 \
      --use-cdn \
      --dn-number 50 \
      --dn-box-noise-scale 0.6 \
      --dn-label-noise-scale 0.5 \
      --dn-loss-coef 0.5 \
      --backbone-register-border-tokens 1 \
      --seed "${SEED}" \
      --output-dir "${OUTPUT_DIR}" \
      "${RESUME_ARGS[@]}" \
      > "${TRAIN_LOG}" 2>&1
else
  log_status "SKIP ${VARIANT} seed${SEED}: ${COMPLETED}/${EPOCHS} complete"
fi

PARAMS=$(sed -nE 's/.*Number of trainable parameters: ([0-9]+).*/\1/p' \
  "${TRAIN_LOG}" | head -n 1)
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${VARIANT}" "${SHARE_MODE}" "${SEED}" "${PARAMS:-unknown}" \
  "$(best_row "${OUTPUT_DIR}")" > "${RESULT_FILE}"
log_status "DONE ${VARIANT} seed${SEED}: $(best_row "${OUTPUT_DIR}")"
