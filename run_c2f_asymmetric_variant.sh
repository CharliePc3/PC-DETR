#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
VARIANT="${VARIANT:?VARIANT is required}"
SOURCE_MODE="${SOURCE_MODE:?SOURCE_MODE is required}"
C2F_BLOCKS="${C2F_BLOCKS:?C2F_BLOCKS is required}"
SEED="${SEED:-43}"
EPOCHS="${EPOCHS:-24}"
MAX_USED_MIB="${MAX_USED_MIB:-8000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
SUMMARY_DIR="${ROOT}/output/c2f_asymmetric_stage2"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a \
    "${ROOT}/c2f_asymmetric_stage2_phase2_status.log"
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
      log_status "GPU ${GPU_ID} ready for ${VARIANT}: ${used} MiB used"
      return
    fi
    log_status "WAIT GPU ${GPU_ID} for ${VARIANT}: ${used:-unknown} MiB used"
    sleep "${POLL_SECONDS}"
  done
}

NAME="coco_medium_ssmsp_a1_${SOURCE_MODE}_c2f_${C2F_BLOCKS// /-}_${VARIANT}_seed${SEED}_e${EPOCHS}"
OUTPUT_DIR="${ROOT}/output/${NAME}"
TRAIN_LOG="${ROOT}/${NAME}_nohup.log"
RESULT_FILE="${SUMMARY_DIR}/${VARIANT}_seed${SEED}.tsv"
COMPLETED=$(epoch_records "${OUTPUT_DIR}")

if (( COMPLETED < EPOCHS )); then
  RESUME_ARGS=()
  if (( COMPLETED > 0 )) && [[ -s "${OUTPUT_DIR}/checkpoint.pth" ]]; then
    RESUME_ARGS=(--resume "${OUTPUT_DIR}/checkpoint.pth")
    log_status "RESUME ${VARIANT} seed${SEED} from ${COMPLETED} records"
  fi
  wait_for_gpu
  log_status "START ${VARIANT} GPU${GPU_ID}: mode=${SOURCE_MODE} C2f=[${C2F_BLOCKS}] seed=${SEED}"
  read -r -a BLOCK_ARGS <<< "${C2F_BLOCKS}"
  env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${ROOT}/src" \
    MPLCONFIGDIR="/tmp/mpl_c2f_stage2_gpu${GPU_ID}" \
    XDG_CACHE_HOME="/tmp/c2f_stage2_gpu${GPU_ID}_cache" \
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
      --projector-source-mode "${SOURCE_MODE}" \
      --projector-c2f-blocks "${BLOCK_ARGS[@]}" \
      --projector-type multiscale \
      --projector-p5-mode group2_first_full \
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
  log_status "SKIP ${VARIANT} seed${SEED}: ${COMPLETED}/${EPOCHS} records"
fi

PARAMS=$(sed -nE 's/.*Number of trainable parameters: ([0-9]+).*/\1/p' \
  "${TRAIN_LOG}" | head -n 1)
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${VARIANT}" "${SOURCE_MODE}" "${C2F_BLOCKS// /,}" "${SEED}" \
  "${PARAMS:-unknown}" "$(best_row "${OUTPUT_DIR}")" > "${RESULT_FILE}"
log_status "DONE ${VARIANT} seed${SEED}: $(best_row "${OUTPUT_DIR}")"
