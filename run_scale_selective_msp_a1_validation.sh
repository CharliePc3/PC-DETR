#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index.}"
SOURCE_MODE="${SOURCE_MODE:?Set SOURCE_MODE to mask or prune.}"
SEED="${SEED:?Set SEED for this validation run.}"
EPOCHS="${EPOCHS:-24}"
MAX_USED_MIB="${MAX_USED_MIB:-8000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

case "${SOURCE_MODE}" in
  mask|prune)
    ;;
  *)
    printf 'Unsupported SOURCE_MODE=%s; expected mask or prune.\n' "${SOURCE_MODE}" >&2
    exit 2
    ;;
esac

P3_INDEXES="2 5 11"
P4_INDEXES="2 5 8 11"
P5_INDEXES="2 8 11"
RUN_NAME="coco_medium_ssmsp_a1_${SOURCE_MODE}_p3-2-5-11_p4-2-5-8-11_p5-2-8-11_g2firstfull_seed${SEED}_e${EPOCHS}"
OUTPUT_DIR="${ROOT}/output/${RUN_NAME}"
SUMMARY_DIR="${ROOT}/output/scale_selective_msp_a1_validation"
TRAIN_LOG="${ROOT}/${RUN_NAME}_nohup.log"
STATUS_LOG="${ROOT}/scale_selective_msp_a1_${SOURCE_MODE}_seed${SEED}_gpu${GPU_ID}_status.log"
PID_FILE="${ROOT}/scale_selective_msp_a1_${SOURCE_MODE}_seed${SEED}_gpu${GPU_ID}.pid"
LOCK_FILE="${ROOT}/scale_selective_msp_a1_${SOURCE_MODE}_seed${SEED}_gpu${GPU_ID}.lock"
RESULT_TSV="${SUMMARY_DIR}/a1_${SOURCE_MODE}_seed${SEED}_result.tsv"

cd "${ROOT}"
mkdir -p \
  "${OUTPUT_DIR}" \
  "${SUMMARY_DIR}" \
  "/tmp/mpl_ssmsp_a1_${SOURCE_MODE}_seed${SEED}_gpu${GPU_ID}" \
  "/tmp/ssmsp_a1_${SOURCE_MODE}_seed${SEED}_gpu${GPU_ID}_cache"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/mpl_ssmsp_a1_${SOURCE_MODE}_seed${SEED}_gpu${GPU_ID}"
export XDG_CACHE_HOME="/tmp/ssmsp_a1_${SOURCE_MODE}_seed${SEED}_gpu${GPU_ID}_cache"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another run already holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${STATUS_LOG}"
}

record_result() {
  if [[ ! -s "${OUTPUT_DIR}/log.txt" ]]; then
    log_status "NO RESULT: ${OUTPUT_DIR}/log.txt is missing"
    return 1
  fi
  result=$(jq -s -r '
    map(select(type == "object" and .test_coco_eval_bbox != null))
    | max_by(.test_coco_eval_bbox[0])
    | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
       .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
       .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
    | @tsv
  ' "${OUTPUT_DIR}/log.txt")
  printf 'mode\tseed\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n%s\t%s\t%s\n' \
    "${SOURCE_MODE}" "${SEED}" "${result}" > "${RESULT_TSV}"
}

on_exit() {
  status=$?
  if [[ ${status} -eq 0 ]]; then
    log_status "FINISHED successfully"
  else
    log_status "FAILED with exit code ${status}"
  fi
  rm -f "${PID_FILE}"
}
trap on_exit EXIT

completed_epochs=0
if [[ -s "${OUTPUT_DIR}/log.txt" ]]; then
  completed_epochs=$(wc -l < "${OUTPUT_DIR}/log.txt")
fi
if (( completed_epochs >= EPOCHS )); then
  log_status "SKIP: ${completed_epochs}/${EPOCHS} epoch records already exist"
  record_result
  exit 0
fi

resume_args=()
if (( completed_epochs > 0 )) && [[ -s "${OUTPUT_DIR}/checkpoint.pth" ]]; then
  resume_args=(--resume "${OUTPUT_DIR}/checkpoint.pth")
  log_status "RESUME from epoch record ${completed_epochs}"
fi

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
  if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
    break
  fi
  log_status "WAIT GPU ${GPU_ID}: ${used:-unknown} MiB used (threshold ${MAX_USED_MIB} MiB)"
  sleep "${POLL_SECONDS}"
done

log_status "START A1 ${SOURCE_MODE} on GPU ${GPU_ID}; seed=${SEED}"
read -r -a p3_args <<< "${P3_INDEXES}"
read -r -a p4_args <<< "${P4_INDEXES}"
read -r -a p5_args <<< "${P5_INDEXES}"

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
  --projector-p3-indexes "${p3_args[@]}" \
  --projector-p4-indexes "${p4_args[@]}" \
  --projector-p5-indexes "${p5_args[@]}" \
  --projector-source-mode "${SOURCE_MODE}" \
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
  "${resume_args[@]}" \
  > "${TRAIN_LOG}" 2>&1

record_result
