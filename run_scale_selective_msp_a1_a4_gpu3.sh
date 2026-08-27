#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:-3}"
SEED="${SEED:-43}"
EPOCHS="${EPOCHS:-24}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_LOG="${ROOT}/scale_selective_msp_a1_a4_gpu3_queue_nohup.log"
PID_FILE="${ROOT}/scale_selective_msp_a1_a4_gpu3_queue.pid"
LOCK_FILE="${ROOT}/scale_selective_msp_a1_a4_gpu3_queue.lock"
SUMMARY_DIR="${ROOT}/output/scale_selective_msp_a1_a4_seed${SEED}"
SUMMARY_TSV="${SUMMARY_DIR}/summary.tsv"

cd "${ROOT}"
mkdir -p \
  "${SUMMARY_DIR}" \
  "/tmp/mpl_scale_selective_msp_gpu${GPU_ID}" \
  "/tmp/scale_selective_msp_gpu${GPU_ID}_cache"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/mpl_scale_selective_msp_gpu${GPU_ID}"
export XDG_CACHE_HOME="/tmp/scale_selective_msp_gpu${GPU_ID}_cache"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another A1-A4 queue already holds %s\n' \
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
    log_status "QUEUE FINISHED successfully"
  else
    log_status "QUEUE FAILED with exit code ${status}"
  fi
  rm -f "${PID_FILE}"
}
trap on_exit EXIT

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
      log_status "GPU ${GPU_ID} ready: ${used} MiB used (threshold ${MAX_USED_MIB} MiB)"
      return
    fi
    log_status "WAIT GPU ${GPU_ID}: ${used:-unknown} MiB used (threshold ${MAX_USED_MIB} MiB)"
    sleep "${POLL_SECONDS}"
  done
}

record_result() {
  variant="$1"
  output_dir="$2"
  if [[ ! -s "${output_dir}/log.txt" ]]; then
    log_status "NO RESULT for ${variant}: ${output_dir}/log.txt is missing"
    return 1
  fi
  result=$(jq -s -r '
    map(select(.test_coco_eval_bbox != null))
    | max_by(.test_coco_eval_bbox[0])
    | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
       .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
       .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
    | @tsv
  ' "${output_dir}/log.txt")
  printf '%s\t%s\n' "${variant}" "${result}" >> "${SUMMARY_TSV}"
}

run_variant() {
  variant="$1"
  p3_indexes="$2"
  p4_indexes="$3"
  p5_indexes="$4"
  output_name="coco_medium_ssmsp_${variant}_mask_p3-${p3_indexes// /-}_p4-${p4_indexes// /-}_p5-${p5_indexes// /-}_g2firstfull_seed${SEED}_e${EPOCHS}"
  output_dir="${ROOT}/output/${output_name}"
  train_log="${ROOT}/${output_name}_nohup.log"

  completed_epochs=0
  if [[ -s "${output_dir}/log.txt" ]]; then
    completed_epochs=$(wc -l < "${output_dir}/log.txt")
  fi
  if (( completed_epochs >= EPOCHS )); then
    log_status "SKIP ${variant}: ${completed_epochs}/${EPOCHS} epoch records already exist"
    record_result "${variant}" "${output_dir}"
    return
  fi

  resume_args=()
  if (( completed_epochs > 0 )) && [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
    log_status "RESUME ${variant} from epoch record ${completed_epochs}"
  fi

  wait_for_gpu
  log_status "START ${variant}: P3=[${p3_indexes}] P4=[${p4_indexes}] P5=[${p5_indexes}] seed=${SEED}"
  read -r -a p3_args <<< "${p3_indexes}"
  read -r -a p4_args <<< "${p4_indexes}"
  read -r -a p5_args <<< "${p5_indexes}"
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
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    > "${train_log}" 2>&1
  log_status "DONE ${variant}"
  record_result "${variant}" "${output_dir}"
}

printf 'variant\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' > "${SUMMARY_TSV}"
log_status "QUEUE START on physical GPU ${GPU_ID}; source-selection mode=functional-mask"
run_variant a1 "2 5 11" "2 5 8 11" "2 8 11"
run_variant a2 "2 5 8" "2 5 8 11" "2 8 11"
run_variant a3 "2 5 11" "2 5 8 11" "5 8 11"
run_variant a4 "2 8 11" "2 5 8 11" "2 8 11"
