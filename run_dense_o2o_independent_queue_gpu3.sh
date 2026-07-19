#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-3}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_LOG="${ROOT}/dense_o2o_independent_gpu3_queue_nohup.log"
PID_FILE="${ROOT}/dense_o2o_independent_gpu3_queue.pid"
LOCK_FILE="${ROOT}/dense_o2o_independent_gpu3_queue.lock"
SUMMARY_DIR="${ROOT}/output/dense_o2o_independent_ablation"
SUMMARY_TSV="${SUMMARY_DIR}/summary.tsv"
CURRENT_BASELINE_DIR="${ROOT}/output/coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_strict_current_20260719_seed42_e24"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}" /tmp/matplotlib_dense_o2o_stage2
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}"
export MPLCONFIGDIR="/tmp/matplotlib_dense_o2o_stage2"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another Dense O2O queue already holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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
  mode="$2"
  output_name="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_denseo2o_${mode}_e2-12_cb21_seed42_e24"
  output_dir="${ROOT}/output/${output_name}"
  train_log="${ROOT}/${output_name}_nohup.log"

  if [[ -s "${output_dir}/log.txt" ]] && [[ $(wc -l < "${output_dir}/log.txt") -ge 24 ]]; then
    log_status "SKIP ${variant}: 24 epoch records already exist"
    record_result "${variant}" "${output_dir}"
    return
  fi

  resume_args=()
  if [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
    log_status "RESUME ${variant} from ${output_dir}/checkpoint.pth"
  fi

  wait_for_gpu
  log_status "START ${variant}, mode=${mode}, seed=42"
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
    --dense-o2o-mode "${mode}" \
    --dense-o2o-start-epoch 2 \
    --dense-o2o-image-stop-epoch 12 \
    --dense-o2o-copyblend-stop-epoch 21 \
    --dense-o2o-mosaic-prob 0.5 \
    --dense-o2o-mixup-prob 0.5 \
    --dense-o2o-copyblend-prob 0.5 \
    --dense-o2o-copyblend-area-threshold 100 \
    --dense-o2o-copyblend-num-objects 3 \
    --dense-o2o-copyblend-expand-ratios 0.1 0.25 \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    > "${train_log}" 2>&1
  log_status "DONE ${variant}"
  record_result "${variant}" "${output_dir}"
}

printf 'variant\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' > "${SUMMARY_TSV}"
log_status "QUEUE START on physical GPU ${GPU_ID}"
run_variant d1_image image
run_variant d2_enhanced enhanced

if [[ -s "${CURRENT_BASELINE_DIR}/log.txt" ]] && [[ $(wc -l < "${CURRENT_BASELINE_DIR}/log.txt") -ge 24 ]]; then
  record_result d0_strict_current "${CURRENT_BASELINE_DIR}"
else
  log_status "Current strict baseline is still running; it will be available in the SA confirmation summary."
fi
