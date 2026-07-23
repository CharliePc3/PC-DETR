#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-0}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_LOG="${ROOT}/dense_o2o_d2_ablation_gpu${GPU_ID}_queue_nohup.log"
PID_FILE="${ROOT}/dense_o2o_d2_ablation_gpu${GPU_ID}_queue.pid"
LOCK_FILE="${ROOT}/dense_o2o_d2_ablation_gpu${GPU_ID}_queue.lock"
SUMMARY_DIR="${ROOT}/output/dense_o2o_d2_ablation"
SUMMARY_TSV="${SUMMARY_DIR}/summary.tsv"
EXISTING_D2_DIR="${ROOT}/output/coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_denseo2o_enhanced_e2-12_cb21_seed42_e24"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}" /tmp/matplotlib_dense_o2o_d2
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/matplotlib_dense_o2o_d2"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another D2 ablation queue already holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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
  seed="$2"
  output_dir="$3"
  runtime_seconds="$4"
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
  printf '%s\t%s\t%s\t%s\n' "${variant}" "${seed}" "${runtime_seconds}" "${result}" >> "${SUMMARY_TSV}"
}

run_variant() {
  variant="$1"
  seed="$2"
  output_tag="$3"
  shift 3
  extra_args=("$@")
  output_name="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_d2_${output_tag}_seed${seed}_e24"
  output_dir="${ROOT}/output/${output_name}"
  train_log="${ROOT}/${output_name}_nohup.log"

  if [[ -s "${output_dir}/log.txt" ]] && [[ $(wc -l < "${output_dir}/log.txt") -ge 24 ]]; then
    log_status "SKIP ${variant}: 24 epoch records already exist"
    record_result "${variant}" "${seed}" "${output_dir}" "existing"
    return
  fi

  resume_args=()
  if [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
    log_status "RESUME ${variant} from ${output_dir}/checkpoint.pth"
  fi

  wait_for_gpu
  log_status "START ${variant}, seed=${seed}"
  start_seconds=$(date +%s)
  "${PYTHON}" -u run_coco_subset.py \
    --subset medium \
    --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
    --epochs 24 \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --num-workers 2 \
    --device cuda \
    --seed "${seed}" \
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
    --dense-o2o-copyblend-num-objects 3 \
    --dense-o2o-copyblend-expand-ratios 0.1 0.25 \
    --output-dir "${output_dir}" \
    "${extra_args[@]}" \
    "${resume_args[@]}" \
    > "${train_log}" 2>&1
  runtime_seconds=$(($(date +%s) - start_seconds))
  log_status "DONE ${variant}, runtime=${runtime_seconds}s"
  record_result "${variant}" "${seed}" "${output_dir}" "${runtime_seconds}"
}

printf 'variant\tseed\truntime_seconds\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' > "${SUMMARY_TSV}"
if [[ -s "${EXISTING_D2_DIR}/log.txt" ]]; then
  record_result d2_base 42 "${EXISTING_D2_DIR}" existing
fi

log_status "QUEUE START on physical GPU ${GPU_ID}"
run_variant d2_base_seed43 43 base_cb3_p05_stop21_ctx01025
run_variant d2_cb_n1 42 cb1_p05_stop21_ctx01025 \
  --dense-o2o-copyblend-num-objects 1
run_variant d2_cb_p025 42 cb3_p025_stop21_ctx01025 \
  --dense-o2o-copyblend-prob 0.25
run_variant d2_cb_stop12 42 cb3_p05_stop12_ctx01025 \
  --dense-o2o-copyblend-stop-epoch 12
run_variant d2_cb_ctx00010 42 cb3_p05_stop21_ctx00010 \
  --dense-o2o-copyblend-expand-ratios 0.0 0.1
