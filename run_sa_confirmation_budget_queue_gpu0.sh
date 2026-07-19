#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-0}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_LOG="${ROOT}/sa_confirmation_budget_gpu0_queue_nohup.log"
PID_FILE="${ROOT}/sa_confirmation_budget_gpu0_queue.pid"
LOCK_FILE="${ROOT}/sa_confirmation_budget_gpu0_queue.lock"
SUMMARY_DIR="${ROOT}/output/sa_confirmation_budget_ablation"
SUMMARY_TSV="${SUMMARY_DIR}/summary.tsv"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}" /tmp/matplotlib_sa_confirmation
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}"
export MPLCONFIGDIR="/tmp/matplotlib_sa_confirmation"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another SA confirmation queue already holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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
  printf '%s\t%s\t%s\n' "${variant}" "${seed}" "${result}" >> "${SUMMARY_TSV}"
}

run_variant() {
  variant="$1"
  seed="$2"
  output_name="$3"
  shift 3
  extra_args=("$@")
  output_dir="${ROOT}/output/${output_name}"
  train_log="${ROOT}/${output_name}_nohup.log"

  if [[ -s "${output_dir}/log.txt" ]] && [[ $(wc -l < "${output_dir}/log.txt") -ge 24 ]]; then
    log_status "SKIP ${variant}: 24 epoch records already exist"
    record_result "${variant}" "${seed}" "${output_dir}"
    return
  fi

  resume_args=()
  if [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
    log_status "RESUME ${variant} from ${output_dir}/checkpoint.pth"
  fi

  wait_for_gpu
  log_status "START ${variant}, seed=${seed}"
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
    --output-dir "${output_dir}" \
    "${extra_args[@]}" \
    "${resume_args[@]}" \
    > "${train_log}" 2>&1
  log_status "DONE ${variant}"
  record_result "${variant}" "${seed}" "${output_dir}"
}

printf 'variant\tseed\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' > "${SUMMARY_TSV}"

log_status "QUEUE START on physical GPU ${GPU_ID}"
run_variant strict_current_seed42 42 \
  coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_strict_current_20260719_seed42_e24
run_variant strict_current_seed43 43 \
  coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_strict_current_20260719_seed43_e24
run_variant bsa679_s3_seed43 43 \
  coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_bsa679_s3_e2-20_seed43_e24 \
  --use-budgeted-sa --sa-start-epoch 2 --sa-stop-epoch 20 --sa-total-budgets 6 7 9
run_variant bsa678_s3_seed42 42 \
  coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_bsa678_s3_e2-20_seed42_e24 \
  --use-budgeted-sa --sa-start-epoch 2 --sa-stop-epoch 20 --sa-total-budgets 6 7 8
run_variant bsa6810_s3_seed42 42 \
  coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_bsa6810_s3_e2-20_seed42_e24 \
  --use-budgeted-sa --sa-start-epoch 2 --sa-stop-epoch 20 --sa-total-budgets 6 8 10
run_variant bsa789_s3_seed42 42 \
  coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_bsa789_s3_e2-20_seed42_e24 \
  --use-budgeted-sa --sa-start-epoch 2 --sa-stop-epoch 20 --sa-total-budgets 7 8 9
