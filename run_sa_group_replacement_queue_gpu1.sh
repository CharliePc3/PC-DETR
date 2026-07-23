#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_LOG="${ROOT}/sa_group_replacement_gpu1_queue_nohup.log"
PID_FILE="${ROOT}/sa_group_replacement_gpu1_queue.pid"
LOCK_FILE="${ROOT}/sa_group_replacement_gpu1_queue.lock"
SUMMARY_DIR="${ROOT}/output/sa_group_replacement_ablation"
SUMMARY_TSV="${SUMMARY_DIR}/summary.tsv"
G6_BASELINE_DIR="${ROOT}/output/coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_strict_current_20260719_seed42_e24"
G6_SA_DIR="${ROOT}/output/coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_bsa679_s3_e2-20_e24"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}" /tmp/matplotlib_sa_group_replacement
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/matplotlib_sa_group_replacement"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another SA replacement queue already holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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
  group_count="$2"
  budgets="$3"
  output_dir="$4"
  runtime_seconds="$5"
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
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${group_count}" "${budgets}" "${runtime_seconds}" "${result}" >> "${SUMMARY_TSV}"
}

run_variant() {
  variant="$1"
  group_count="$2"
  budgets_tag="$3"
  use_sa="$4"
  shift 4
  budgets=("$@")
  output_name="coco_medium_tb16_bs8_g${group_count}_p3p4p5_ms_exp_cdn_${variant}_seed42_e24"
  output_dir="${ROOT}/output/${output_name}"
  train_log="${ROOT}/${output_name}_nohup.log"

  if [[ -s "${output_dir}/log.txt" ]] && [[ $(wc -l < "${output_dir}/log.txt") -ge 24 ]]; then
    log_status "SKIP ${variant}: 24 epoch records already exist"
    record_result "${variant}" "${group_count}" "${budgets_tag}" "${output_dir}" existing
    return
  fi

  resume_args=()
  if [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
    log_status "RESUME ${variant} from ${output_dir}/checkpoint.pth"
  fi
  sa_args=()
  if [[ "${use_sa}" == "1" ]]; then
    sa_args=(
      --use-budgeted-sa
      --sa-start-epoch 2
      --sa-stop-epoch 20
      --sa-total-budgets "${budgets[@]}"
      --sa-area-thresholds 1024 9216
    )
  fi

  wait_for_gpu
  log_status "START ${variant}, group=${group_count}, budgets=${budgets_tag}, seed=42"
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
    --group-detr "${group_count}" \
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
    "${sa_args[@]}" \
    "${resume_args[@]}" \
    > "${train_log}" 2>&1
  runtime_seconds=$(($(date +%s) - start_seconds))
  log_status "DONE ${variant}, runtime=${runtime_seconds}s"
  record_result "${variant}" "${group_count}" "${budgets_tag}" "${output_dir}" "${runtime_seconds}"
}

printf 'variant\tgroup\tbudgets_sml\truntime_seconds\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' > "${SUMMARY_TSV}"
if [[ -s "${G6_BASELINE_DIR}/log.txt" ]]; then
  record_result g6_strict_existing 6 strict "${G6_BASELINE_DIR}" existing
fi
if [[ -s "${G6_SA_DIR}/log.txt" ]]; then
  record_result g6_sa679_existing 6 6-7-9 "${G6_SA_DIR}" existing
fi

log_status "QUEUE START on physical GPU ${GPU_ID}"
run_variant g1_strict 1 strict 0
run_variant g1_sa179_aux 1 1-7-9 1 1 7 9
run_variant g2_strict 2 strict 0
run_variant g2_sa279_aux 2 2-7-9 1 2 7 9
run_variant g4_strict 4 strict 0
run_variant g4_sa479_aux 4 4-7-9 1 4 7 9
