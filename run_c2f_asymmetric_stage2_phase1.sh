#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
EPOCHS="${EPOCHS:-24}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_USED_MIB="${MAX_USED_MIB:-8000}"
ACCEPT_DROP="${ACCEPT_DROP:-0.0015}"
REJECT_DROP="${REJECT_DROP:-0.0030}"
STATUS_LOG="${ROOT}/c2f_asymmetric_stage2_phase1_status.log"
LOCK_FILE="${ROOT}/c2f_asymmetric_stage2_phase1.lock"
PID_FILE="${ROOT}/c2f_asymmetric_stage2_phase1.pid"
SUMMARY_DIR="${ROOT}/output/c2f_asymmetric_stage2"
SUMMARY_TSV="${SUMMARY_DIR}/phase1_summary.tsv"
DECISION_FILE="${SUMMARY_DIR}/source_mode_decision.txt"

MASK43_DIR="${ROOT}/output/coco_medium_ssmsp_a1_mask_p3-2-5-11_p4-2-5-8-11_p5-2-8-11_g2firstfull_seed43_e24"
MASK42_DIR="${ROOT}/output/coco_medium_ssmsp_a1_mask_p3-2-5-11_p4-2-5-8-11_p5-2-8-11_g2firstfull_seed42_e24"
PRUNE43_DIR="${ROOT}/output/coco_medium_ssmsp_a1_prune_p3-2-5-11_p4-2-5-8-11_p5-2-8-11_g2firstfull_seed43_e24"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another stage2 phase1 controller is already running\n' \
    "$(date '+%F %T')" >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${STATUS_LOG}"
}

on_exit() {
  status=$?
  if [[ ${status} -eq 0 ]]; then
    log_status "CONTROLLER FINISHED successfully"
  else
    log_status "CONTROLLER FAILED with exit code ${status}"
  fi
  rm -f "${PID_FILE}"
}
trap on_exit EXIT

epoch_records() {
  local output_dir="$1"
  if [[ -s "${output_dir}/log.txt" ]]; then
    wc -l < "${output_dir}/log.txt"
  else
    printf '0\n'
  fi
}

wait_for_result() {
  local label="$1"
  local output_dir="$2"
  local records
  while true; do
    records=$(epoch_records "${output_dir}")
    if (( records >= EPOCHS )); then
      log_status "RESULT READY ${label}: ${records}/${EPOCHS} records"
      return
    fi
    log_status "WAIT RESULT ${label}: ${records}/${EPOCHS} records"
    sleep "${POLL_SECONDS}"
  done
}

best_row() {
  local output_dir="$1"
  jq -s -r '
    map(select(.test_coco_eval_bbox != null))
    | max_by(.test_coco_eval_bbox[0])
    | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
       .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
       .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
    | @tsv
  ' "${output_dir}/log.txt"
}

best_ap() {
  best_row "$1" | cut -f2
}

float_le() {
  awk -v lhs="$1" -v rhs="$2" 'BEGIN { exit !(lhs <= rhs) }'
}

wait_for_gpu() {
  local gpu_id="$1"
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      -i "${gpu_id}" | tr -d ' ')
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
      log_status "GPU ${gpu_id} ready: ${used} MiB used"
      return
    fi
    log_status "WAIT GPU ${gpu_id}: ${used:-unknown} MiB used"
    sleep "${POLL_SECONDS}"
  done
}

output_name() {
  local variant="$1"
  local mode="$2"
  local blocks="$3"
  local seed="$4"
  printf 'coco_medium_ssmsp_a1_%s_c2f_%s_%s_seed%s_e%s\n' \
    "${mode}" "${blocks// /-}" "${variant}" "${seed}" "${EPOCHS}"
}

record_result() {
  local variant="$1"
  local mode="$2"
  local blocks="$3"
  local seed="$4"
  local output_dir="$5"
  local train_log="$6"
  local params="unknown"
  local row
  row=$(best_row "${output_dir}")
  if [[ -s "${train_log}" ]]; then
    params=$(sed -nE \
      's/.*Number of trainable parameters: ([0-9]+).*/\1/p' \
      "${train_log}" | head -n 1)
    params="${params:-unknown}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${mode}" "${blocks// /,}" "${seed}" "${params}" "${row}" \
    > "${SUMMARY_DIR}/${variant}_seed${seed}.tsv"
}

run_variant() {
  local variant="$1"
  local gpu_id="$2"
  local mode="$3"
  local blocks="$4"
  local seed="$5"
  local name output_dir train_log completed
  name=$(output_name "${variant}" "${mode}" "${blocks}" "${seed}")
  output_dir="${ROOT}/output/${name}"
  train_log="${ROOT}/${name}_nohup.log"
  completed=$(epoch_records "${output_dir}")

  if (( completed >= EPOCHS )); then
    log_status "SKIP ${variant} seed${seed}: result already complete"
    record_result "${variant}" "${mode}" "${blocks}" "${seed}" \
      "${output_dir}" "${train_log}"
    return
  fi

  local -a resume_args=()
  if (( completed > 0 )) && [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
    log_status "RESUME ${variant} seed${seed} from ${completed} records"
  fi

  wait_for_gpu "${gpu_id}"
  log_status "START ${variant} GPU${gpu_id}: mode=${mode} C2f=[${blocks}] seed=${seed}"
  read -r -a block_args <<< "${blocks}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHONPATH="${ROOT}/src" \
    MPLCONFIGDIR="/tmp/mpl_c2f_stage2_gpu${gpu_id}" \
    XDG_CACHE_HOME="/tmp/c2f_stage2_gpu${gpu_id}_cache" \
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
      --projector-source-mode "${mode}" \
      --projector-c2f-blocks "${block_args[@]}" \
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
      --seed "${seed}" \
      --output-dir "${output_dir}" \
      "${resume_args[@]}" \
      > "${train_log}" 2>&1
  record_result "${variant}" "${mode}" "${blocks}" "${seed}" \
    "${output_dir}" "${train_log}"
  log_status "DONE ${variant} GPU${gpu_id} seed${seed}: $(best_row "${output_dir}")"
}

choose_source_mode() {
  local mask43_ap prune43_ap drop mode
  wait_for_result "A1 mask seed42" "${MASK42_DIR}"
  wait_for_result "A1 prune seed43" "${PRUNE43_DIR}"
  mask43_ap=$(best_ap "${MASK43_DIR}")
  prune43_ap=$(best_ap "${PRUNE43_DIR}")
  drop=$(awk -v base="${mask43_ap}" -v candidate="${prune43_ap}" \
    'BEGIN { printf "%.9f", base - candidate }')
  log_status "A1 seed43 comparison: mask=${mask43_ap}, prune=${prune43_ap}, drop=${drop}"

  if float_le "${drop}" "${ACCEPT_DROP}"; then
    mode="prune"
    log_status "DECISION prune: seed43 drop <= ${ACCEPT_DROP}"
  elif ! float_le "${drop}" "${REJECT_DROP}"; then
    mode="mask"
    log_status "DECISION mask: seed43 drop > ${REJECT_DROP}"
  else
    log_status "AMBIGUOUS: running paired A1 prune seed42 before stage2"
    run_variant "b0_pair" 0 "prune" "3 3 3" 42
    local mask42_ap prune42_dir prune42_ap mean_drop
    mask42_ap=$(best_ap "${MASK42_DIR}")
    prune42_dir="${ROOT}/output/$(output_name b0_pair prune '3 3 3' 42)"
    prune42_ap=$(best_ap "${prune42_dir}")
    mean_drop=$(awk \
      -v m43="${mask43_ap}" -v p43="${prune43_ap}" \
      -v m42="${mask42_ap}" -v p42="${prune42_ap}" \
      'BEGIN { printf "%.9f", ((m43-p43) + (m42-p42)) / 2 }')
    if float_le "${mean_drop}" "0.0020"; then
      mode="prune"
    else
      mode="mask"
    fi
    log_status "PAIRED DECISION ${mode}: mean drop=${mean_drop}"
  fi
  printf '%s\n' "${mode}" > "${DECISION_FILE}"
}

printf 'variant\tmode\tc2f_blocks\tseed\tparams\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' \
  > "${SUMMARY_TSV}"
log_status "CONTROLLER START: polling every ${POLL_SECONDS}s"
choose_source_mode
SOURCE_MODE=$(cat "${DECISION_FILE}")

run_variant "b1_p5" 0 "${SOURCE_MODE}" "3 3 1" 43 &
PID_B1=$!
run_variant "b2_p3" 1 "${SOURCE_MODE}" "1 3 3" 43 &
PID_B2=$!
wait "${PID_B1}"
wait "${PID_B2}"

for result in "${SUMMARY_DIR}"/b[12]_*.tsv; do
  cat "${result}" >> "${SUMMARY_TSV}"
done
log_status "PHASE1 RESULTS written to ${SUMMARY_TSV}"
