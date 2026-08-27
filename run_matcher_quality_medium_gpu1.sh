#!/usr/bin/env bash

set -uo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-1}"
INITIAL_DELAY_SECONDS="${INITIAL_DELAY_SECONDS:-21600}"
POLL_SECONDS="${POLL_SECONDS:-600}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
QUEUE_LOG="${ROOT}/matcher_quality_gpu${GPU_ID}_delayed_queue_nohup.log"
PID_FILE="${ROOT}/matcher_quality_gpu${GPU_ID}_queue.pid"
LOCK_FILE="${ROOT}/matcher_quality_gpu${GPU_ID}_queue.lock"
SUMMARY_DIR="${ROOT}/output/matcher_quality_medium"
SUMMARY_TSV="${SUMMARY_DIR}/summary.tsv"
REFERENCE_DIR="${ROOT}/output/coco_medium_seed44_iterref_scaleroute_p3p4p5_group2_first_full_g6_cdn_denseo2o_d2_cb1_lrd20_e24"

cd "${ROOT}" || exit 1
mkdir -p "${SUMMARY_DIR}" /tmp/matplotlib_matcher_quality_gpu1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/matplotlib_matcher_quality_gpu1"
export XDG_CACHE_HOME="/tmp/rfdetr_matcher_quality_gpu1_cache"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another matcher-quality queue holds %s\n' \
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
    log_status "QUEUE FINISHED"
  else
    log_status "QUEUE EXITED with status ${status}"
  fi
  rm -f "${PID_FILE}"
}
trap on_exit EXIT

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      -i "${GPU_ID}" 2>/dev/null | tr -d ' ')
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
      log_status "GPU ${GPU_ID} ready: ${used} MiB used (threshold ${MAX_USED_MIB} MiB)"
      return
    fi
    log_status "WAIT GPU ${GPU_ID}: ${used:-unknown} MiB used (next check in ${POLL_SECONDS}s)"
    sleep "${POLL_SECONDS}"
  done
}

record_result() {
  variant="$1"
  mode="$2"
  output_dir="$3"
  runtime="$4"
  status="$5"
  result=""
  if [[ -s "${output_dir}/log.txt" ]]; then
    result=$(jq -s -r '
      map(select(.test_coco_eval_bbox != null))
      | if length == 0 then "" else
          max_by(.test_coco_eval_bbox[0])
          | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
             .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
             .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
          | @tsv
        end
    ' "${output_dir}/log.txt")
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${mode}" "${runtime}" "${status}" "${output_dir}" "${result}" \
    >> "${SUMMARY_TSV}"
}

run_variant() {
  variant="$1"
  mode="$2"
  output_name="coco_medium_seed44_matcher_${variant}_g6_cdn_denseo2o_cb1_lrd20_e24"
  output_dir="${ROOT}/output/${output_name}"
  train_log="${ROOT}/${output_name}_nohup.log"

  if [[ -s "${output_dir}/log.txt" ]] && [[ $(wc -l < "${output_dir}/log.txt") -ge 24 ]]; then
    log_status "SKIP ${variant}: 24 epoch records already exist"
    record_result "${variant}" "${mode}" "${output_dir}" existing skipped
    return
  fi

  resume_args=()
  if [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
    log_status "RESUME ${variant} from checkpoint.pth"
  fi

  wait_for_gpu
  log_status "START ${variant}: matcher_quality_mode=${mode}, seed=44"
  start_seconds=$(date +%s)
  "${PYTHON}" -u run_coco_subset.py \
    --subset medium \
    --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
    --epochs 24 \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --num-workers 2 \
    --device cuda \
    --seed 44 \
    --group-detr 6 \
    --projector-scale P3 P4 P5 \
    --projector-p5-mode group2_first_full \
    --out-feature-indexes 2 5 8 11 \
    --multi-scale \
    --expanded-scales \
    --resolution 640 \
    --dec-layers 4 \
    --num-queries 300 \
    --num-select 300 \
    --no-lite-refpoint-refine \
    --bbox-refine-mode shared \
    --scale-routing \
    --scale-routing-mode legacy \
    --lr 0.0001 \
    --lr-encoder 0.00015 \
    --lr-drop 20 \
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
    --dense-o2o-copyblend-expand-ratios 0.1 0.25 \
    --matcher-quality-mode "${mode}" \
    --matcher-quality-start-epoch 2 \
    --matcher-quality-ramp-epoch 8 \
    --matcher-quality-thresholds 0.1 0.2 0.3 \
    --matcher-quality-log \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    > "${train_log}" 2>&1
  exit_code=$?
  runtime_seconds=$(($(date +%s) - start_seconds))
  if [[ ${exit_code} -eq 0 ]]; then
    log_status "DONE ${variant}: runtime=${runtime_seconds}s"
    record_result "${variant}" "${mode}" "${output_dir}" "${runtime_seconds}" done
  else
    log_status "FAILED ${variant}: exit=${exit_code}, runtime=${runtime_seconds}s"
    record_result "${variant}" "${mode}" "${output_dir}" "${runtime_seconds}" "failed:${exit_code}"
  fi
}

printf 'variant\tmode\truntime_seconds\tstatus\toutput_dir\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' \
  > "${SUMMARY_TSV}"
if [[ -s "${REFERENCE_DIR}/log.txt" ]]; then
  record_result reference_36p2 none "${REFERENCE_DIR}" existing reference
fi

log_status "QUEUE ARMED on physical GPU ${GPU_ID}; initial delay=${INITIAL_DELAY_SECONDS}s"
sleep "${INITIAL_DELAY_SECONDS}"
log_status "INITIAL DELAY complete; polling GPU ${GPU_ID} every ${POLL_SECONDS}s"

run_variant q0_stats none
run_variant q1_aux_group_gate aux_group_gate
run_variant q2_cls_ignore cls_ignore
run_variant q3_combined aux_group_gate_cls_ignore
