#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-0}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_LOG="${ROOT}/sdsr_v2_medium_ablation_gpu${GPU_ID}_queue_nohup.log"
PID_FILE="${ROOT}/sdsr_v2_medium_ablation_gpu${GPU_ID}_queue.pid"
LOCK_FILE="${ROOT}/sdsr_v2_medium_ablation_gpu${GPU_ID}_queue.lock"
SUMMARY_DIR="${ROOT}/output/sdsr_v2_medium_ablation"
SUMMARY_TSV="${SUMMARY_DIR}/summary.tsv"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}" /tmp/matplotlib_sdsr_v2_medium
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/matplotlib_sdsr_v2_medium"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another SDSR-v2 medium queue already holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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
  train_log="$3"
  runtime_seconds="$4"
  result=$(jq -s -r '
    map(select(.test_coco_eval_bbox != null))
    | max_by(.test_coco_eval_bbox[0])
    | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
       .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
       .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
    | @tsv
  ' "${output_dir}/log.txt")
  parameters=$(rg -o 'Number of trainable parameters: [0-9]+' "${train_log}" | tail -1 | awk '{print $5}')
  max_memory=$(rg -o 'max mem: [0-9]+' "${train_log}" | awk 'BEGIN {max=0} {if ($3>max) max=$3} END {print max}')
  printf '%s\t42\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${runtime_seconds}" "${parameters:-unknown}" "${max_memory:-unknown}" "${result}" >> "${SUMMARY_TSV}"
}

run_variant() {
  variant="$1"
  output_tag="$2"
  shift 2
  extra_args=("$@")
  output_name="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_sdsrv2_${output_tag}_seed42_e24"
  output_dir="${ROOT}/output/${output_name}"
  train_log="${ROOT}/${output_name}_nohup.log"

  if [[ -s "${output_dir}/log.txt" ]] && [[ $(wc -l < "${output_dir}/log.txt") -ge 24 ]]; then
    log_status "SKIP ${variant}: 24 epoch records already exist"
    record_result "${variant}" "${output_dir}" "${train_log}" "existing"
    return
  fi

  resume_args=()
  if [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
    log_status "RESUME ${variant} from ${output_dir}/checkpoint.pth"
  fi

  wait_for_gpu
  log_status "START ${variant}, seed=42, output=${output_name}"
  start_seconds=$(date +%s)
  "${PYTHON}" -u run_coco_subset.py \
    --subset medium \
    --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
    --epochs 24 \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --num-workers 2 \
    --device cuda \
    --world-size 1 \
    --seed 42 \
    --lr 1e-4 \
    --lr-encoder 1.5e-4 \
    --lr-drop 100 \
    --warmup-epochs 0.0 \
    --weight-decay 1e-4 \
    --lr-vit-layer-decay 0.8 \
    --lr-component-decay 0.7 \
    --group-detr 6 \
    --projector-scale P3 P4 P5 \
    --out-feature-indexes 2 5 8 11 \
    --multi-scale \
    --expanded-scales \
    --resolution 640 \
    --dec-layers 4 \
    --dec-n-points 2 \
    --lite-refpoint-refine \
    --query-init learned \
    --num-queries 300 \
    --num-select 300 \
    --use-cdn \
    --dn-number 50 \
    --dn-label-noise-scale 0.5 \
    --dn-box-noise-scale 0.6 \
    --dn-loss-coef 0.5 \
    --eval-max-dets 100 \
    --aug-preset default \
    --output-dir "${output_dir}" \
    "${extra_args[@]}" \
    "${resume_args[@]}" \
    > "${train_log}" 2>&1
  runtime_seconds=$(($(date +%s) - start_seconds))
  log_status "DONE ${variant}, runtime=${runtime_seconds}s"
  record_result "${variant}" "${output_dir}" "${train_log}" "${runtime_seconds}"
}

printf 'variant\tseed\truntime_seconds\ttrainable_parameters\tmax_memory_mib\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' > "${SUMMARY_TSV}"
log_status "QUEUE START on physical GPU ${GPU_ID}, commit=$(git rev-parse --short HEAD), dirty_files=$(git status --porcelain | wc -l)"

# Fresh current-code baseline. This prevents unrelated branch changes from
# contaminating comparison with the earlier B0 run.
run_variant b0_current b0_current \
  --projector-type multiscale

# Test the new layer-preserving, scale-specific fusion without P3 detail reassembly.
run_variant v2_core_rank64 core_rank64 \
  --projector-type sdsr_v2 \
  --sdsr-rank-channels 64 \
  --sdsr-detail-channels 32 \
  --no-sdsr-use-local-reassembly \
  --no-sdsr-use-directional-guide

# Add the v1 component with positive evidence: directional local P3 reassembly.
run_variant v2_directional_rank64 directional_rank64 \
  --projector-type sdsr_v2 \
  --sdsr-rank-channels 64 \
  --sdsr-detail-channels 32 \
  --sdsr-use-local-reassembly \
  --sdsr-use-directional-guide

# Capacity sensitivity check; this remains far below MultiScaleProjector size.
run_variant v2_directional_rank96 directional_rank96 \
  --projector-type sdsr_v2 \
  --sdsr-rank-channels 96 \
  --sdsr-detail-channels 32 \
  --sdsr-use-local-reassembly \
  --sdsr-use-directional-guide
