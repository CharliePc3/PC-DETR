#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SNAPSHOT_ROOT="${ROOT}/experiment_snapshots/sdsr_v46_denseo2o_medium_seed43"
EXPERIMENT="sdsr_v40_match_stage2_group_seed43"
QUEUE_LOG="${ROOT}/${EXPERIMENT}_gpu${GPU_ID}_queue.log"
PID_FILE="${ROOT}/${EXPERIMENT}_gpu${GPU_ID}_queue.pid"
LOCK_FILE="${ROOT}/${EXPERIMENT}_gpu${GPU_ID}_queue.lock"

if [[ ! -f "${SNAPSHOT_ROOT}/.snapshot_complete" ]]; then
  printf 'Missing frozen snapshot: %s\n' "${SNAPSHOT_ROOT}" >&2
  exit 2
fi
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'Another process holds %s\n' "${LOCK_FILE}" >&2
  exit 3
fi
printf '%s\n' "$$" > "${PID_FILE}"
trap 'status=$?; rm -f "${PID_FILE}"; exit ${status}' EXIT

gpu_used_mib() {
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null \
    | sed -n "$((GPU_ID + 1))p" \
    | awk -F ',' '{gsub(/[^0-9]/, "", $3); print $3}'
}

wait_for_gpu() {
  while true; do
    used_mib="$(gpu_used_mib || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_USED_MIB )); then
      return
    fi
    sleep "${POLL_SECONDS}"
  done
}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${SNAPSHOT_ROOT}/src"
export DINOV3_REPO_DIR="/data/cpc/root/project/DINOv3"
export DINOV3_WEIGHTS_DIR="${ROOT}/weights/dinov3"
export MPLCONFIGDIR="/tmp/mpl_${EXPERIMENT}"
export XDG_CACHE_HOME="/tmp/cache_${EXPERIMENT}"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

COMMON_ARGS=(
  --subset medium
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST
  --epochs 24
  --batch-size 8
  --grad-accum-steps 2
  --num-workers 2
  --resolution 640
  --dec-layers 4
  --num-queries 300
  --num-select 300
  --dec-n-points 2
  --no-lite-refpoint-refine
  --bbox-refine-mode shared
  --scale-routing
  --scale-routing-mode legacy
  --p5-attention-bias 0.0
  --projector-scale P3 P4 P5
  --projector-type sdsr_v40_p4_learnable
  --sdsr-rank-channels 64
  --sdsr-cross-scale-mode none
  --sdsr-cross-scale-rank 32
  --no-sdsr-use-local-reassembly
  --no-sdsr-use-directional-guide
  --sdsr-use-phase-downsample
  --multi-scale
  --expanded-scales
  --aug-preset default
  --lr 0.0001
  --lr-encoder 0.00015
  --lr-drop 20
  --weight-decay 0.0001
  --lr-vit-layer-decay 0.8
  --lr-component-decay 0.7
  --use-cdn
  --dn-number 50
  --dn-box-noise-scale 0.6
  --dn-label-noise-scale 0.5
  --dn-loss-coef 0.5
  --backbone-register-border-tokens 1
  --seed 43
  --detector-init-seed 1043
  --use-dense-o2o
  --dense-o2o-mode enhanced
  --dense-o2o-start-epoch 2
  --dense-o2o-image-stop-epoch 12
  --dense-o2o-copyblend-stop-epoch 21
  --dense-o2o-mosaic-prob 0.5
  --dense-o2o-mixup-prob 0.0
  --dense-o2o-copyblend-prob 0.5
  --dense-o2o-copyblend-area-threshold 100
  --dense-o2o-copyblend-num-objects 1
  --dense-o2o-copyblend-expand-ratios 0.1 0.25
)

run_cell() {
  local group="$1"
  local cell="dense_nomix_g${group}_cdn50_l05"
  local run_name="coco_medium_s43_detinit1043_sdsr_v40_${cell}_lrd20_e24"
  local output_dir="${ROOT}/output/${run_name}"
  local train_log="${ROOT}/${run_name}_gpu${GPU_ID}.log"

  if [[ -e "${output_dir}" ]]; then
    printf 'Refusing to overwrite existing output: %s\n' "${output_dir}" >&2
    exit 4
  fi
  wait_for_gpu
  mkdir -p "${output_dir}"
  printf '[%s] START cell=%s gpu=%s\n' "$(date '+%F %T')" "${cell}" "${GPU_ID}" >> "${QUEUE_LOG}"
  "${PYTHON}" -u "${SNAPSHOT_ROOT}/run_coco_subset.py" \
    "${COMMON_ARGS[@]}" \
    --group-detr "${group}" \
    --output-dir "${output_dir}" >> "${train_log}" 2>&1
  printf '[%s] DONE cell=%s\n' "$(date '+%F %T')" "${cell}" >> "${QUEUE_LOG}"
}

printf '[%s] QUEUE START gpu=%s snapshot=%s\n' \
  "$(date '+%F %T')" "${GPU_ID}" "${SNAPSHOT_ROOT}" >> "${QUEUE_LOG}"

run_cell 4
run_cell 2

printf '[%s] QUEUE COMPLETE\n' "$(date '+%F %T')" >> "${QUEUE_LOG}"
