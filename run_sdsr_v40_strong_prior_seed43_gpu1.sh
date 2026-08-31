#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EXPERIMENT="sdsr_v40_strong_prior_medium_seed43"
SNAPSHOT_ROOT="${ROOT}/experiment_snapshots/${EXPERIMENT}"
MASTER_LOG="${ROOT}/${EXPERIMENT}_gpu${GPU_ID}_queue.log"
PID_FILE="${ROOT}/${EXPERIMENT}_gpu${GPU_ID}_queue.pid"
LOCK_FILE="${ROOT}/${EXPERIMENT}_gpu${GPU_ID}_queue.lock"

mkdir -p "${SNAPSHOT_ROOT}" "/tmp/mpl_${EXPERIMENT}" "/tmp/cache_${EXPERIMENT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'Another process holds %s\n' "${LOCK_FILE}" >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"
trap 'status=$?; rm -f "${PID_FILE}"; exit ${status}' EXIT

if [[ ! -f "${SNAPSHOT_ROOT}/.snapshot_complete" ]]; then
  if [[ -e "${SNAPSHOT_ROOT}/src" || -e "${SNAPSHOT_ROOT}/run_coco_subset.py" ]]; then
    printf 'Incomplete snapshot already exists: %s\n' "${SNAPSHOT_ROOT}" >&2
    exit 2
  fi
  cp -a "${ROOT}/src" "${SNAPSHOT_ROOT}/src"
  cp -a "${ROOT}/run_coco_subset.py" "${SNAPSHOT_ROOT}/run_coco_subset.py"
  find "${SNAPSHOT_ROOT}/src" -type f -name '*.py' -print0 \
    | sort -z | xargs -0 sha256sum > "${SNAPSHOT_ROOT}/source_sha256.txt"
  sha256sum "${SNAPSHOT_ROOT}/run_coco_subset.py" >> "${SNAPSHOT_ROOT}/source_sha256.txt"
  git -C "${ROOT}" rev-parse HEAD > "${SNAPSHOT_ROOT}/git_head.txt"
  git -C "${ROOT}" status --short > "${SNAPSHOT_ROOT}/git_status.txt"
  date '+%F %T %z' > "${SNAPSHOT_ROOT}/.snapshot_complete"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${SNAPSHOT_ROOT}/src"
export DINOV3_REPO_DIR="/data/cpc/root/project/DINOv3"
export DINOV3_WEIGHTS_DIR="${ROOT}/weights/dinov3"
export MPLCONFIGDIR="/tmp/mpl_${EXPERIMENT}"
export XDG_CACHE_HOME="/tmp/cache_${EXPERIMENT}"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

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
  --group-detr 6
  --dec-n-points 2
  --no-lite-refpoint-refine
  --bbox-refine-mode shared
  --scale-routing
  --scale-routing-mode legacy
  --p5-attention-bias 0.0
  --projector-scale P3 P4 P5
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
)

gpu_used_mib() {
  nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null \
    | sed -n "$((GPU_ID + 1))p" \
    | awk -F ',' '{gsub(/[^0-9]/, "", $3); print $3}'
}

wait_for_gpu() {
  local used_mib
  while true; do
    used_mib="$(gpu_used_mib || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_USED_MIB )); then
      printf '[%s] GPU_READY gpu=%s used=%sMiB\n' \
        "$(date '+%F %T')" "${GPU_ID}" "${used_mib}"
      return
    fi
    printf '[%s] WAIT gpu=%s used=%sMiB\n' \
      "$(date '+%F %T')" "${GPU_ID}" "${used_mib:-unknown}"
    sleep "${POLL_SECONDS}"
  done
}

run_cell() {
  local cell="$1"
  local projector_type="$2"
  local description="$3"
  local run_name="coco_medium_s43_detinit1043_${cell}_lrd20_e24"
  local output_dir="${ROOT}/output/${run_name}"
  local train_log="${ROOT}/${run_name}_gpu${GPU_ID}.log"

  if [[ -s "${output_dir}/log.txt" ]] \
      && (( $(wc -l < "${output_dir}/log.txt") >= 24 )); then
    printf '[%s] SKIP cell=%s already complete\n' "$(date '+%F %T')" "${cell}"
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    printf '[%s] REFUSE partial output exists: %s\n' \
      "$(date '+%F %T')" "${output_dir}" >&2
    exit 3
  fi

  wait_for_gpu
  mkdir -p "${output_dir}"
  {
    printf '[%s] START cell=%s projector=%s gpu=%s\n' \
      "$(date '+%F %T')" "${cell}" "${projector_type}" "${GPU_ID}"
    printf 'description=%s\nsnapshot=%s\noutput_dir=%s\n' \
      "${description}" "${SNAPSHOT_ROOT}" "${output_dir}"
  } | tee -a "${train_log}"

  "${PYTHON}" -u "${SNAPSHOT_ROOT}/run_coco_subset.py" \
    "${COMMON_ARGS[@]}" --projector-type "${projector_type}" \
    --output-dir "${output_dir}" >> "${train_log}" 2>&1
  printf '[%s] DONE cell=%s\n' \
    "$(date '+%F %T')" "${cell}" | tee -a "${train_log}"
}

exec >> "${MASTER_LOG}" 2>&1
printf '[%s] QUEUE START gpu=%s snapshot=%s\n' \
  "$(date '+%F %T')" "${GPU_ID}" "${SNAPSHOT_ROOT}"
run_cell "sdsr_v40_p4_fixed_strong" "sdsr_v40_p4_fixed" \
  "Fixed two-seed mean P4 depth prior with no trainable routing parameters"
run_cell "sdsr_v40_p4_learnable_strong" "sdsr_v40_p4_learnable" \
  "Four learnable logits initialized from the same strong P4 depth prior"
run_cell "sdsr_v40_p4_bounded_dynamic" "sdsr_v40_p4_bounded_dynamic" \
  "Fixed strong prior plus a zero-initialized bounded per-image logit residual"
printf '[%s] QUEUE COMPLETE\n' "$(date '+%F %T')"
