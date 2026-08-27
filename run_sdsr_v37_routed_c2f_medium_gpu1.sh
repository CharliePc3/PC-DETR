#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:-1}"
EXPERIMENT="sdsr_v37_routed_c2f_medium_seed42"
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

# Freeze the dirty working tree once so later edits cannot alter queued cells.
if [[ ! -f "${SNAPSHOT_ROOT}/.snapshot_complete" ]]; then
  if [[ -e "${SNAPSHOT_ROOT}/src" || -e "${SNAPSHOT_ROOT}/run_coco_subset.py" ]]; then
    printf 'Incomplete snapshot already exists: %s\n' "${SNAPSHOT_ROOT}" >&2
    exit 2
  fi
  cp -a "${ROOT}/src" "${SNAPSHOT_ROOT}/src"
  cp -a "${ROOT}/run_coco_subset.py" "${SNAPSHOT_ROOT}/run_coco_subset.py"
  find "${SNAPSHOT_ROOT}/src" -type f -name '*.py' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${SNAPSHOT_ROOT}/source_sha256.txt"
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
  --seed 42
  --detector-init-seed 1042
)

run_cell() {
  local projector_type="$1"
  local description="$2"
  local run_name="coco_medium_s42_detinit1042_${projector_type}_routed_c2f_lrd20_e24"
  local output_dir="${ROOT}/output/${run_name}"
  local train_log="${ROOT}/${run_name}_gpu${GPU_ID}.log"

  if [[ -s "${output_dir}/log.txt" ]] \
      && (( $(wc -l < "${output_dir}/log.txt") >= 24 )); then
    printf '[%s] SKIP projector=%s already complete\n' \
      "$(date '+%F %T')" "${projector_type}"
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    printf '[%s] REFUSE partial output exists: %s\n' \
      "$(date '+%F %T')" "${output_dir}" >&2
    exit 3
  fi

  mkdir -p "${output_dir}"
  {
    printf '[%s] START projector=%s gpu=%s\n' \
      "$(date '+%F %T')" "${projector_type}" "${GPU_ID}"
    printf 'description=%s\n' "${description}"
    printf 'snapshot=%s\n' "${SNAPSHOT_ROOT}"
    printf 'output_dir=%s\n' "${output_dir}"
  } | tee -a "${train_log}"

  "${PYTHON}" -u "${SNAPSHOT_ROOT}/run_coco_subset.py" \
    "${COMMON_ARGS[@]}" \
    --projector-type "${projector_type}" \
    --output-dir "${output_dir}" >> "${train_log}" 2>&1

  printf '[%s] DONE projector=%s\n' \
    "$(date '+%F %T')" "${projector_type}" | tee -a "${train_log}"
}

exec >> "${MASTER_LOG}" 2>&1
printf '[%s] QUEUE START gpu=%s snapshot=%s\n' \
  "$(date '+%F %T')" "${GPU_ID}" "${SNAPSHOT_ROOT}"

run_cell sdsr_v37_r1 "P3 layer-preserving routing plus C2f; exact v23 P4/P5"
run_cell sdsr_v37_r2 "P4 layer-preserving routing plus C2f; exact v23 P3/P5"
run_cell sdsr_v37_r3 "Independent P3/P4 layer-preserving routing plus C2f; exact v23 P5"

printf '[%s] QUEUE COMPLETE\n' "$(date '+%F %T')"
