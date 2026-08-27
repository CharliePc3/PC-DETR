#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:-0}"
EXPERIMENT="v23_lazystrike_dense_o2o_factorial_seed42"
SNAPSHOT_ROOT="${ROOT}/experiment_snapshots/${EXPERIMENT}"
MASTER_LOG="${ROOT}/${EXPERIMENT}_gpu${GPU_ID}_queue.log"
PID_FILE="${ROOT}/${EXPERIMENT}_gpu${GPU_ID}_queue.pid"
LOCK_FILE="${ROOT}/${EXPERIMENT}_gpu${GPU_ID}_queue.lock"
LAZYSTRIKE_CKPT="${ROOT}/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_v2_conservative/checkpoints/model_final.pt"

if [[ ! -s "${LAZYSTRIKE_CKPT}" ]]; then
  printf 'Missing LazyStrike checkpoint: %s\n' "${LAZYSTRIKE_CKPT}" >&2
  exit 2
fi

mkdir -p "${SNAPSHOT_ROOT}" "/tmp/mpl_${EXPERIMENT}" "/tmp/cache_${EXPERIMENT}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'Another process holds %s\n' "${LOCK_FILE}" >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"
trap 'status=$?; rm -f "${PID_FILE}"; exit ${status}' EXIT

# Freeze the exact dirty working-tree source once so all four cells use identical code.
if [[ ! -f "${SNAPSHOT_ROOT}/.snapshot_complete" ]]; then
  rm -rf "${SNAPSHOT_ROOT}/src"
  cp -a "${ROOT}/src" "${SNAPSHOT_ROOT}/src"
  cp -a "${ROOT}/run_coco_subset.py" "${SNAPSHOT_ROOT}/run_coco_subset.py"
  find "${SNAPSHOT_ROOT}/src" -type f -name '*.py' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${SNAPSHOT_ROOT}/source_sha256.txt"
  sha256sum "${SNAPSHOT_ROOT}/run_coco_subset.py" >> "${SNAPSHOT_ROOT}/source_sha256.txt"
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
  --projector-type sdsr_v23
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

DENSE_O2O_ARGS=(
  --use-dense-o2o
  --dense-o2o-mode enhanced
  --dense-o2o-start-epoch 2
  --dense-o2o-image-stop-epoch 12
  --dense-o2o-copyblend-stop-epoch 21
  --dense-o2o-mosaic-prob 0.5
  --dense-o2o-mixup-prob 0.5
  --dense-o2o-copyblend-prob 0.5
  --dense-o2o-copyblend-area-threshold 100
  --dense-o2o-copyblend-num-objects 1
  --dense-o2o-copyblend-expand-ratios 0.1 0.25
)

run_cell() {
  local cell="$1"
  local backbone="$2"
  local dense_o2o="$3"
  local run_name="coco_medium_s42_v23_factorial_${cell}"
  local output_dir="${ROOT}/output/${run_name}"
  local train_log="${ROOT}/${run_name}_gpu${GPU_ID}.log"
  local -a extra_args=()

  if [[ "${backbone}" == "lazystrike" ]]; then
    extra_args+=(--pretrained-encoder "${LAZYSTRIKE_CKPT}")
  fi
  if [[ "${dense_o2o}" == "on" ]]; then
    extra_args+=("${DENSE_O2O_ARGS[@]}")
  fi

  if [[ -s "${output_dir}/log.txt" ]] \
      && (( $(wc -l < "${output_dir}/log.txt") >= 24 )); then
    printf '[%s] SKIP cell=%s already complete\n' "$(date '+%F %T')" "${cell}"
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    printf '[%s] REFUSE cell=%s partial output exists: %s\n' \
      "$(date '+%F %T')" "${cell}" "${output_dir}" >&2
    exit 3
  fi

  mkdir -p "${output_dir}"
  {
    printf '[%s] START cell=%s backbone=%s dense_o2o=%s gpu=%s\n' \
      "$(date '+%F %T')" "${cell}" "${backbone}" "${dense_o2o}" "${GPU_ID}"
    printf 'snapshot=%s\n' "${SNAPSHOT_ROOT}"
    printf 'output_dir=%s\n' "${output_dir}"
  } | tee -a "${train_log}"

  "${PYTHON}" -u "${SNAPSHOT_ROOT}/run_coco_subset.py" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    "${extra_args[@]}" >> "${train_log}" 2>&1

  printf '[%s] DONE cell=%s\n' "$(date '+%F %T')" "${cell}" | tee -a "${train_log}"
}

exec >> "${MASTER_LOG}" 2>&1
printf '[%s] QUEUE START gpu=%s snapshot=%s\n' \
  "$(date '+%F %T')" "${GPU_ID}" "${SNAPSHOT_ROOT}"

# A/B/C/D form the complete 2x2 backbone-initialization x Dense-O2O factorial.
run_cell "A_official_o2o0" "official" "off"
run_cell "B_official_o2o1" "official" "on"
run_cell "C_lazystrike_o2o0" "lazystrike" "off"
run_cell "D_lazystrike_o2o1" "lazystrike" "on"

printf '[%s] QUEUE COMPLETE\n' "$(date '+%F %T')"
