#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
OUTPUT_ROOT="${ROOT}/output/posthoc_backbone_refinement"
STATUS_LOG="${ROOT}/posthoc_backbone_refinement_corrected_eval_status.log"

cd "${ROOT}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${STATUS_LOG}"
}

evaluate_detector() {
  local variant="$1"
  local gpu="$2"
  local checkpoint="$3"
  local output_dir="${OUTPUT_ROOT}/eval_${variant}_corrected"
  local log_file="${ROOT}/posthoc_backbone_refinement_eval_${variant}_corrected_gpu${gpu}.log"

  log_status "START corrected ${variant} eval gpu=${gpu}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    DINOV3_REPO_DIR="/data/cpc/root/project/DINOv3" \
    DINOV3_WEIGHTS_DIR="${ROOT}/weights/dinov3" \
    MPLCONFIGDIR="/tmp/mpl_posthoc_corrected_${gpu}" \
    XDG_CACHE_HOME="/tmp/cache_posthoc_corrected_${gpu}" \
    ALBUMENTATIONS_DISABLE_VERSION_CHECK=1 \
    "${PYTHON}" -u run_coco_subset.py \
      --subset medium \
      --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
      --eval-only \
      --resume "${checkpoint}" \
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
      --out-feature-indexes 2 5 8 11 \
      --projector-scale P3 P4 P5 \
      --projector-p3-indexes 2 5 11 \
      --projector-p4-indexes 2 5 8 11 \
      --projector-p5-indexes 2 8 11 \
      --projector-source-mode prune \
      --projector-c2f-blocks 3 3 3 \
      --projector-type multiscale \
      --projector-p5-mode group2_first_full \
      --projector-resample-share p5 \
      --multi-scale \
      --expanded-scales \
      --backbone-register-border-tokens 1 \
      --eval-max-dets 100 \
      --output-dir "${output_dir}" \
      > "${log_file}" 2>&1
  log_status "DONE corrected ${variant} eval gpu=${gpu}"
}

evaluate_detector lazy 0 "${OUTPUT_ROOT}/detector_posthoc_lazy.pth" &
PID0=$!
evaluate_detector dense 1 "${OUTPUT_ROOT}/detector_posthoc_dense.pth" &
PID1=$!
status=0
wait "${PID0}" || status=1
wait "${PID1}" || status=1
if (( status != 0 )); then
  log_status "CONTROLLER FAILED"
  exit "${status}"
fi
log_status "CONTROLLER COMPLETE"
