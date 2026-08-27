#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
CHECKPOINT_ROOT="${ROOT}/output/token_analysis/online_refinement/checkpoints"
OUTPUT_ROOT="${ROOT}/output/token_analysis/online_refinement"

cd "${ROOT}"

run_diagnostic() {
  local variant="$1"
  local gpu="$2"
  local checkpoint="${3:-${CHECKPOINT_ROOT}/${variant}_best_backbone.pt}"
  local output_dir="${OUTPUT_ROOT}/${variant}_cocoval200"
  local log_file="${ROOT}/token_analysis_online_${variant}_cocoval200.log"

  if [[ -s "${output_dir}/metrics.json" ]]; then
    return
  fi

  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    MPLCONFIGDIR="/tmp/mpl_token_online_${gpu}" \
    XDG_CACHE_HOME="/tmp/cache_token_online_${gpu}" \
    ALBUMENTATIONS_DISABLE_VERSION_CHECK=1 \
    "${PYTHON}" -u tools/analyze_dinov3_tokens.py \
      --data-root /data/cpc/root/dataset/COCO \
      --split val2017 \
      --output-dir "${output_dir}" \
      --encoder dinov3_small \
      --pretrained-encoder "${checkpoint}" \
      --resolution 640 \
      --num-images 200 \
      --visualize 12 \
      --topk 10 \
      --pib-topk 1 5 10 \
      --coverage-topk 10 50 100 \
      --size-coverage-topk 100 \
      --dominance-topk 10 50 100 \
      --last-topk 1 \
      --consistency \
      --fp-gp-sigma 1 \
      --gp-exclude-radius 1 \
      --device cuda \
      --seed 42 \
      > "${log_file}" 2>&1
}

run_diagnostic object_global_c0p1 0 &
PID0=$!
run_diagnostic object_global_c0p05 1 &
PID1=$!
status=0
wait "${PID0}" || status=1
wait "${PID1}" || status=1
if (( status != 0 )); then
  exit "${status}"
fi

run_diagnostic posthoc_lazy 0 \
  "${ROOT}/output/posthoc_backbone_refinement/lazy_from_detector_best/checkpoints/model_final.pt" &
PID0=$!
run_diagnostic posthoc_dense 1 \
  "${ROOT}/output/posthoc_backbone_refinement/dense_from_detector_best/checkpoints/model_final.pt" &
PID1=$!
wait "${PID0}" || status=1
wait "${PID1}" || status=1
exit "${status}"
