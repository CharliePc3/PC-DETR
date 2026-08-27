#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
EXPERIMENT="posthoc_backbone_refinement"
OUTPUT_ROOT="${ROOT}/output/${EXPERIMENT}"
STATUS_LOG="${ROOT}/${EXPERIMENT}_status.log"
DETECTOR="${ROOT}/output/integrated_refinement_s2_denseo2o/official_seed42/checkpoint_best_regular.pth"
INITIAL_BACKBONE="${OUTPUT_ROOT}/official_best_ap36p103_backbone.pt"
GPU_THRESHOLD_MIB="${GPU_THRESHOLD_MIB:-2500}"

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${STATUS_LOG}"
}

wait_for_gpu() {
  local gpu="$1"
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    if (( used <= GPU_THRESHOLD_MIB )); then
      return
    fi
    sleep 300
  done
}

evaluate_detector() {
  local variant="$1"
  local gpu="$2"
  local checkpoint="$3"
  local output_dir="${OUTPUT_ROOT}/eval_${variant}"
  local log_file="${ROOT}/${EXPERIMENT}_eval_${variant}_gpu${gpu}.log"

  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    DINOV3_REPO_DIR="/data/cpc/root/project/DINOv3" \
    DINOV3_WEIGHTS_DIR="${ROOT}/weights/dinov3" \
    MPLCONFIGDIR="/tmp/mpl_${EXPERIMENT}_${gpu}" \
    XDG_CACHE_HOME="/tmp/cache_${EXPERIMENT}_${gpu}" \
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
}

run_lazy() {
  local gpu=0
  local refine_dir="${OUTPUT_ROOT}/lazy_from_detector_best"
  local refined="${refine_dir}/checkpoints/model_final.pt"
  local detector_out="${OUTPUT_ROOT}/detector_posthoc_lazy.pth"
  wait_for_gpu "${gpu}"
  log_status "START posthoc LazyStrike gpu=${gpu}"
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/src" \
    "${PYTHON}" -u tools/train_dinov3_lazystrike_refine.py \
      --data-root /data/cpc/root/dataset/COCO \
      --split train2017 \
      --output-dir "${refine_dir}" \
      --encoder dinov3_small \
      --pretrained-encoder "${INITIAL_BACKBONE}" \
      --objectives lazystrike \
      --resolution 640 \
      --num-images 5000 \
      --epochs 2 \
      --batch-size 2 \
      --workers 4 \
      --lr 5e-6 \
      --weight-decay 0.05 \
      --train-last-blocks 2 \
      --max-boxes-per-image 12 \
      --min-box-patches 1 \
      --lazy-topk 1 \
      --lazy-target-weight 0.1 \
      --cover-margin 0.1 \
      --cover-temperature 0.1 \
      --lambda-align 0.25 \
      --lambda-cover 0.15 \
      --lambda-consistency 0.25 \
      --lambda-distill 1.0 \
      --device cuda \
      --seed 42 \
      --shuffle \
      --save-every-epoch \
      > "${ROOT}/${EXPERIMENT}_lazy_refine_gpu${gpu}.log" 2>&1
  "${PYTHON}" tools/replace_dinov3_backbone.py \
    --detector "${DETECTOR}" --backbone "${refined}" --output "${detector_out}"
  evaluate_detector lazy "${gpu}" "${detector_out}"
  log_status "DONE posthoc LazyStrike gpu=${gpu}"
}

run_dense() {
  local gpu=1
  local refine_dir="${OUTPUT_ROOT}/dense_from_detector_best"
  local refined="${refine_dir}/checkpoints/model_final.pt"
  local detector_out="${OUTPUT_ROOT}/detector_posthoc_dense.pth"
  wait_for_gpu "${gpu}"
  log_status "START posthoc Dense gpu=${gpu}"
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/src" \
    "${PYTHON}" -u tools/train_dinov3_lazystrike_refine.py \
      --data-root /data/cpc/root/dataset/COCO \
      --split train2017 \
      --output-dir "${refine_dir}" \
      --encoder dinov3_small \
      --pretrained-encoder "${INITIAL_BACKBONE}" \
      --objectives dense \
      --resolution 640 \
      --num-images 5000 \
      --epochs 1 \
      --batch-size 2 \
      --workers 4 \
      --lr 2.5e-6 \
      --weight-decay 0.05 \
      --train-last-blocks 4 \
      --max-boxes-per-image 12 \
      --min-box-patches 1 \
      --lambda-align 0 \
      --lambda-cover 0 \
      --lambda-consistency 0 \
      --lambda-distill 1.0 \
      --dense-layers 8 11 \
      --dense-layer-weights 0.35 0.65 \
      --lambda-dense-object 0.15 \
      --lambda-dense-global 0.05 \
      --dense-min-overlap 0 \
      --layer-lr-decay 0.8 \
      --device cuda \
      --seed 42 \
      --shuffle \
      --save-every-epoch \
      > "${ROOT}/${EXPERIMENT}_dense_refine_gpu${gpu}.log" 2>&1
  "${PYTHON}" tools/replace_dinov3_backbone.py \
    --detector "${DETECTOR}" --backbone "${refined}" --output "${detector_out}"
  evaluate_detector dense "${gpu}" "${detector_out}"
  log_status "DONE posthoc Dense gpu=${gpu}"
}

run_lazy &
PID0=$!
run_dense &
PID1=$!
status=0
wait "${PID0}" || status=1
wait "${PID1}" || status=1
if (( status != 0 )); then
  log_status "CONTROLLER FAILED"
  exit "${status}"
fi
log_status "CONTROLLER COMPLETE"
