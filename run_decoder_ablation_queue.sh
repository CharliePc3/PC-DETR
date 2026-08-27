#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index.}"
QUEUE_NAME="${QUEUE_NAME:?Set QUEUE_NAME to a supported decoder ablation queue.}"
SEED="${SEED:-42}"
QUEUE_LOG="${ROOT}/decoder_ablation_${QUEUE_NAME}_queue_nohup.log"
PID_FILE="${ROOT}/decoder_ablation_${QUEUE_NAME}_queue.pid"
LOCK_FILE="${ROOT}/decoder_ablation_${QUEUE_NAME}_queue.lock"
CACHE_ROOT="/tmp/rfdetr_decoder_ablation_${GPU_ID}"

cd "${ROOT}"
mkdir -p "${CACHE_ROOT}/matplotlib" "${CACHE_ROOT}/cache/torch/kernels"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/cache"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another decoder ablation queue holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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

COMMON_ARGS=(
  run_coco_subset.py
  --subset medium
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST
  --epochs 24
  --batch-size 8
  --grad-accum-steps 2
  --num-workers 2
  --device cuda
  --seed "${SEED}"
  --resolution 640
  --dec-layers 4
  --num-queries 300
  --num-select 300
  --group-detr 6
  --out-feature-indexes 2 5 8 11
  --projector-scale P3 P4 P5
  --projector-type multiscale
  --multi-scale
  --expanded-scales
  --aug-preset default
  --lr 1e-4
  --lr-encoder 1.5e-4
  --lr-drop 100
  --warmup-epochs 0
  --weight-decay 1e-4
  --lr-vit-layer-decay 0.8
  --lr-component-decay 0.7
  --use-cdn
  --dn-number 50
  --dn-label-noise-scale 0.5
  --dn-box-noise-scale 0.6
  --dn-loss-coef 0.5
  --dn-neg-loss-coef 1.0
  --backbone-register-border-tokens 1
  --backbone-register-fill randn
  --backbone-register-noise-std 1.0
  --eval-max-dets 100
)

run_experiment() {
  local name=$1
  shift
  local output_dir="${ROOT}/output/${name}"
  local log_file="${ROOT}/${name}_nohup.log"

  if [[ -s "${output_dir}/results.json" ]] && rg -q "Best AP50:95 over training" "${log_file}"; then
    log_status "SKIP ${name}: completed result exists"
    return
  fi

  log_status "START ${name} on physical GPU ${GPU_ID}"
  "${PYTHON}" -u "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    "$@" \
    > "${log_file}" 2>&1
  log_status "DONE ${name}"
}

case "${QUEUE_NAME}" in
  gpu2_main)
    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_iterref_points4_e24 \
      --query-init learned \
      --no-lite-refpoint-refine \
      --dec-n-points 4

    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_iterref_scaleroute_e24 \
      --query-init learned \
      --no-lite-refpoint-refine \
      --dec-n-points 2 \
      --scale-routing

    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_iterref_points4_scaleroute_e24 \
      --query-init learned \
      --no-lite-refpoint-refine \
      --dec-n-points 4 \
      --scale-routing
    ;;

  gpu3_aux)
    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_scaleroute_e24 \
      --query-init learned \
      --lite-refpoint-refine \
      --dec-n-points 2 \
      --scale-routing

    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_hicq_grad_gate0_e24 \
      --query-init hybrid_topk \
      --no-query-memory-detach \
      --query-init-gate 0 \
      --lite-refpoint-refine \
      --dec-n-points 2
    ;;

  seed43_gpu2)
    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_b0_seed43_e24 \
      --query-init learned \
      --lite-refpoint-refine \
      --dec-n-points 2

    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_iterref_scaleroute_seed43_e24 \
      --query-init learned \
      --no-lite-refpoint-refine \
      --dec-n-points 2 \
      --scale-routing
    ;;

  seed43_gpu3)
    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_iterref_seed43_e24 \
      --query-init learned \
      --no-lite-refpoint-refine \
      --dec-n-points 2
    ;;

  routing_v2_gpu2)
    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_iterref_scaleroute_strictinit_e24 \
      --query-init learned \
      --no-lite-refpoint-refine \
      --dec-n-points 2 \
      --scale-routing

    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_iterref_scaleroute_strictinit_early3_e24 \
      --query-init learned \
      --no-lite-refpoint-refine \
      --dec-n-points 2 \
      --scale-routing \
      --scale-routing-layers 0 1 2
    ;;

  routing_v2_gpu3)
    run_experiment \
      coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_iterref_cellroute_e24 \
      --query-init learned \
      --no-lite-refpoint-refine \
      --dec-n-points 2 \
      --scale-routing \
      --scale-routing-mode cell
    ;;

  *)
    printf 'Unknown QUEUE_NAME: %s\n' "${QUEUE_NAME}" >&2
    exit 2
    ;;
esac
