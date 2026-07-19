#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-3}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
VARIANT="base_b811lr02_g002_obj6040"
REFINE_NAME="dinov3_small_coco5k_dense_mid_${VARIANT}_e1"
REFINE_DIR="${ROOT}/output/lazystrike_refine/${REFINE_NAME}"
REFINE_LOG="${ROOT}/train_${REFINE_NAME}_nohup.log"
DIAG_DIR="${ROOT}/output/token_analysis/dense_mid_${VARIANT}_cocoval200"
DIAG_LOG="${ROOT}/token_analysis_dense_mid_${VARIANT}_cocoval200_nohup.log"
DETECT_NAME="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_dense_mid_${VARIANT}_e1_e24"
DETECT_DIR="${ROOT}/output/${DETECT_NAME}"
DETECT_LOG="${ROOT}/medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_dense_mid_${VARIANT}_e1_e24_nohup.log"
QUEUE_LOG="${ROOT}/dense_midpoint_candidate_gpu3_queue_nohup.log"
PID_FILE="${ROOT}/dense_midpoint_candidate_gpu3_queue.pid"
LOCK_FILE="${ROOT}/dense_midpoint_candidate_gpu3_queue.lock"
SUMMARY_TSV="${ROOT}/output/lazystrike_dense_optimization/summary.tsv"

cd "${ROOT}"
mkdir -p "$(dirname "${SUMMARY_TSV}")" /tmp/matplotlib_dense_mid
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}"
export MPLCONFIGDIR="/tmp/matplotlib_dense_mid"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another midpoint queue holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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

run_refine() {
  if [[ -s "${REFINE_DIR}/checkpoints/model_epoch1.pt" ]]; then
    log_status "SKIP refinement: checkpoint exists"
    return
  fi
  wait_for_gpu
  log_status "START refinement ${VARIANT}"
  "${PYTHON}" -u tools/train_dinov3_lazystrike_refine.py \
    --data-root /data/cpc/root/dataset/COCO \
    --split train2017 \
    --output-dir "${REFINE_DIR}" \
    --encoder dinov3_small \
    --resolution 640 \
    --num-images 5000 \
    --epochs 1 \
    --batch-size 2 \
    --workers 4 \
    --lr 2.5e-6 \
    --weight-decay 0.05 \
    --train-block-indexes 8 9 10 11 \
    --train-block-lr-scales 0.512 0.64 0.8 0.2 \
    --freeze-final-norm \
    --max-boxes-per-image 12 \
    --min-box-patches 1 \
    --lazy-topk 1 \
    --lazy-target-weight 0.10 \
    --cover-margin 0.10 \
    --cover-temperature 0.10 \
    --lambda-align 0.0 \
    --lambda-cover 0.0 \
    --lambda-consistency 0.0 \
    --lambda-distill 1.5 \
    --dense-layers 8 11 \
    --dense-layer-weights 0.25 0.75 \
    --dense-object-layer-weights 0.6 0.4 \
    --lambda-dense-object 0.20 \
    --lambda-dense-global 0.02 \
    --device cuda \
    --seed 42 \
    --shuffle \
    --log-every 50 \
    --save-every-epoch \
    > "${REFINE_LOG}" 2>&1
  log_status "DONE refinement"
}

run_diagnostic() {
  if [[ -s "${DIAG_DIR}/metrics.json" ]]; then
    log_status "SKIP diagnostic: metrics exist"
    return
  fi
  wait_for_gpu
  log_status "START token diagnostic"
  "${PYTHON}" -u tools/analyze_dinov3_tokens.py \
    --data-root /data/cpc/root/dataset/COCO \
    --split val2017 \
    --output-dir "${DIAG_DIR}" \
    --encoder dinov3_small \
    --pretrained-encoder "${REFINE_DIR}/checkpoints/model_epoch1.pt" \
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
    > "${DIAG_LOG}" 2>&1
  log_status "DONE token diagnostic"
}

run_detection() {
  if [[ -s "${DETECT_DIR}/log.txt" ]] && [[ $(wc -l < "${DETECT_DIR}/log.txt") -ge 24 ]]; then
    log_status "SKIP detection: 24 epoch records exist"
    return
  fi
  wait_for_gpu
  log_status "START aligned medium detection"
  "${PYTHON}" -u run_coco_subset.py \
    --subset medium \
    --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
    --epochs 24 \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --num-workers 2 \
    --device cuda \
    --seed 42 \
    --group-detr 6 \
    --projector-scale P3 P4 P5 \
    --multi-scale \
    --expanded-scales \
    --resolution 640 \
    --dec-layers 4 \
    --num-queries 300 \
    --num-select 300 \
    --use-cdn \
    --dn-number 50 \
    --dn-label-noise-scale 0.5 \
    --dn-box-noise-scale 0.6 \
    --dn-loss-coef 0.5 \
    --pretrained-encoder "${REFINE_DIR}/checkpoints/model_epoch1.pt" \
    --output-dir "${DETECT_DIR}" \
    > "${DETECT_LOG}" 2>&1
  log_status "DONE aligned medium detection"
}

record_result() {
  if [[ ! -s "${SUMMARY_TSV}" ]]; then
    printf 'variant\tinit\tlambda_global\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\tcls_hflip_cosine\tlast_vote_hflip_consistency\tfp_gp_proxy_ratio\n' > "${SUMMARY_TSV}"
  fi
  if rg -q "^${VARIANT}\t" "${SUMMARY_TSV}"; then
    log_status "SKIP summary: row exists"
    return
  fi
  detection=$(jq -s -r '
    map(select(.test_coco_eval_bbox != null))
    | max_by(.test_coco_eval_bbox[0])
    | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
       .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
       .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
    | @tsv
  ' "${DETECT_DIR}/log.txt")
  diagnostic=$(jq -r '[.summary.cls_hflip_cosine, .summary.last_vote_hflip_consistency, .summary.fp_gp_proxy_ratio] | @tsv' "${DIAG_DIR}/metrics.json")
  printf '%s\tbase\t0.02\t%s\t%s\n' "${VARIANT}" "${detection}" "${diagnostic}" >> "${SUMMARY_TSV}"
}

log_status "QUEUE START on physical GPU ${GPU_ID}"
run_refine
run_diagnostic
run_detection
record_result
