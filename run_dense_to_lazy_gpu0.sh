#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-0}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DENSE_INIT="${ROOT}/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_dsdr_dense_only_e1/checkpoints/model_epoch1.pt"
REFINE_NAME="dinov3_small_coco5k_dense_to_lazy_v2_e2"
REFINE_DIR="${ROOT}/output/lazystrike_refine/${REFINE_NAME}"
REFINE_LOG="${ROOT}/train_${REFINE_NAME}_nohup.log"
DETECT_NAME="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_dense_to_lazy_v2_epoch2_e24"
DETECT_DIR="${ROOT}/output/${DETECT_NAME}"
DETECT_LOG="${ROOT}/medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_dense_to_lazy_v2_epoch2_e24_nohup.log"
RESULT_DIR="${ROOT}/output/dense_to_lazy"
RESULT_TSV="${RESULT_DIR}/summary.tsv"
QUEUE_LOG="${ROOT}/dense_to_lazy_gpu0_queue_nohup.log"
PID_FILE="${ROOT}/dense_to_lazy_gpu0_queue.pid"
LOCK_FILE="${ROOT}/dense_to_lazy_gpu0_queue.lock"

cd "${ROOT}"
mkdir -p "${RESULT_DIR}" /tmp/matplotlib_dense_to_lazy
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}"
export MPLCONFIGDIR="/tmp/matplotlib_dense_to_lazy"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another Dense-to-Lazy queue holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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

run_refinement() {
  if [[ -s "${REFINE_DIR}/checkpoints/model_epoch2.pt" ]]; then
    log_status "SKIP refinement: epoch-2 checkpoint exists"
    return
  fi
  wait_for_gpu
  log_status "START Dense-to-Lazy refinement from ${DENSE_INIT}"
  "${PYTHON}" -u tools/train_dinov3_lazystrike_refine.py \
    --data-root /data/cpc/root/dataset/COCO \
    --split train2017 \
    --output-dir "${REFINE_DIR}" \
    --encoder dinov3_small \
    --pretrained-encoder "${DENSE_INIT}" \
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
    --log-every 50 \
    --save-every-epoch \
    > "${REFINE_LOG}" 2>&1
  log_status "DONE Dense-to-Lazy refinement"
}

run_diagnostic() {
  epoch="$1"
  checkpoint="${REFINE_DIR}/checkpoints/model_epoch${epoch}.pt"
  diag_dir="${ROOT}/output/token_analysis/dense_to_lazy_v2_epoch${epoch}_cocoval200"
  diag_log="${ROOT}/token_analysis_dense_to_lazy_v2_epoch${epoch}_cocoval200_nohup.log"

  if [[ -s "${diag_dir}/metrics.json" ]]; then
    log_status "SKIP epoch-${epoch} diagnostic: metrics exist"
    return
  fi
  wait_for_gpu
  log_status "START epoch-${epoch} token diagnostic"
  "${PYTHON}" -u tools/analyze_dinov3_tokens.py \
    --data-root /data/cpc/root/dataset/COCO \
    --split val2017 \
    --output-dir "${diag_dir}" \
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
    > "${diag_log}" 2>&1
  log_status "DONE epoch-${epoch} token diagnostic"
}

run_detection() {
  if [[ -s "${DETECT_DIR}/log.txt" ]] && [[ $(wc -l < "${DETECT_DIR}/log.txt") -ge 24 ]]; then
    log_status "SKIP epoch-2 detection: 24 epoch records exist"
    return
  fi
  wait_for_gpu
  log_status "START aligned epoch-2 medium detection"
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
    --pretrained-encoder "${REFINE_DIR}/checkpoints/model_epoch2.pt" \
    --output-dir "${DETECT_DIR}" \
    > "${DETECT_LOG}" 2>&1
  log_status "DONE aligned epoch-2 medium detection"
}

record_results() {
  if [[ -s "${RESULT_TSV}" ]]; then
    log_status "SKIP summary: ${RESULT_TSV} exists"
    return
  fi

  printf 'variant\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\tlast5_AP\tcoverage100\tsmall_coverage100\tmedium_coverage100\tlarge_coverage100\tcls_hflip_cosine\tfp_gp_proxy_ratio\n' > "${RESULT_TSV}"
  detection=$(jq -s -r '
    map(select(.test_coco_eval_bbox != null)) as $rows
    | ($rows | max_by(.test_coco_eval_bbox[0])) as $best
    | ($rows | sort_by(.epoch) | .[-5:]
       | map(.test_coco_eval_bbox[0]) | add / length) as $last5
    | [($best.epoch + 1), $best.test_coco_eval_bbox[0],
       $best.test_coco_eval_bbox[1], $best.test_coco_eval_bbox[2],
       $best.test_coco_eval_bbox[3], $best.test_coco_eval_bbox[4],
       $best.test_coco_eval_bbox[5], $last5] | @tsv
  ' "${DETECT_DIR}/log.txt")
  diagnostic=$(jq -r '[.summary.patch_score_box_coverage_top100,
    .summary.patch_score_small_box_coverage_top100,
    .summary.patch_score_medium_box_coverage_top100,
    .summary.patch_score_large_box_coverage_top100,
    .summary.cls_hflip_cosine, .summary.fp_gp_proxy_ratio] | @tsv' \
    "${ROOT}/output/token_analysis/dense_to_lazy_v2_epoch2_cocoval200/metrics.json")
  printf 'dense_to_lazy_v2_epoch2\t%s\t%s\n' "${detection}" "${diagnostic}" >> "${RESULT_TSV}"
  log_status "WROTE ${RESULT_TSV}"
}

log_status "QUEUE START on physical GPU ${GPU_ID}"
run_refinement
run_diagnostic 1
run_diagnostic 2
run_detection
record_results
