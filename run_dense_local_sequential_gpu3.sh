#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-3}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
V2_CHECKPOINT="${ROOT}/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_v2_conservative/checkpoints/model_final.pt"
QUEUE_LOG="${ROOT}/dense_local_sequential_gpu3_queue_nohup.log"
PID_FILE="${ROOT}/dense_local_sequential_gpu3_queue.pid"
LOCK_FILE="${ROOT}/dense_local_sequential_gpu3_queue.lock"
SUMMARY_DIR="${ROOT}/output/lazystrike_dense_optimization"
SUMMARY_TSV="${SUMMARY_DIR}/summary.tsv"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}" /tmp/matplotlib_dense_local
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}"
export MPLCONFIGDIR="/tmp/matplotlib_dense_local"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another dense-local queue already holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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
  variant="$1"
  initialization="$2"
  global_weight="$3"
  refine_name="dinov3_small_coco5k_dense_local_${variant}_e1"
  refine_dir="${ROOT}/output/lazystrike_refine/${refine_name}"
  refine_log="${ROOT}/train_${refine_name}_nohup.log"
  pretrained_args=()
  if [[ "${initialization}" == "v2" ]]; then
    pretrained_args=(--pretrained-encoder "${V2_CHECKPOINT}")
  fi

  if [[ -s "${refine_dir}/checkpoints/model_epoch1.pt" ]]; then
    log_status "SKIP refinement ${variant}: checkpoint exists"
    return
  fi

  wait_for_gpu
  log_status "START refinement ${variant} init=${initialization} global=${global_weight}"
  "${PYTHON}" -u tools/train_dinov3_lazystrike_refine.py \
    --data-root /data/cpc/root/dataset/COCO \
    --split train2017 \
    --output-dir "${refine_dir}" \
    --encoder dinov3_small \
    "${pretrained_args[@]}" \
    --resolution 640 \
    --num-images 5000 \
    --epochs 1 \
    --batch-size 2 \
    --workers 4 \
    --lr 2.5e-6 \
    --weight-decay 0.05 \
    --train-block-indexes 8 9 10 \
    --freeze-final-norm \
    --layer-lr-decay 0.8 \
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
    --dense-object-layer-weights 0.75 0.25 \
    --lambda-dense-object 0.20 \
    --lambda-dense-global "${global_weight}" \
    --device cuda \
    --seed 42 \
    --shuffle \
    --log-every 50 \
    --save-every-epoch \
    > "${refine_log}" 2>&1
  log_status "DONE refinement ${variant}"
}

run_diagnostic() {
  variant="$1"
  refine_name="dinov3_small_coco5k_dense_local_${variant}_e1"
  checkpoint="${ROOT}/output/lazystrike_refine/${refine_name}/checkpoints/model_epoch1.pt"
  diag_dir="${ROOT}/output/token_analysis/dense_local_${variant}_cocoval200"
  diag_log="${ROOT}/token_analysis_dense_local_${variant}_cocoval200_nohup.log"

  if [[ -s "${diag_dir}/metrics.json" ]]; then
    log_status "SKIP diagnostic ${variant}: metrics exist"
    return
  fi

  wait_for_gpu
  log_status "START token diagnostic ${variant}"
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
  log_status "DONE token diagnostic ${variant}"
}

record_result() {
  variant="$1"
  initialization="$2"
  global_weight="$3"
  detect_dir="$4"
  diag_dir="${ROOT}/output/token_analysis/dense_local_${variant}_cocoval200"

  if [[ ! -s "${SUMMARY_TSV}" ]]; then
    printf 'variant\tinit\tlambda_global\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\tcls_hflip_cosine\tlast_vote_hflip_consistency\tfp_gp_proxy_ratio\n' > "${SUMMARY_TSV}"
  fi

  detection=$(jq -s -r '
    map(select(.test_coco_eval_bbox != null))
    | max_by(.test_coco_eval_bbox[0])
    | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
       .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
       .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
    | @tsv
  ' "${detect_dir}/log.txt")
  diagnostic=$(jq -r '[.summary.cls_hflip_cosine, .summary.last_vote_hflip_consistency, .summary.fp_gp_proxy_ratio] | @tsv' "${diag_dir}/metrics.json")
  printf '%s\t%s\t%s\t%s\t%s\n' "${variant}" "${initialization}" "${global_weight}" "${detection}" "${diagnostic}" >> "${SUMMARY_TSV}"
}

run_detection() {
  variant="$1"
  initialization="$2"
  global_weight="$3"
  refine_name="dinov3_small_coco5k_dense_local_${variant}_e1"
  checkpoint="${ROOT}/output/lazystrike_refine/${refine_name}/checkpoints/model_epoch1.pt"
  detect_name="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_dense_local_${variant}_e1_e24"
  detect_dir="${ROOT}/output/${detect_name}"
  detect_log="${ROOT}/medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_dense_local_${variant}_e1_e24_nohup.log"

  if [[ -s "${detect_dir}/log.txt" ]] && [[ $(wc -l < "${detect_dir}/log.txt") -ge 24 ]]; then
    log_status "SKIP detection ${variant}: 24 epoch records exist"
  else
    wait_for_gpu
    log_status "START aligned medium detection ${variant}"
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
      --pretrained-encoder "${checkpoint}" \
      --output-dir "${detect_dir}" \
      > "${detect_log}" 2>&1
    log_status "DONE aligned medium detection ${variant}"
  fi
  record_result "${variant}" "${initialization}" "${global_weight}" "${detect_dir}"
}

run_variant() {
  variant="$1"
  initialization="$2"
  global_weight="$3"
  run_refine "${variant}" "${initialization}" "${global_weight}"
  run_diagnostic "${variant}"
  run_detection "${variant}" "${initialization}" "${global_weight}"
}

log_status "QUEUE START on physical GPU ${GPU_ID}"
run_variant base_g002 base 0.02
run_variant v2seq_g002 v2 0.02
run_variant v2seq_objonly v2 0.0
