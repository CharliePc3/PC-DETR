#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
UNI_INIT="/data/cpc/root/project/UniRefiner/outputs/dinov3_small_640_unirefiner_coco_balanced_zero3e/checkpoints/model_final.pt"
QUEUE_LOG="${ROOT}/unified_refinement_phase_a_gpu1_queue_nohup.log"
PID_FILE="${ROOT}/unified_refinement_phase_a_gpu1_queue.pid"
LOCK_FILE="${ROOT}/unified_refinement_phase_a_gpu1_queue.lock"
SUMMARY_DIR="${ROOT}/output/unified_refinement"
SUMMARY_TSV="${SUMMARY_DIR}/phase_a_summary.tsv"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}" /tmp/matplotlib_unified_phase_a
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}"
export MPLCONFIGDIR="/tmp/matplotlib_unified_phase_a"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another unified Phase-A queue holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
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
  objective_mode="$2"
  refine_name="dinov3_small_coco5k_unified_${variant}_e1"
  refine_dir="${ROOT}/output/unified_refinement/${refine_name}"
  refine_log="${ROOT}/train_${refine_name}_nohup.log"
  objective_args=(--objectives dense)
  if [[ "${objective_mode}" == "joint" ]]; then
    objective_args=(
      --objectives unirefiner dense
      --lambda-unirefiner 0.005
      --uni-reg-factor 37
      --uni-register-fill zero
      --uni-num-proposals 3
      --uni-disable-attention-hijack-filter
      --uni-uniformity-strength 0.3
      --uni-scd-start-stage 0.1
      --uni-scd-weight 0.4
    )
  fi

  if [[ -s "${refine_dir}/checkpoints/model_epoch1.pt" ]]; then
    log_status "SKIP refinement ${variant}: checkpoint exists"
    return
  fi
  wait_for_gpu
  log_status "START refinement ${variant} mode=${objective_mode}"
  "${PYTHON}" -u tools/train_dinov3_lazystrike_refine.py \
    --data-root /data/cpc/root/dataset/COCO \
    --split train2017 \
    --output-dir "${refine_dir}" \
    --encoder dinov3_small \
    --pretrained-encoder "${UNI_INIT}" \
    "${objective_args[@]}" \
    --resolution 640 \
    --num-images 5000 \
    --epochs 1 \
    --batch-size 2 \
    --workers 4 \
    --lr 2.5e-6 \
    --weight-decay 0.05 \
    --train-last-blocks 4 \
    --layer-lr-decay 0.8 \
    --max-boxes-per-image 12 \
    --lambda-align 0.0 \
    --lambda-cover 0.0 \
    --lambda-consistency 0.0 \
    --lambda-distill 1.0 \
    --dense-layers 8 11 \
    --dense-layer-weights 0.35 0.65 \
    --lambda-dense-object 0.15 \
    --lambda-dense-global 0.05 \
    --amp \
    --amp-dtype bf16 \
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
  refine_name="dinov3_small_coco5k_unified_${variant}_e1"
  checkpoint="${ROOT}/output/unified_refinement/${refine_name}/checkpoints/model_epoch1.pt"
  diag_dir="${ROOT}/output/token_analysis/unified_${variant}_cocoval200"
  diag_log="${ROOT}/token_analysis_unified_${variant}_cocoval200_nohup.log"

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

run_detection() {
  variant="$1"
  refine_name="dinov3_small_coco5k_unified_${variant}_e1"
  checkpoint="${ROOT}/output/unified_refinement/${refine_name}/checkpoints/model_epoch1.pt"
  detect_name="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_unified_${variant}_e1_e24"
  detect_dir="${ROOT}/output/${detect_name}"
  detect_log="${ROOT}/medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_unified_${variant}_e1_e24_nohup.log"

  if [[ -s "${detect_dir}/log.txt" ]] && [[ $(wc -l < "${detect_dir}/log.txt") -ge 24 ]]; then
    log_status "SKIP detection ${variant}: 24 epoch records exist"
    return
  fi
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
}

record_result() {
  variant="$1"
  mode="$2"
  detect_dir="${ROOT}/output/coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_unified_${variant}_e1_e24"
  diag_dir="${ROOT}/output/token_analysis/unified_${variant}_cocoval200"
  if [[ ! -s "${SUMMARY_TSV}" ]]; then
    printf 'variant\tmode\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\tcoverage100\tsmall_coverage100\tmedium_coverage100\tlarge_coverage100\tcls_hflip_cosine\tfp_gp_proxy_ratio\n' > "${SUMMARY_TSV}"
  fi
  if rg -q "^${variant}\t" "${SUMMARY_TSV}"; then
    return
  fi
  detection=$(jq -s -r '
    map(select(.test_coco_eval_bbox != null))
    | max_by(.test_coco_eval_bbox[0])
    | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
       .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
       .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]] | @tsv
  ' "${detect_dir}/log.txt")
  diagnostic=$(jq -r '[.summary.patch_score_box_coverage_top100,
    .summary.patch_score_small_box_coverage_top100,
    .summary.patch_score_medium_box_coverage_top100,
    .summary.patch_score_large_box_coverage_top100,
    .summary.cls_hflip_cosine, .summary.fp_gp_proxy_ratio] | @tsv' "${diag_dir}/metrics.json")
  printf '%s\t%s\t%s\t%s\n' "${variant}" "${mode}" "${detection}" "${diagnostic}" >> "${SUMMARY_TSV}"
}

run_variant() {
  variant="$1"
  mode="$2"
  run_refine "${variant}" "${mode}"
  run_diagnostic "${variant}"
  run_detection "${variant}"
  record_result "${variant}" "${mode}"
}

log_status "QUEUE START on physical GPU ${GPU_ID}"
run_variant "uni_balanced_to_dense_seq" "sequential"
run_variant "uni_balanced_dense_joint_u005" "joint"
