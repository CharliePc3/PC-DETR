#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
EXPERIMENT="backbone_refine_retention_s2_denseo2o"
SNAPSHOT_ROOT="${ROOT}/experiment_snapshots/${EXPERIMENT}"
SUMMARY_DIR="${ROOT}/output/${EXPERIMENT}"
STATUS_LOG="${ROOT}/${EXPERIMENT}_status.log"
PID_FILE="${ROOT}/${EXPERIMENT}.pid"
LOCK_FILE="${ROOT}/${EXPERIMENT}.lock"
MAX_USED_MIB="${MAX_USED_MIB:-2500}"
POLL_SECONDS="${POLL_SECONDS:-600}"

LAZYSTRIKE_CKPT="${ROOT}/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_v2_conservative/checkpoints/model_final.pt"
DENSE_CKPT="${ROOT}/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_dsdr_dense_only_e1/checkpoints/model_epoch1.pt"
LAZYSTRIKE_SHA256="d5e16ae010841a8060927f981db610be0f1918a8648c0f6de60973b7ab5d6333"
DENSE_SHA256="397fb9637e1d15e40378dad4c710e78aa460838671635c4783c6af0488664605"

GPU_LAZY="${GPU_LAZY:-0}"
GPU_DENSE="${GPU_DENSE:-1}"
SEED="${SEED:-42}"
DETECTOR_INIT_SEED="${DETECTOR_INIT_SEED:-1042}"
EPOCHS="${EPOCHS:-24}"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}" "${SNAPSHOT_ROOT}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'Another backbone-refinement retention controller is running.\n' >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"
trap 'status=$?; rm -f "${PID_FILE}"; exit ${status}' EXIT

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${STATUS_LOG}"
}

verify_checkpoint() {
  local path="$1" expected="$2"
  [[ -s "${path}" ]] || { printf 'Missing checkpoint: %s\n' "${path}" >&2; exit 2; }
  printf '%s  %s\n' "${expected}" "${path}" | sha256sum --check --status
}

wait_for_gpu() {
  local gpu="$1" variant="$2" used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
      return
    fi
    log_status "WAIT variant=${variant} gpu=${gpu} used=${used:-unknown}MiB threshold=${MAX_USED_MIB}MiB"
    sleep "${POLL_SECONDS}"
  done
}

verify_checkpoint "${LAZYSTRIKE_CKPT}" "${LAZYSTRIKE_SHA256}"
verify_checkpoint "${DENSE_CKPT}" "${DENSE_SHA256}"

if [[ ! -f "${SNAPSHOT_ROOT}/.snapshot_complete" ]]; then
  if [[ -e "${SNAPSHOT_ROOT}/src" || -e "${SNAPSHOT_ROOT}/run_coco_subset.py" ]]; then
    printf 'Incomplete snapshot exists: %s\n' "${SNAPSHOT_ROOT}" >&2
    exit 3
  fi
  cp -a "${ROOT}/src" "${SNAPSHOT_ROOT}/src"
  cp -a "${ROOT}/run_coco_subset.py" "${SNAPSHOT_ROOT}/run_coco_subset.py"
  find "${SNAPSHOT_ROOT}/src" -type f -name '*.py' -print0 \
    | sort -z | xargs -0 sha256sum > "${SNAPSHOT_ROOT}/source_sha256.txt"
  sha256sum "${SNAPSHOT_ROOT}/run_coco_subset.py" >> "${SNAPSHOT_ROOT}/source_sha256.txt"
  date '+%F %T %z' > "${SNAPSHOT_ROOT}/manifest.txt"
  touch "${SNAPSHOT_ROOT}/.snapshot_complete"
fi

COMMON_ARGS=(
  --subset medium
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST
  --epochs "${EPOCHS}"
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
  --out-feature-indexes 2 5 8 11
  --projector-scale P3 P4 P5
  --projector-p3-indexes 2 5 11
  --projector-p4-indexes 2 5 8 11
  --projector-p5-indexes 2 8 11
  --projector-source-mode prune
  --projector-c2f-blocks 3 3 3
  --projector-type multiscale
  --projector-p5-mode group2_first_full
  --projector-resample-share p5
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
  --backbone-register-border-tokens 1
  --seed "${SEED}"
  --detector-init-seed "${DETECTOR_INIT_SEED}"
)

run_variant() {
  local variant="$1" gpu="$2" checkpoint="$3" lr_scale="$4" anchor_coef="$5"
  shift 5
  local blocks=("$@")
  local output_dir="${SUMMARY_DIR}/${variant}_seed${SEED}"
  local train_log="${ROOT}/${EXPERIMENT}_${variant}_seed${SEED}_gpu${gpu}.log"
  local -a resume_args=()

  if [[ -s "${output_dir}/log.txt" ]] && (( $(wc -l < "${output_dir}/log.txt") >= EPOCHS )); then
    log_status "SKIP variant=${variant}: complete"
    return
  fi
  if [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
  elif [[ -s "${output_dir}/log.txt" ]]; then
    printf 'Partial output has no checkpoint: %s\n' "${output_dir}" >&2
    return 4
  fi

  wait_for_gpu "${gpu}" "${variant}"
  mkdir -p "${output_dir}" "/tmp/mpl_${EXPERIMENT}_gpu${gpu}" "/tmp/cache_${EXPERIMENT}_gpu${gpu}"
  log_status "START variant=${variant} gpu=${gpu} blocks=${blocks[*]} lr_scale=${lr_scale} anchor=${anchor_coef}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${SNAPSHOT_ROOT}/src" \
    DINOV3_REPO_DIR="/data/cpc/root/project/DINOv3" \
    DINOV3_WEIGHTS_DIR="${ROOT}/weights/dinov3" \
    MPLCONFIGDIR="/tmp/mpl_${EXPERIMENT}_gpu${gpu}" \
    XDG_CACHE_HOME="/tmp/cache_${EXPERIMENT}_gpu${gpu}" \
    ALBUMENTATIONS_DISABLE_VERSION_CHECK=1 \
    "${PYTHON}" -u "${SNAPSHOT_ROOT}/run_coco_subset.py" \
      "${COMMON_ARGS[@]}" \
      --pretrained-encoder "${checkpoint}" \
      --backbone-refine-blocks "${blocks[@]}" \
      --backbone-refine-lr-scale "${lr_scale}" \
      --backbone-refine-anchor-coef "${anchor_coef}" \
      --backbone-refine-anchor-stop-epoch 20 \
      "${resume_args[@]}" \
      --output-dir "${output_dir}" \
      > "${train_log}" 2>&1
  log_status "DONE variant=${variant} gpu=${gpu}"
}

summarize() {
  local summary_file="${SUMMARY_DIR}/summary.tsv" variant output_dir train_log gpu params
  printf 'variant\tparams\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\tlast5_AP\n' > "${summary_file}"
  for variant in lazy_lr01 dense_lr01 lazy_anchor1e3 dense_anchor1e3; do
    case "${variant}" in
      lazy*) gpu="${GPU_LAZY}" ;;
      dense*) gpu="${GPU_DENSE}" ;;
    esac
    output_dir="${SUMMARY_DIR}/${variant}_seed${SEED}"
    train_log="${ROOT}/${EXPERIMENT}_${variant}_seed${SEED}_gpu${gpu}.log"
    [[ -s "${output_dir}/log.txt" ]] || continue
    params=$(sed -nE 's/.*Number of trainable parameters: ([0-9]+).*/\1/p' "${train_log}" | head -n1)
    jq -s -r --arg variant "${variant}" --arg params "${params:-unknown}" '
      map(select(.test_coco_eval_bbox != null)) as $rows
      | ($rows|max_by(.test_coco_eval_bbox[0])) as $best
      | ($rows|sort_by(.epoch)|.[-5:]|map(.test_coco_eval_bbox[0])|add/length) as $last5
      | [$variant,$params,($best.epoch+1),$best.test_coco_eval_bbox[0],
         $best.test_coco_eval_bbox[1],$best.test_coco_eval_bbox[2],
         $best.test_coco_eval_bbox[3],$best.test_coco_eval_bbox[4],
         $best.test_coco_eval_bbox[5],$last5] | @tsv
    ' "${output_dir}/log.txt" >> "${summary_file}"
  done
  log_status "SUMMARY ${summary_file}"
}

log_status "CONTROLLER START seed=${SEED} detinit=${DETECTOR_INIT_SEED}"

# Phase 1: isolate target-block LR protection.
run_variant lazy_lr01 "${GPU_LAZY}" "${LAZYSTRIKE_CKPT}" 0.1 0.0 10 11 &
PID_LAZY=$!
run_variant dense_lr01 "${GPU_DENSE}" "${DENSE_CKPT}" 0.1 0.0 8 9 10 11 &
PID_DENSE=$!
wait "${PID_LAZY}"
wait "${PID_DENSE}"

# Phase 2: isolate L2-SP while retaining the original detector LR.
run_variant lazy_anchor1e3 "${GPU_LAZY}" "${LAZYSTRIKE_CKPT}" 1.0 0.001 10 11 &
PID_LAZY=$!
run_variant dense_anchor1e3 "${GPU_DENSE}" "${DENSE_CKPT}" 1.0 0.001 8 9 10 11 &
PID_DENSE=$!
wait "${PID_LAZY}"
wait "${PID_DENSE}"

summarize
log_status "CONTROLLER COMPLETE"
