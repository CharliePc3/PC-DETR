#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
EXPERIMENT="integrated_refinement_s2_denseo2o"
SNAPSHOT_ROOT="${ROOT}/experiment_snapshots/${EXPERIMENT}"
SUMMARY_DIR="${ROOT}/output/${EXPERIMENT}"
STATUS_LOG="${ROOT}/${EXPERIMENT}_status.log"
PID_FILE="${ROOT}/${EXPERIMENT}.pid"
LOCK_FILE="${ROOT}/${EXPERIMENT}.lock"

LAZYSTRIKE_CKPT="${ROOT}/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_v2_conservative/checkpoints/model_final.pt"
DENSE_CKPT="${ROOT}/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_dsdr_dense_only_e1/checkpoints/model_epoch1.pt"
LAZYSTRIKE_SHA256="d5e16ae010841a8060927f981db610be0f1918a8648c0f6de60973b7ab5d6333"
DENSE_SHA256="397fb9637e1d15e40378dad4c710e78aa460838671635c4783c6af0488664605"

GPU_OFFICIAL="${GPU_OFFICIAL:-0}"
GPU_LAZYSTRIKE="${GPU_LAZYSTRIKE:-1}"
GPU_DENSE="${GPU_DENSE:-3}"
SEED="${SEED:-42}"
DETECTOR_INIT_SEED="${DETECTOR_INIT_SEED:-$((1000 + SEED))}"
EPOCHS="${EPOCHS:-24}"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}" "${SNAPSHOT_ROOT}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'Another integrated-refinement controller is already running.\n' >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"
trap 'status=$?; rm -f "${PID_FILE}"; exit ${status}' EXIT

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${STATUS_LOG}"
}

verify_checkpoint() {
  local path="$1"
  local expected="$2"
  if [[ ! -s "${path}" ]]; then
    printf 'Missing refinement checkpoint: %s\n' "${path}" >&2
    exit 2
  fi
  printf '%s  %s\n' "${expected}" "${path}" | sha256sum --check --status
}

verify_checkpoint "${LAZYSTRIKE_CKPT}" "${LAZYSTRIKE_SHA256}"
verify_checkpoint "${DENSE_CKPT}" "${DENSE_SHA256}"

# Freeze the dirty working tree once so every cell uses byte-identical model code.
if [[ ! -f "${SNAPSHOT_ROOT}/.snapshot_complete" ]]; then
  if [[ -e "${SNAPSHOT_ROOT}/src" || -e "${SNAPSHOT_ROOT}/run_coco_subset.py" ]]; then
    printf 'Incomplete snapshot exists: %s\n' "${SNAPSHOT_ROOT}" >&2
    exit 3
  fi
  cp -a "${ROOT}/src" "${SNAPSHOT_ROOT}/src"
  cp -a "${ROOT}/run_coco_subset.py" "${SNAPSHOT_ROOT}/run_coco_subset.py"
  find "${SNAPSHOT_ROOT}/src" -type f -name '*.py' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${SNAPSHOT_ROOT}/source_sha256.txt"
  sha256sum "${SNAPSHOT_ROOT}/run_coco_subset.py" >> "${SNAPSHOT_ROOT}/source_sha256.txt"
  {
    printf 'created=%s\n' "$(date '+%F %T %z')"
    printf 'lazystrike_sha256=%s\n' "${LAZYSTRIKE_SHA256}"
    printf 'dense_sha256=%s\n' "${DENSE_SHA256}"
  } > "${SNAPSHOT_ROOT}/manifest.txt"
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

run_cell() {
  local variant="$1"
  local gpu="$2"
  local checkpoint="$3"
  local output_dir="${SUMMARY_DIR}/${variant}_seed${SEED}"
  local train_log="${ROOT}/${EXPERIMENT}_${variant}_seed${SEED}_gpu${gpu}.log"
  local -a encoder_args=()
  local -a resume_args=()

  if [[ -n "${checkpoint}" ]]; then
    encoder_args=(--pretrained-encoder "${checkpoint}")
  fi
  if [[ -s "${output_dir}/log.txt" ]] \
      && (( $(wc -l < "${output_dir}/log.txt") >= EPOCHS )); then
    log_status "SKIP variant=${variant} seed=${SEED}: complete"
    return
  fi
  if [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
    log_status "RESUME variant=${variant} seed=${SEED} gpu=${gpu}"
  elif [[ -s "${output_dir}/log.txt" ]]; then
    printf 'Partial output has no checkpoint: %s\n' "${output_dir}" >&2
    return 4
  fi

  mkdir -p "${output_dir}" "/tmp/mpl_${EXPERIMENT}_gpu${gpu}" "/tmp/cache_${EXPERIMENT}_gpu${gpu}"
  log_status "START variant=${variant} seed=${SEED} detinit=${DETECTOR_INIT_SEED} gpu=${gpu}"
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
      "${encoder_args[@]}" \
      "${resume_args[@]}" \
      --output-dir "${output_dir}" \
      > "${train_log}" 2>&1
  log_status "DONE variant=${variant} seed=${SEED} gpu=${gpu}"
}

summarize() {
  local summary_file="${SUMMARY_DIR}/summary_seed${SEED}.tsv"
  printf 'variant\tseed\tdetector_init_seed\tparams\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\tlast5_AP\n' > "${summary_file}"
  local variant output_dir train_log gpu params
  for variant in official lazystrike dense_only; do
    case "${variant}" in
      official) gpu="${GPU_OFFICIAL}" ;;
      lazystrike) gpu="${GPU_LAZYSTRIKE}" ;;
      dense_only) gpu="${GPU_DENSE}" ;;
    esac
    output_dir="${SUMMARY_DIR}/${variant}_seed${SEED}"
    train_log="${ROOT}/${EXPERIMENT}_${variant}_seed${SEED}_gpu${gpu}.log"
    if [[ ! -s "${output_dir}/log.txt" ]]; then
      continue
    fi
    params=$(sed -nE 's/.*Number of trainable parameters: ([0-9]+).*/\1/p' "${train_log}" | head -n 1)
    jq -s -r \
      --arg variant "${variant}" \
      --arg seed "${SEED}" \
      --arg detinit "${DETECTOR_INIT_SEED}" \
      --arg params "${params:-unknown}" '
        map(select(.test_coco_eval_bbox != null)) as $rows
        | ($rows | max_by(.test_coco_eval_bbox[0])) as $best
        | ($rows | sort_by(.epoch) | .[-5:] | map(.test_coco_eval_bbox[0]) | add / length) as $last5
        | [$variant, $seed, $detinit, $params, ($best.epoch + 1),
           $best.test_coco_eval_bbox[0], $best.test_coco_eval_bbox[1],
           $best.test_coco_eval_bbox[2], $best.test_coco_eval_bbox[3],
           $best.test_coco_eval_bbox[4], $best.test_coco_eval_bbox[5], $last5]
        | @tsv
      ' "${output_dir}/log.txt" >> "${summary_file}"
  done
  log_status "SUMMARY seed=${SEED}: ${summary_file}"
}

log_status "CONTROLLER START seed=${SEED} detinit=${DETECTOR_INIT_SEED} snapshot=${SNAPSHOT_ROOT}"
run_cell official "${GPU_OFFICIAL}" "" &
PID_OFFICIAL=$!
run_cell lazystrike "${GPU_LAZYSTRIKE}" "${LAZYSTRIKE_CKPT}" &
PID_LAZYSTRIKE=$!
run_cell dense_only "${GPU_DENSE}" "${DENSE_CKPT}" &
PID_DENSE=$!

status=0
wait "${PID_OFFICIAL}" || status=1
wait "${PID_LAZYSTRIKE}" || status=1
wait "${PID_DENSE}" || status=1
if (( status != 0 )); then
  log_status "CONTROLLER FAILED seed=${SEED}"
  exit "${status}"
fi

summarize
log_status "CONTROLLER COMPLETE seed=${SEED}"
