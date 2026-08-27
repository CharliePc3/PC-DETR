#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/cpc/root/project/RF-DETR-DINOv3"
PYTHON="/home/cpc/.conda/envs/rfdetr-dinov3/bin/python"
GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index}"
VARIANT="${VARIANT:?Set VARIANT to aa, noaa, or aa_bidir}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TARGET_EPOCHS="${TARGET_EPOCHS:-6}"
DETECTOR_INIT_SEED="${DETECTOR_INIT_SEED:-1042}"

case "${VARIANT}" in
  aa)
    OUTPUT_TAG="exactp3p4_aap5_stableinit${DETECTOR_INIT_SEED}"
    PROJECTOR_ARGS=(
      --projector-type sdsr_v10
      --sdsr-rank-channels 64
      --sdsr-cross-scale-mode none
      --sdsr-cross-scale-rank 32
      --no-sdsr-use-local-reassembly
      --no-sdsr-use-directional-guide
      --sdsr-use-phase-downsample
    )
    ;;
  noaa)
    OUTPUT_TAG="exactp3p4_learnedp5_stableinit${DETECTOR_INIT_SEED}"
    PROJECTOR_ARGS=(
      --projector-type sdsr_v10
      --sdsr-rank-channels 64
      --sdsr-cross-scale-mode none
      --sdsr-cross-scale-rank 32
      --no-sdsr-use-local-reassembly
      --no-sdsr-use-directional-guide
      --no-sdsr-use-phase-downsample
    )
    ;;
  aa_bidir)
    OUTPUT_TAG="exactp3p4_aap5_bidir32_stableinit${DETECTOR_INIT_SEED}"
    PROJECTOR_ARGS=(
      --projector-type sdsr_v10
      --sdsr-rank-channels 64
      --sdsr-cross-scale-mode bidirectional
      --sdsr-cross-scale-rank 32
      --no-sdsr-use-local-reassembly
      --no-sdsr-use-directional-guide
      --sdsr-use-phase-downsample
    )
    ;;
  *)
    printf 'Unsupported VARIANT=%s\n' "${VARIANT}" >&2
    exit 2
    ;;
esac

OUTPUT_NAME="coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_sdsrv10_${OUTPUT_TAG}_seed42_e24"
OUTPUT_DIR="${ROOT}/output/${OUTPUT_NAME}"
TRAIN_LOG="${ROOT}/${OUTPUT_NAME}_nohup.log"
RUN_LOG="${ROOT}/sdsr_v10_${VARIANT}_gpu${GPU_ID}_runner_nohup.log"
PID_FILE="${ROOT}/sdsr_v10_${VARIANT}_gpu${GPU_ID}.pid"
LOCK_FILE="${ROOT}/sdsr_v10_${VARIANT}_gpu${GPU_ID}.lock"
RESULT_DIR="${ROOT}/output/sdsr_v10_medium_ablation"
RESULT_TSV="${RESULT_DIR}/${VARIANT}_seed42.tsv"

cd "${ROOT}"
mkdir -p "${RESULT_DIR}" "/tmp/matplotlib_sdsr_v10_${VARIANT}_${GPU_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/matplotlib_sdsr_v10_${VARIANT}_${GPU_ID}"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another process holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${RUN_LOG}"
}

on_exit() {
  local exit_status=$?
  trap - EXIT
  if [[ ${exit_status} -eq 0 ]]; then
    log_status "RUN FINISHED successfully"
  else
    log_status "RUN FAILED with exit code ${exit_status}"
  fi
  rm -f "${PID_FILE}"
  exit "${exit_status}"
}
trap on_exit EXIT

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
  if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
    break
  fi
  log_status "WAIT GPU ${GPU_ID}: ${used:-unknown} MiB used"
  sleep "${POLL_SECONDS}"
done

RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/checkpoint.pth" ]] && [[ $(wc -l < "${OUTPUT_DIR}/log.txt" 2>/dev/null || echo 0) -lt "${TARGET_EPOCHS}" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/checkpoint.pth")
  log_status "RESUME ${VARIANT} from ${OUTPUT_DIR}/checkpoint.pth"
fi

if [[ -s "${OUTPUT_DIR}/log.txt" ]] && [[ $(wc -l < "${OUTPUT_DIR}/log.txt") -ge "${TARGET_EPOCHS}" ]]; then
  log_status "SKIP ${VARIANT}: ${TARGET_EPOCHS} epoch records already exist"
else
  log_status "START ${VARIANT} to epoch ${TARGET_EPOCHS} on physical GPU ${GPU_ID}, output=${OUTPUT_NAME}"
  START_SECONDS=$(date +%s)
  "${PYTHON}" -u run_coco_subset.py \
    --subset medium \
    --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
    --epochs "${TARGET_EPOCHS}" \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --num-workers 2 \
    --device cuda \
    --world-size 1 \
    --seed 42 \
    --detector-init-seed "${DETECTOR_INIT_SEED}" \
    --lr 1e-4 \
    --lr-encoder 1.5e-4 \
    --lr-drop 100 \
    --warmup-epochs 0.0 \
    --weight-decay 1e-4 \
    --lr-vit-layer-decay 0.8 \
    --lr-component-decay 0.7 \
    --group-detr 6 \
    --projector-scale P3 P4 P5 \
    --out-feature-indexes 2 5 8 11 \
    --multi-scale \
    --expanded-scales \
    --resolution 640 \
    --dec-layers 4 \
    --dec-n-points 2 \
    --lite-refpoint-refine \
    --query-init learned \
    --num-queries 300 \
    --num-select 300 \
    --use-cdn \
    --dn-number 50 \
    --dn-label-noise-scale 0.5 \
    --dn-box-noise-scale 0.6 \
    --dn-loss-coef 0.5 \
    --eval-max-dets 100 \
    --aug-preset default \
    --output-dir "${OUTPUT_DIR}" \
    "${PROJECTOR_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    > "${TRAIN_LOG}" 2>&1
  RUNTIME_SECONDS=$(($(date +%s) - START_SECONDS))
  log_status "DONE ${VARIANT}, runtime=${RUNTIME_SECONDS}s"
fi

RESULT=$(jq -s -r '
  map(select(.test_coco_eval_bbox != null))
  | max_by(.test_coco_eval_bbox[0])
  | [(.epoch + 1), .test_coco_eval_bbox[0], .test_coco_eval_bbox[1],
     .test_coco_eval_bbox[2], .test_coco_eval_bbox[3],
     .test_coco_eval_bbox[4], .test_coco_eval_bbox[5]]
  | @tsv
' "${OUTPUT_DIR}/log.txt")
PARAMETERS=$(rg -o 'Number of trainable parameters: [0-9]+' "${TRAIN_LOG}" | tail -1 | awk '{print $5}')
MAX_MEMORY=$(rg -o 'max mem: [0-9]+' "${TRAIN_LOG}" | awk 'BEGIN {max=0} {if ($3>max) max=$3} END {print max}')
printf 'variant\tseed\tdetector_init_seed\ttrainable_parameters\tmax_memory_mib\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' > "${RESULT_TSV}"
printf '%s\t42\t%s\t%s\t%s\t%s\n' \
  "${VARIANT}" "${DETECTOR_INIT_SEED}" "${PARAMETERS:-unknown}" "${MAX_MEMORY:-unknown}" "${RESULT}" >> "${RESULT_TSV}"
