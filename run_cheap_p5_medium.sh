#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index.}"
P5_MODE="${P5_MODE:?Set P5_MODE, for example group2, group4, or group8.}"
SEED="${SEED:-43}"
EPOCHS="${EPOCHS:-24}"
LR_DROP="${LR_DROP:-20}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-0}"
DENSE_O2O="${DENSE_O2O:-0}"
DENSE_O2O_COPYBLEND_NUM_OBJECTS="${DENSE_O2O_COPYBLEND_NUM_OBJECTS:-1}"

case "${P5_MODE}" in
  full|pool|dwconv|group2|group2_mix|group2_mix128|group2_fullmix|group2_first_full|group2_first2_full|group2_last_full|group4|group8|fusion|fusion_wide|fusion_refine|fusion_residual)
    ;;
  *)
    printf 'Unsupported P5_MODE=%s\n' "${P5_MODE}" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-coco_medium_seed${SEED}_iterref_scaleroute_p3p4p5_${P5_MODE}_lrd${LR_DROP}_exactsteps_e${EPOCHS}}"
OUTPUT_DIR="${ROOT}/output/${RUN_NAME}"
TRAIN_LOG="${ROOT}/${RUN_NAME}_gpu${GPU_ID}.log"

mkdir -p \
  "${OUTPUT_DIR}" \
  "/tmp/mpl_rfdetr_${P5_MODE}_${SEED}" \
  "/tmp/rfdetr_${P5_MODE}_${SEED}_cache"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/mpl_rfdetr_${P5_MODE}_${SEED}"
export XDG_CACHE_HOME="/tmp/rfdetr_${P5_MODE}_${SEED}_cache"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

completed_epochs=0
if [[ -s "${OUTPUT_DIR}/log.txt" ]]; then
  completed_epochs=$(wc -l < "${OUTPUT_DIR}/log.txt")
fi
if (( completed_epochs >= EPOCHS )); then
  printf '%s already has %d/%d epoch records.\n' \
    "${RUN_NAME}" "${completed_epochs}" "${EPOCHS}"
  exit 0
fi

resume_args=()
if (( completed_epochs > 0 )) && [[ -s "${OUTPUT_DIR}/checkpoint.pth" ]]; then
  resume_args=(--resume "${OUTPUT_DIR}/checkpoint.pth")
fi

dense_o2o_args=()
if [[ "${DENSE_O2O}" == "1" ]]; then
  dense_o2o_args=(
    --use-dense-o2o
    --dense-o2o-mode enhanced
    --dense-o2o-start-epoch 2
    --dense-o2o-image-stop-epoch 12
    --dense-o2o-copyblend-stop-epoch 21
    --dense-o2o-mosaic-prob 0.5
    --dense-o2o-mixup-prob 0.5
    --dense-o2o-copyblend-prob 0.5
    --dense-o2o-copyblend-num-objects "${DENSE_O2O_COPYBLEND_NUM_OBJECTS}"
    --dense-o2o-copyblend-expand-ratios 0.1 0.25
  )
fi

cd "${ROOT}"
exec >> "${TRAIN_LOG}" 2>&1
printf '[%s] QUEUED mode=%s seed=%s gpu=%s completed=%s/%s\n' \
  "$(date '+%F %T')" \
  "${P5_MODE}" \
  "${SEED}" \
  "${GPU_ID}" \
  "${completed_epochs}" \
  "${EPOCHS}"

while true; do
  if [[ "${SKIP_GPU_WAIT}" == "1" ]]; then
    break
  fi
  used_mib=$(
    nvidia-smi \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits \
      -i "${GPU_ID}" |
      tr -d ' '
  )
  if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_USED_MIB )); then
    break
  fi
  printf '[%s] WAIT gpu=%s memory=%sMiB threshold=%sMiB\n' \
    "$(date '+%F %T')" \
    "${GPU_ID}" \
    "${used_mib:-unknown}" \
    "${MAX_USED_MIB}"
  sleep "${POLL_SECONDS}"
done

printf '[%s] START mode=%s seed=%s gpu=%s\n' \
  "$(date '+%F %T')" "${P5_MODE}" "${SEED}" "${GPU_ID}"

exec "${PYTHON}" -u run_coco_subset.py \
  --subset medium \
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
  --epochs "${EPOCHS}" \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --num-workers 2 \
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
  --p5-attention-bias 0.0 \
  --projector-scale P3 P4 P5 \
  --projector-type multiscale \
  --projector-p5-mode "${P5_MODE}" \
  --multi-scale \
  --expanded-scales \
  --aug-preset default \
  --lr 0.0001 \
  --lr-encoder 0.00015 \
  --lr-drop "${LR_DROP}" \
  --weight-decay 0.0001 \
  --lr-vit-layer-decay 0.8 \
  --lr-component-decay 0.7 \
  --use-cdn \
  --dn-number 50 \
  --dn-box-noise-scale 0.6 \
  --dn-label-noise-scale 0.5 \
  --dn-loss-coef 0.5 \
  "${dense_o2o_args[@]}" \
  --backbone-register-border-tokens 1 \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "${resume_args[@]}"
