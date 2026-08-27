#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
GPU_ID="${GPU_ID:?Set GPU_ID to the physical GPU index.}"
VARIANT="${VARIANT:?Set a supported SDSR variant (v10_aa through v36_packedfullgroup).}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-6}"
LR_DROP="${LR_DROP:-20}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
DETECTOR_INIT_SEED="${DETECTOR_INIT_SEED:-$((SEED + 1000))}"
CROSS_SCALE_MODE="${CROSS_SCALE_MODE:-none}"
CROSS_SCALE_RANK="${CROSS_SCALE_RANK:-32}"
SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-0}"
EXTERNAL_RESUME="${EXTERNAL_RESUME:-}"

case "${VARIANT}" in
  v10_aa)
    PROJECTOR_TYPE="sdsr_v10"
    VARIANT_TAG="v10_exactp3p4_centeredaa_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v10_noaa)
    PROJECTOR_TYPE="sdsr_v10"
    VARIANT_TAG="v10_exactp3p4_learned_p5"
    PHASE_FLAG=(--no-sdsr-use-phase-downsample)
    ;;
  v11_basis2)
    PROJECTOR_TYPE="sdsr_v11"
    VARIANT_TAG="v11_exactp3p4_twobasis_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v11_basis2_noaa)
    PROJECTOR_TYPE="sdsr_v11"
    VARIANT_TAG="v11_exactp3p4_twobasis_learned_p5"
    PHASE_FLAG=(--no-sdsr-use-phase-downsample)
    ;;
  v12_basis4)
    PROJECTOR_TYPE="sdsr_v12"
    VARIANT_TAG="v12_exactp3p4_fourbasis_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v13_highpass)
    PROJECTOR_TYPE="sdsr_v13"
    VARIANT_TAG="v13_exactp3p4_highpass_twobasis_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v14_layeradaptive)
    PROJECTOR_TYPE="sdsr_v14"
    VARIANT_TAG="v14_exactp3p4_firstfull_later_twobasis_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v15_bottomup)
    PROJECTOR_TYPE="sdsr_v15"
    VARIANT_TAG="v15_exactp3p4_twobasis_p5_zeroinit_bottomup"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v16_p4p5pool)
    PROJECTOR_TYPE="sdsr_v16"
    VARIANT_TAG="v16_exactp3p4_twobasis_p5_zeroinit_p4p5pool"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v17_fastp4p5pool)
    PROJECTOR_TYPE="sdsr_v17"
    VARIANT_TAG="v17_exactp3p4_twobasis_p5_fast_zeroinit_p4p5pool"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v18_adaptive_lowpass)
    PROJECTOR_TYPE="sdsr_v18"
    VARIANT_TAG="v18_exactp3p4_adaptive_lowpass_twobasis_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v19_deepp5)
    PROJECTOR_TYPE="sdsr_v19"
    VARIANT_TAG="v19_exactp3p4_twobasis_deepfusion_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v20_layeradapter)
    PROJECTOR_TYPE="sdsr_v20"
    VARIANT_TAG="v20_exactp3p4_twobasis_layeradapter64_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v21_spatialsemantic)
    PROJECTOR_TYPE="sdsr_v21"
    VARIANT_TAG="v21_exactp3p4_twobasis_spatialsemantic64_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v22_groupdetail)
    PROJECTOR_TYPE="sdsr_v22"
    VARIANT_TAG="v22_exactp3p4_twobasis_groupdetail16_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v23_deepgroupdetail)
    PROJECTOR_TYPE="sdsr_v23"
    VARIANT_TAG="v23_exactp3p4_twobasis_deepgroupdetail16_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v24_earlygroupdetail)
    PROJECTOR_TYPE="sdsr_v24"
    VARIANT_TAG="v24_exactp3p4_twobasis_earlygroupdetail4_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v25_earlyfulldetail)
    PROJECTOR_TYPE="sdsr_v25"
    VARIANT_TAG="v25_exactp3p4_twobasis_earlyfulldetail_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v26_endpointdetail)
    PROJECTOR_TYPE="sdsr_v26"
    VARIANT_TAG="v26_exactp3p4_twobasis_endpointdetail4x16_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v27_deepgroup8)
    PROJECTOR_TYPE="sdsr_v27"
    VARIANT_TAG="v27_exactp3p4_twobasis_deepgroupdetail8_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v28_deepgroup32)
    PROJECTOR_TYPE="sdsr_v28"
    VARIANT_TAG="v28_exactp3p4_twobasis_deepgroupdetail32_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v29_residualbasis)
    PROJECTOR_TYPE="sdsr_v29"
    VARIANT_TAG="v29_exactp3p4_twobasis_deepsemanticresidualbasis16_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v30_p3detail)
    PROJECTOR_TYPE="sdsr_v30"
    VARIANT_TAG="v30_exactp3p4_twobasis_deepgroupdetail16_p5_semanticgatedimage_p3"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v31_complementaryp5)
    PROJECTOR_TYPE="sdsr_v31"
    VARIANT_TAG="v31_exactp3p4_twobasis_deepgroupdetail16_semanticresidualbasis16_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v32_phasecalibrated)
    PROJECTOR_TYPE="sdsr_v32"
    VARIANT_TAG="v32_exactp3p4_twobasis_deepgroupdetail16_phasecalibrated_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v33_gatedgroupdetail)
    PROJECTOR_TYPE="sdsr_v33"
    VARIANT_TAG="v33_exactp3p4_twobasis_deepgatedoffdiagonalgroupdetail16_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v34_packedseparable)
    PROJECTOR_TYPE="sdsr_v34"
    VARIANT_TAG="v34_exactp3p4_packedtwobasis_deepseparablegroupdetail16_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v35_fusedkernel)
    PROJECTOR_TYPE="sdsr_v35"
    VARIANT_TAG="v35_exactp3p4_fusedkerneltwobasis_deepseparablegroupdetail16_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  v36_packedfullgroup)
    PROJECTOR_TYPE="sdsr_v36"
    VARIANT_TAG="v36_exactp3p4_packedtwobasis_deepfullgroupdetail16_p5"
    PHASE_FLAG=(--sdsr-use-phase-downsample)
    ;;
  *)
    printf 'Unsupported VARIANT=%s\n' "${VARIANT}" >&2
    exit 2
    ;;
esac

XSCALE_TAG=""
if [[ "${CROSS_SCALE_MODE}" != "none" ]]; then
  XSCALE_TAG="_xscale${CROSS_SCALE_MODE}${CROSS_SCALE_RANK}"
fi
RUN_NAME="${RUN_NAME:-coco_medium_seed${SEED}_detinit${DETECTOR_INIT_SEED}_iterref_scaleroute_${VARIANT_TAG}${XSCALE_TAG}_lrd${LR_DROP}_e24}"
OUTPUT_DIR="${ROOT}/output/${RUN_NAME}"
TRAIN_LOG="${ROOT}/${RUN_NAME}_gpu${GPU_ID}.log"
PID_FILE="${ROOT}/${RUN_NAME}_gpu${GPU_ID}.pid"
LOCK_FILE="${ROOT}/${RUN_NAME}_gpu${GPU_ID}.lock"

mkdir -p \
  "${OUTPUT_DIR}" \
  "/tmp/mpl_${VARIANT}_${SEED}_${GPU_ID}" \
  "/tmp/cache_${VARIANT}_${SEED}_${GPU_ID}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="/tmp/mpl_${VARIANT}_${SEED}_${GPU_ID}"
export XDG_CACHE_HOME="/tmp/cache_${VARIANT}_${SEED}_${GPU_ID}"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another process holds %s\n' "$(date '+%F %T')" "${LOCK_FILE}" >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"
trap 'status=$?; rm -f "${PID_FILE}"; exit ${status}' EXIT

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
if [[ -n "${EXTERNAL_RESUME}" ]]; then
  if [[ ! -s "${EXTERNAL_RESUME}" ]]; then
    printf 'External checkpoint does not exist: %s\n' "${EXTERNAL_RESUME}" >&2
    exit 2
  fi
  if (( completed_epochs > 0 )); then
    printf 'EXTERNAL_RESUME requires an empty output log: %s\n' "${OUTPUT_DIR}" >&2
    exit 2
  fi
  resume_args=(--resume "${EXTERNAL_RESUME}")
elif (( completed_epochs > 0 )) && [[ -s "${OUTPUT_DIR}/checkpoint.pth" ]]; then
  resume_args=(--resume "${OUTPUT_DIR}/checkpoint.pth")
fi

cd "${ROOT}"
exec >> "${TRAIN_LOG}" 2>&1
printf '[%s] QUEUED variant=%s seed=%s detinit=%s gpu=%s completed=%s/%s\n' \
  "$(date '+%F %T')" "${VARIANT}" "${SEED}" "${DETECTOR_INIT_SEED}" \
  "${GPU_ID}" "${completed_epochs}" "${EPOCHS}"

while true; do
  if [[ "${SKIP_GPU_WAIT}" == "1" ]]; then
    break
  fi
  used_mib=$(
    nvidia-smi \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader |
      sed -n "$((GPU_ID + 1))p" |
      awk -F ',' '{gsub(/[^0-9]/, "", $3); print $3}'
  )
  if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_USED_MIB )); then
    break
  fi
  printf '[%s] WAIT gpu=%s memory=%sMiB threshold=%sMiB\n' \
    "$(date '+%F %T')" "${GPU_ID}" "${used_mib:-unknown}" "${MAX_USED_MIB}"
  sleep "${POLL_SECONDS}"
done

printf '[%s] START variant=%s seed=%s detinit=%s gpu=%s target_epochs=%s\n' \
  "$(date '+%F %T')" "${VARIANT}" "${SEED}" "${DETECTOR_INIT_SEED}" \
  "${GPU_ID}" "${EPOCHS}"

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
  --projector-type "${PROJECTOR_TYPE}" \
  --sdsr-rank-channels 64 \
  --sdsr-cross-scale-mode "${CROSS_SCALE_MODE}" \
  --sdsr-cross-scale-rank "${CROSS_SCALE_RANK}" \
  --no-sdsr-use-local-reassembly \
  --no-sdsr-use-directional-guide \
  "${PHASE_FLAG[@]}" \
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
  --backbone-register-border-tokens 1 \
  --seed "${SEED}" \
  --detector-init-seed "${DETECTOR_INIT_SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "${resume_args[@]}"
