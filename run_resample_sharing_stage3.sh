#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
WORKER="${ROOT}/run_resample_sharing_variant.sh"
SUMMARY_DIR="${ROOT}/output/resample_sharing_stage3"
STATUS_LOG="${ROOT}/resample_sharing_stage3_status.log"
SUMMARY_FILE="${SUMMARY_DIR}/stage3_summary.tsv"
LOCK_FILE="${ROOT}/resample_sharing_stage3.lock"
PID_FILE="${ROOT}/resample_sharing_stage3.pid"
BASE43_DIR="${ROOT}/output/coco_medium_ssmsp_a1_prune_p3-2-5-11_p4-2-5-8-11_p5-2-8-11_g2firstfull_seed43_e24"
BASE42_DIR="${ROOT}/output/coco_medium_ssmsp_a1_prune_c2f_3-3-3_b0_pair_seed42_e24"
MAX_DROP="${MAX_DROP:-0.0015}"
COMBINATION_MARGIN="${COMBINATION_MARGIN:-0.0010}"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'Another stage3 controller is already running.\n' >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" >> "${STATUS_LOG}"
}

cleanup() {
  local status=$?
  log_status "CONTROLLER EXIT status=${status}"
  rm -f "${PID_FILE}"
}
trap cleanup EXIT

best_ap_dir() {
  jq -s -r 'map(select(.test_coco_eval_bbox != null)) | max_by(.test_coco_eval_bbox[0]) | .test_coco_eval_bbox[0]' "$1/log.txt"
}

result_ap() {
  cut -f6 "$1"
}

passes_drop() {
  awk -v base="$1" -v candidate="$2" -v limit="$3" \
    'BEGIN { exit !((base - candidate) <= limit) }'
}

run_variant() {
  local gpu="$1" variant="$2" share="$3" seed="$4"
  GPU_ID="${gpu}" VARIANT="${variant}" SHARE_MODE="${share}" SEED="${seed}" \
    "${WORKER}"
}

BASE43_AP=$(best_ap_dir "${BASE43_DIR}")
BASE42_AP=$(best_ap_dir "${BASE42_DIR}")
log_status "CONTROLLER START base43=${BASE43_AP} base42=${BASE42_AP}"

run_variant 0 s1_p3 p3 43 &
PID_S1=$!
run_variant 1 s2_p5 p5 43 &
PID_S2=$!
wait "${PID_S1}"
wait "${PID_S2}"

S1_FILE="${SUMMARY_DIR}/s1_p3_seed43.tsv"
S2_FILE="${SUMMARY_DIR}/s2_p5_seed43.tsv"
S1_AP=$(result_ap "${S1_FILE}")
S2_AP=$(result_ap "${S2_FILE}")
S1_PASS=0
S2_PASS=0
passes_drop "${BASE43_AP}" "${S1_AP}" "${MAX_DROP}" && S1_PASS=1
passes_drop "${BASE43_AP}" "${S2_AP}" "${MAX_DROP}" && S2_PASS=1
log_status "SINGLE RESULTS s1=${S1_AP} pass=${S1_PASS}; s2=${S2_AP} pass=${S2_PASS}"

if (( S1_PASS == 0 && S2_PASS == 0 )); then
  log_status "STOP: neither single-factor sharing variant passed"
else
  if awk -v a="${S1_AP}" -v b="${S2_AP}" 'BEGIN { exit !(a >= b) }'; then
    BEST_VARIANT=s1_p3
    BEST_SHARE=p3
    BEST_AP="${S1_AP}"
  else
    BEST_VARIANT=s2_p5
    BEST_SHARE=p5
    BEST_AP="${S2_AP}"
  fi

  if (( S1_PASS == 1 && S2_PASS == 1 )); then
    run_variant 0 s3_combined p3_p5 43 &
    PID_S3=$!
    run_variant 1 "${BEST_VARIANT}" "${BEST_SHARE}" 42 &
    PID_BEST42=$!
    wait "${PID_S3}"
    S3_FILE="${SUMMARY_DIR}/s3_combined_seed43.tsv"
    S3_AP=$(result_ap "${S3_FILE}")
    S3_PASS=0
    if passes_drop "${BASE43_AP}" "${S3_AP}" "${MAX_DROP}" && \
       passes_drop "${BEST_AP}" "${S3_AP}" "${COMBINATION_MARGIN}"; then
      S3_PASS=1
    fi
    log_status "COMBINED RESULT s3=${S3_AP} pass=${S3_PASS}"
    if (( S3_PASS == 1 )); then
      run_variant 0 s3_combined p3_p5 42
    fi
    wait "${PID_BEST42}"
  else
    run_variant 0 "${BEST_VARIANT}" "${BEST_SHARE}" 42
  fi
fi

printf 'variant\tshare_mode\tseed\tparams\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' > "${SUMMARY_FILE}"
find "${SUMMARY_DIR}" -maxdepth 1 -name 's*_seed*.tsv' -print0 \
  | sort -z | xargs -0 -r cat >> "${SUMMARY_FILE}"
log_status "CONTROLLER FINISHED summary=${SUMMARY_FILE}"
