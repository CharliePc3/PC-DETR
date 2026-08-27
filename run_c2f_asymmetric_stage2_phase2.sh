#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
POLL_SECONDS="${POLL_SECONDS:-300}"
ACCEPT_DROP="${ACCEPT_DROP:-0.0015}"
SUMMARY_DIR="${ROOT}/output/c2f_asymmetric_stage2"
PHASE1_SUMMARY="${SUMMARY_DIR}/phase1_summary.tsv"
PHASE2_SUMMARY="${SUMMARY_DIR}/phase2_summary.tsv"
DECISION_FILE="${SUMMARY_DIR}/source_mode_decision.txt"
STATUS_LOG="${ROOT}/c2f_asymmetric_stage2_phase2_status.log"
LOCK_FILE="${ROOT}/c2f_asymmetric_stage2_phase2.lock"
PID_FILE="${ROOT}/c2f_asymmetric_stage2_phase2.pid"
WORKER="${ROOT}/run_c2f_asymmetric_variant.sh"
BASE_PRUNE43="${ROOT}/output/coco_medium_ssmsp_a1_prune_p3-2-5-11_p4-2-5-8-11_p5-2-8-11_g2firstfull_seed43_e24"
BASE_MASK43="${ROOT}/output/coco_medium_ssmsp_a1_mask_p3-2-5-11_p4-2-5-8-11_p5-2-8-11_g2firstfull_seed43_e24"

cd "${ROOT}"
mkdir -p "${SUMMARY_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another stage2 phase2 controller is already running\n' \
    "$(date '+%F %T')" >&2
  exit 1
fi
printf '%s\n' "$$" > "${PID_FILE}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${STATUS_LOG}"
}

on_exit() {
  local status=$?
  if [[ ${status} -eq 0 ]]; then
    log_status "PHASE2 CONTROLLER FINISHED successfully"
  else
    log_status "PHASE2 CONTROLLER FAILED with exit code ${status}"
  fi
  rm -f "${PID_FILE}"
}
trap on_exit EXIT

float_le() {
  awk -v lhs="$1" -v rhs="$2" 'BEGIN { exit !(lhs <= rhs) }'
}

best_ap_from_log() {
  jq -s -r 'map(select(.test_coco_eval_bbox != null)) | max_by(.test_coco_eval_bbox[0]) | .test_coco_eval_bbox[0]' \
    "$1/log.txt"
}

result_ap() {
  cut -f7 "${SUMMARY_DIR}/$1_seed${2:-43}.tsv"
}

acceptable() {
  local candidate_ap="$1"
  local drop
  drop=$(awk -v base="${BASE_AP}" -v candidate="${candidate_ap}" \
    'BEGIN { printf "%.9f", base - candidate }')
  float_le "${drop}" "${ACCEPT_DROP}"
}

run_worker() {
  local variant="$1"
  local gpu="$2"
  local blocks="$3"
  local seed="${4:-43}"
  env GPU_ID="${gpu}" VARIANT="${variant}" SOURCE_MODE="${SOURCE_MODE}" \
    C2F_BLOCKS="${blocks}" SEED="${seed}" "${WORKER}"
}

while [[ ! -s "${DECISION_FILE}" ]] || [[ ! -s "${PHASE1_SUMMARY}" ]] || \
      (( $(wc -l < "${PHASE1_SUMMARY}") < 3 )); do
  records=0
  [[ -s "${PHASE1_SUMMARY}" ]] && records=$(wc -l < "${PHASE1_SUMMARY}")
  log_status "WAIT PHASE1: summary lines=${records}/3"
  sleep "${POLL_SECONDS}"
done

SOURCE_MODE=$(cat "${DECISION_FILE}")
BASE43="${BASE_PRUNE43}"
if [[ "${SOURCE_MODE}" == "mask" ]]; then
  BASE43="${BASE_MASK43}"
fi
BASE_AP=$(best_ap_from_log "${BASE43}")
log_status "PHASE2 START: mode=${SOURCE_MODE}, baseline AP=${BASE_AP}"

run_worker b3_joint 0 "1 3 1" 43 &
PID_B3=$!
run_worker b4_p4 1 "3 1 3" 43 &
PID_B4=$!
wait "${PID_B3}"
wait "${PID_B4}"

declare -a REFINE_VARIANTS=()
declare -a REFINE_BLOCKS=()
if ! acceptable "$(result_ap b1_p5)"; then
  REFINE_VARIANTS+=(b1r_p5_n2)
  REFINE_BLOCKS+=("3 3 2")
fi
if ! acceptable "$(result_ap b2_p3)"; then
  REFINE_VARIANTS+=(b2r_p3_n2)
  REFINE_BLOCKS+=("2 3 3")
fi
if ! acceptable "$(result_ap b4_p4)"; then
  REFINE_VARIANTS+=(b4r_p4_n2)
  REFINE_BLOCKS+=("3 2 3")
fi

for ((i=0; i<${#REFINE_VARIANTS[@]}; i+=2)); do
  run_worker "${REFINE_VARIANTS[i]}" 0 "${REFINE_BLOCKS[i]}" 43 &
  PID_LEFT=$!
  PID_RIGHT=""
  if (( i + 1 < ${#REFINE_VARIANTS[@]} )); then
    run_worker "${REFINE_VARIANTS[i+1]}" 1 "${REFINE_BLOCKS[i+1]}" 43 &
    PID_RIGHT=$!
  fi
  wait "${PID_LEFT}"
  [[ -z "${PID_RIGHT}" ]] || wait "${PID_RIGHT}"
done

P3_DEPTH=3
if acceptable "$(result_ap b2_p3)"; then
  P3_DEPTH=1
elif [[ -s "${SUMMARY_DIR}/b2r_p3_n2_seed43.tsv" ]] && \
     acceptable "$(result_ap b2r_p3_n2)"; then
  P3_DEPTH=2
fi

P4_DEPTH=3
if acceptable "$(result_ap b4_p4)"; then
  P4_DEPTH=1
elif [[ -s "${SUMMARY_DIR}/b4r_p4_n2_seed43.tsv" ]] && \
     acceptable "$(result_ap b4r_p4_n2)"; then
  P4_DEPTH=2
fi

P5_DEPTH=3
if acceptable "$(result_ap b1_p5)"; then
  P5_DEPTH=1
elif [[ -s "${SUMMARY_DIR}/b1r_p5_n2_seed43.tsv" ]] && \
     acceptable "$(result_ap b1r_p5_n2)"; then
  P5_DEPTH=2
fi

COMPOSED_BLOCKS="${P3_DEPTH} ${P4_DEPTH} ${P5_DEPTH}"
log_status "Independent-depth selection: C2f=[${COMPOSED_BLOCKS}]"
ALIAS_VARIANT=""
case "${COMPOSED_BLOCKS}" in
  "3 3 1") ALIAS_VARIANT="b1_p5" ;;
  "1 3 3") ALIAS_VARIANT="b2_p3" ;;
  "1 3 1") ALIAS_VARIANT="b3_joint" ;;
  "3 1 3") ALIAS_VARIANT="b4_p4" ;;
  "3 3 2") ALIAS_VARIANT="b1r_p5_n2" ;;
  "2 3 3") ALIAS_VARIANT="b2r_p3_n2" ;;
  "3 2 3") ALIAS_VARIANT="b4r_p4_n2" ;;
esac
if [[ -n "${ALIAS_VARIANT}" ]]; then
  sed "s/^${ALIAS_VARIANT}/${ALIAS_VARIANT}_selected/" \
    "${SUMMARY_DIR}/${ALIAS_VARIANT}_seed43.tsv" \
    > "${SUMMARY_DIR}/b5_composed_seed43.tsv"
  log_status "REUSE ${ALIAS_VARIANT} as composed seed43 result"
else
  run_worker b5_composed 0 "${COMPOSED_BLOCKS}" 43
fi

run_worker b0_pair 0 "3 3 3" 42 &
PID_BASE42=$!
run_worker b5_composed 1 "${COMPOSED_BLOCKS}" 42 &
PID_COMPOSED42=$!
wait "${PID_BASE42}"
wait "${PID_COMPOSED42}"

BASE42_AP=$(result_ap b0_pair 42)
COMPOSED42_AP=$(result_ap b5_composed 42)
SEED43_AP=$(result_ap b5_composed 43)
DROP43=$(awk -v base="${BASE_AP}" -v candidate="${SEED43_AP}" \
  'BEGIN { printf "%.9f", base - candidate }')
DROP42=$(awk -v base="${BASE42_AP}" -v candidate="${COMPOSED42_AP}" \
  'BEGIN { printf "%.9f", base - candidate }')
printf 'blocks\tbase43\tcandidate43\tdrop43\tbase42\tcandidate42\tdrop42\n' \
  > "${SUMMARY_DIR}/stability_summary.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${COMPOSED_BLOCKS// /,}" "${BASE_AP}" "${SEED43_AP}" "${DROP43}" \
  "${BASE42_AP}" "${COMPOSED42_AP}" "${DROP42}" \
  >> "${SUMMARY_DIR}/stability_summary.tsv"
log_status "STABILITY C2f=[${COMPOSED_BLOCKS}]: drop43=${DROP43}, drop42=${DROP42}"

printf 'variant\tmode\tc2f_blocks\tseed\tparams\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\n' \
  > "${PHASE2_SUMMARY}"
for result in "${SUMMARY_DIR}"/b*_seed*.tsv; do
  cat "${result}" >> "${PHASE2_SUMMARY}"
done
log_status "PHASE2 RESULTS written to ${PHASE2_SUMMARY}"
