#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/cpc/root/project/RF-DETR-DINOv3}"
PYTHON="${PYTHON:-/home/cpc/.conda/envs/rfdetr-dinov3/bin/python}"
EXPERIMENT="online_backbone_refinement_stage1_v2"
OUTPUT_ROOT="${ROOT}/output/${EXPERIMENT}"
SNAPSHOT_ROOT="${ROOT}/experiment_snapshots/${EXPERIMENT}"
STATUS_LOG="${ROOT}/${EXPERIMENT}_status.log"
DENSE_CKPT="${ROOT}/output/lazystrike_refine/dinov3_small_coco5k_lazystrike_dsdr_dense_only_e1/checkpoints/model_epoch1.pt"

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${SNAPSHOT_ROOT}"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1" | tee -a "${STATUS_LOG}"
}

if [[ ! -f "${SNAPSHOT_ROOT}/.snapshot_complete" ]]; then
  cp -a "${ROOT}/src" "${SNAPSHOT_ROOT}/src"
  cp -a "${ROOT}/run_coco_subset.py" "${SNAPSHOT_ROOT}/run_coco_subset.py"
  touch "${SNAPSHOT_ROOT}/.snapshot_complete"
fi

COMMON_ARGS=(
  --subset medium
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST
  --epochs 24
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
  --seed 42
  --detector-init-seed 1042
)

run_variant() {
  local variant="$1"
  local gpu="$2"
  shift 2
  local output_dir="${OUTPUT_ROOT}/${variant}_seed42"
  local log_file="${ROOT}/${EXPERIMENT}_${variant}_gpu${gpu}.log"
  local -a resume_args=()

  if [[ -s "${output_dir}/log.txt" ]] && (( $(wc -l < "${output_dir}/log.txt") >= 24 )); then
    log_status "SKIP ${variant}: complete"
    return
  fi
  if [[ -s "${output_dir}/checkpoint.pth" ]]; then
    resume_args=(--resume "${output_dir}/checkpoint.pth")
  fi

  mkdir -p "${output_dir}" "/tmp/mpl_${EXPERIMENT}_${gpu}" "/tmp/cache_${EXPERIMENT}_${gpu}"
  log_status "START ${variant} gpu=${gpu}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${SNAPSHOT_ROOT}/src" \
    DINOV3_REPO_DIR="/data/cpc/root/project/DINOv3" \
    DINOV3_WEIGHTS_DIR="${ROOT}/weights/dinov3" \
    MPLCONFIGDIR="/tmp/mpl_${EXPERIMENT}_${gpu}" \
    XDG_CACHE_HOME="/tmp/cache_${EXPERIMENT}_${gpu}" \
    ALBUMENTATIONS_DISABLE_VERSION_CHECK=1 \
    "${PYTHON}" -u "${SNAPSHOT_ROOT}/run_coco_subset.py" \
      "${COMMON_ARGS[@]}" \
      "$@" \
      "${resume_args[@]}" \
      --output-dir "${output_dir}" \
      > "${log_file}" 2>&1
  log_status "DONE ${variant} gpu=${gpu}"
}

summarize() {
  local summary="${OUTPUT_ROOT}/summary.tsv"
  printf 'variant\tbest_epoch\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\tlast5_AP\n' > "${summary}"
  local variant output_dir
  for variant in official_object_e0_6_c0p1 dense_teacher_e0_6_c1p0; do
    output_dir="${OUTPUT_ROOT}/${variant}_seed42"
    jq -s -r --arg variant "${variant}" '
      map(select(.test_coco_eval_bbox != null)) as $rows
      | ($rows | max_by(.test_coco_eval_bbox[0])) as $best
      | ($rows | sort_by(.epoch) | .[-5:] | map(.test_coco_eval_bbox[0]) | add / length) as $last5
      | [$variant, ($best.epoch + 1), $best.test_coco_eval_bbox[0],
         $best.test_coco_eval_bbox[1], $best.test_coco_eval_bbox[2],
         $best.test_coco_eval_bbox[3], $best.test_coco_eval_bbox[4],
         $best.test_coco_eval_bbox[5], $last5] | @tsv
    ' "${output_dir}/log.txt" >> "${summary}"
  done
  log_status "SUMMARY ${summary}"
}

run_variant official_object_e0_6_c0p1 0 \
  --online-refine-mode object_token \
  --online-refine-start-epoch 0 \
  --online-refine-stop-epoch 6 \
  --online-refine-coef 0.1 &
PID0=$!

run_variant dense_teacher_e0_6_c1p0 1 \
  --pretrained-encoder "${DENSE_CKPT}" \
  --online-refine-mode feature_teacher \
  --online-refine-start-epoch 0 \
  --online-refine-stop-epoch 6 \
  --online-refine-coef 1.0 \
  --online-refine-background-weight 0.1 &
PID1=$!

status=0
wait "${PID0}" || status=1
wait "${PID1}" || status=1
if (( status != 0 )); then
  log_status "CONTROLLER FAILED"
  exit "${status}"
fi
summarize
log_status "CONTROLLER COMPLETE"
