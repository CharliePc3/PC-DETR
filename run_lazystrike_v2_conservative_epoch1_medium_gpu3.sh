#!/usr/bin/env zsh
set -euo pipefail

PROJECT=/data/cpc/root/project/RF-DETR-DINOv3
PYTHON=/home/cpc/.conda/envs/rfdetr-dinov3/bin/python
GPU=3
MIN_FREE_MIB=15000

REFINE_NAME=dinov3_small_coco5k_lazystrike_v2_conservative_epoch1_repro
REFINE_DIR="$PROJECT/output/lazystrike_refine/$REFINE_NAME"
REFINE_LOG="$PROJECT/train_${REFINE_NAME}_nohup.log"
CHECKPOINT="$REFINE_DIR/checkpoints/model_final.pt"

DETECT_NAME=coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_lazystrike_v2_conservative_epoch1_e24
DETECT_DIR="$PROJECT/output/$DETECT_NAME"
DETECT_LOG="$PROJECT/medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_lazystrike_v2_conservative_epoch1_e24_nohup.log"

wait_for_memory() {
  local stage="$1"
  local free_mib
  while true; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( free_mib >= MIN_FREE_MIB )); then
      print "[$(date '+%F %T')] GPU $GPU has ${free_mib} MiB free; starting $stage"
      return
    fi
    print "[$(date '+%F %T')] GPU $GPU has ${free_mib} MiB free; waiting for ${MIN_FREE_MIB} MiB before $stage"
    sleep 30
  done
}

cd "$PROJECT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PROJECT"
export MPLCONFIGDIR=/tmp/matplotlib_lazystrike_v2_epoch1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

wait_for_memory "one-epoch refinement"
"$PYTHON" tools/train_dinov3_lazystrike_refine.py \
  --data-root /data/cpc/root/dataset/COCO \
  --split train2017 \
  --output-dir "$REFINE_DIR" \
  --encoder dinov3_small \
  --resolution 640 \
  --num-images 5000 \
  --epochs 1 \
  --batch-size 2 \
  --workers 4 \
  --lr 5e-6 \
  --weight-decay 0.05 \
  --train-last-blocks 2 \
  --max-boxes-per-image 12 \
  --min-box-patches 1 \
  --lazy-topk 1 \
  --lazy-target-weight 0.1 \
  --cover-margin 0.1 \
  --cover-temperature 0.1 \
  --lambda-align 0.25 \
  --lambda-cover 0.15 \
  --lambda-consistency 0.25 \
  --lambda-distill 1.0 \
  --device cuda \
  --seed 42 \
  --shuffle \
  --log-every 50 \
  --save-every-epoch \
  > "$REFINE_LOG" 2>&1

test -s "$CHECKPOINT"
wait_for_memory "aligned medium detection"
"$PYTHON" run_coco_subset.py \
  --subset medium \
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
  --output-dir "$DETECT_DIR" \
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
  --pretrained-encoder "$CHECKPOINT" \
  > "$DETECT_LOG" 2>&1

print "[$(date '+%F %T')] Completed: $DETECT_DIR"
