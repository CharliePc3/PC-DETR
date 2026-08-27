# Medium AP 36 Milestone

Date: 2026-07-30

## Result

The first medium run above the target reached **36.1423 AP** at epoch 23:

| Run | AP | AP50 | AP75 | APS | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Iterative refine, seed 43, epoch 22 | 35.3489 | 52.4520 | 37.5795 | 17.5589 | 37.9293 | 53.3392 |
| Iterative refine + scale routing, seed 43, epoch 22 | 35.4019 | 52.7244 | 36.9872 | 16.3554 | 38.2413 | 55.0138 |
| Iterative refine, one low-LR finish epoch | 35.8974 | 53.0187 | 38.0854 | 17.7872 | 38.0043 | 55.2076 |
| **Iterative refine + scale routing, one low-LR finish epoch** | **36.1423** | **53.6204** | **38.4940** | **17.2933** | **38.8233** | **55.4688** |

The winning checkpoint is:

```text
output/coco_medium_iterref_scaleroute_seed43_best22_lrd22_finish_e28_regb1/checkpoint_best_regular.pth
```

Checkpoint SHA-256:

```text
e62e9dd8dc21ccb9b5bbf289442ff3ea5ce992ddea84acdf3307daed3d95c33f
```

An independent evaluation with the current source reproduced the rounded COCO
metrics `AP=0.361`, `AP50=0.536`, and `AP75=0.385`. Its log is:

```text
eval_medium_ap36p142_checkpoint_current_code_gpu3.log
```

## Diagnosis

The 24-epoch runs used a constant learning rate because `lr_drop=100`.
Both seed-43 variants peaked at epoch 22 and regressed at epoch 23. Resuming the
epoch-22 best checkpoint for one epoch at `1e-5` recovered 0.5485 AP without
scale routing and 0.7404 AP with scale routing. The largest routed-run gain was
AP75 (+1.5068), consistent with better final localization.

Scale routing still contributes after the same low-LR finish: +0.2449 AP,
+0.4086 AP75, +0.8190 APM, and +0.2612 APL versus iterative refinement alone.
It does not improve APS in this comparison.

`register_border_tokens=1` is part of the source run and must be retained.
The current CLI default is zero, which silently changes forward behavior even
though the checkpoint remains loadable.

## Reproduction

This command performs exactly one low-LR finish epoch from the epoch-22 source
checkpoint. `epochs=24` is intentional because resume starts at epoch 23.

```bash
CUDA_VISIBLE_DEVICES=3 \
MPLCONFIGDIR=/tmp/mpl_rfdetr_ap36 \
XDG_CACHE_HOME=/tmp/rfdetr_ap36_cache \
PYTHONPATH=src \
/home/cpc/.conda/envs/rfdetr-dinov3/bin/python -u run_coco_subset.py \
  --subset medium \
  --epochs 24 \
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
  --projector-scale P3 P4 P5 \
  --multi-scale \
  --expanded-scales \
  --aug-preset default \
  --lr 0.0001 \
  --lr-encoder 0.00015 \
  --lr-drop 22 \
  --weight-decay 0.0001 \
  --use-cdn \
  --dn-number 50 \
  --dn-box-noise-scale 0.6 \
  --dn-label-noise-scale 0.5 \
  --dn-loss-coef 0.5 \
  --seed 43 \
  --backbone-register-border-tokens 1 \
  --resume output/coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_dn50_box0.6_loss0.5_iterref_scaleroute_seed43_e24/checkpoint_best_regular.pth \
  --output-dir output/coco_medium_ap36_reproduction
```

## Unclaimed Follow-up

The source now exposes `bbox_refine_mode=layerwise|residual`. Both modes have
unit coverage proving exact shared-mode output at initialization and independent
gradient flow, but neither mode was used to claim the 36.1423 result. They remain
clean follow-up ablations for testing whether the shared bbox MLP limits further
decoder gains.
