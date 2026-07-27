# DINOv3 Backbone Refinement Stage Summary

## Scope

This stage evaluated three offline DINOv3-small refinement directions for the
RF-DETR-DINOv3 detector:

1. UniRefiner-style spurious-token filtering and register absorption.
2. LazyStrike-style foreground-aware CLS aggregation.
3. Detection-aware multilevel dense consistency.

All refinements export a plain DINOv3 state dictionary. They add no detector
module or inference-time latency. Unless noted otherwise, detection comparisons
use the aligned medium protocol: 24 epochs, seed 42, total batch 16, group 6,
CDN, P3/P4/P5, 640 resolution, and expanded multi-scale training.

## Standalone Results

| Refinement | AP | AP50 | AP75 | APs | APm | APl | Last-5 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current DINOv3 baseline | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |
| UniRefiner `balanced_zero3e` | 33.88 | 50.96 | 35.95 | 17.09 | 36.58 | 51.94 | 33.46 |
| LazyStrike v2-conservative e2 | **34.36** | **51.71** | 36.37 | 16.77 | 36.48 | **52.97** | **33.71** |
| Dense consistency `dense_only` | 34.03 | 50.94 | **36.75** | **17.36** | **36.85** | 51.95 | 33.56 |

### UniRefiner

The strongest COCO refinement improved peak AP by 0.27 and last-five AP by
0.30. Under the fixed diagnostic threshold, abnormal-token ratio fell from
9.62% to 5.75%, a relative reduction of about 40%. This validates the token
redistribution mechanism, but downstream evidence remains weak: the
author-recipe seed-43 run did not improve peak AP, and ADE20K linear
segmentation stayed at or below the original 40.89 mIoU baseline.

### LazyStrike

The two-epoch v2-conservative checkpoint is the strongest general detector.
It improved AP by 0.75, AP75 by 1.13, APl by 0.98, and last-five AP by 0.55
at seed 42. Seed 43 retained a smaller positive AP margin, so the direction is
positive across two seeds but has substantial magnitude variance. Its main
weakness is small-object performance.

### Dense Consistency

Dense-only improved AP by 0.42 and last-five AP by 0.40. Its benefit is
localization-specific: AP75 rose by 1.51, APs by 0.48, and APm by 0.83, while
AP50 and APl were essentially unchanged. This validates dense consistency as a
distinct detection signal, but it still requires a seed-43 confirmation.

## Combination Results

| Combination | AP | AP75 | APs | APm | APl | Last-5 AP | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| LazyStrike+dense simultaneous | 33.72 | 35.78 | 16.26 | 36.06 | 53.06 | 33.33 | Final-block conflict |
| LazyStrike to dense sequential | 34.01 | 36.32 | 17.41 | 36.62 | 52.69 | Erased part of Lazy gain |
| Uni+dense joint | 34.11 | 36.39 | 16.17 | 36.05 | 53.67 | Only +0.08 over dense peak |
| Uni to Lazy sequential | 34.04 | 36.66 | 16.24 | 36.86 | 53.32 | Below standalone Lazy |
| Uni+Lazy joint | 33.65 | 35.97 | 15.73 | 36.13 | 52.58 | No positive transfer |
| Dense to Lazy e2 | 33.94 | 35.94 | 15.61 | 36.48 | 53.65 | Shifted quality to large objects |

The gains do not currently add. Simultaneous LazyStrike+dense training
increased final-layer teacher drift, while either sequential order changed the
cross-level feature distribution enough to lose one component's advantage.
UniRefiner pairwise combinations were numerically stable but did not beat the
strongest standalone component. Phase C three-objective training was therefore
not started.

Dense-to-Lazy epoch 1 remains an untested detection checkpoint. Its token
diagnostic retains dense-like LAST-small coverage better than epoch 2, so one
aligned run can determine whether the reverse sequence failed mainly because
of second-epoch over-refinement.

## Engineering Result

`tools/train_dinov3_lazystrike_refine.py` is now a composable refinement
trainer with explicit `unirefiner`, `lazystrike`, and `dense` objectives. It
supports:

- legacy objective inference for existing commands;
- arbitrary supported objective combinations in one optimization loop;
- FP16 or BF16 autocast;
- UniRefiner register/filter/NCE/SCD reuse through
  `tools/refinement/unirefiner_objective.py`;
- shared per-step logging and ordinary DINOv3 checkpoints;
- layer-selective optimization and layer-specific learning-rate scales.

The Phase A, Phase B, and Dense-to-Lazy queue scripts preserve the exact
experiment protocols and write compact TSV summaries.

## Current Decision

- Use LazyStrike v2-conservative e2 as the detection-oriented backbone
  refinement baseline.
- Keep dense-only as the precise-localization and small/medium-object
  alternative.
- Treat UniRefiner as mechanism-validated but downstream-inconclusive.
- Do not run broad three-objective or loss-weight sweeps.
- Before designing another combination, test Dense-to-Lazy epoch 1 and measure
  per-objective gradient cosine by block.
- If combination work continues, preserve dense-teacher layer-11 patches with
  size-aware small/medium-object distillation while applying LazyStrike mainly
  to global and large-object aggregation.

Detailed histories are recorded in `EXPERIMENT_LOG_LAZYSTRIKE_DENSE.md` and
`EXPERIMENT_LOG_UNIFIED_REFINEMENT.md`.
