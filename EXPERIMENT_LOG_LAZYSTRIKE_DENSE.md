# LazyStrike Detection-Aware Dense Consistency

## Goal

Test whether the successful LazyStrike v2 refinement can be improved by moving
part of the objective from CLS aggregation to dense patch consistency at the
DINOv3 layers actually consumed by RF-DETR (`[2, 5, 8, 11]`). The refinement
remains offline: detection inference receives only the refined backbone weights
and gains no extra module or latency.

## Fixed Protocol

- Refinement data: 5,000 shuffled COCO train2017 images, seed 42
- DINOv3: small, 640 resolution, one refinement epoch
- Trainable encoder blocks: 8-11 plus final norm
- Top learning rate: `2.5e-6`; layer decay: `0.8`
- Teacher preservation weight: `1.0`
- Dense layers and weights: block 8/11, `0.35/0.65`
- Dense object/global weights: `0.15/0.05`
- Diagnostic: COCO val2017, 200 images, seed 42, 12 visualizations
- Detection: aligned medium split, 24 epochs, seed 42, total batch 16,
  group 6, CDN, P3/P4/P5, multi-scale expanded
- Hardware: physical GPU 3; stages run serially

## Ablations

| Variant | CLS align/cover/flip | Multi-level distill | Object dense | Global dense | Purpose |
|---|---:|---:|---:|---:|---|
| Existing v2 e1 | 0.25/0.15/0.25 | final only | 0 | 0 | Established reference, best AP 34.08 |
| `last4_control` | 0.25/0.15/0.25 | final only | 0 | 0 | Isolate blocks 8-11 + lower layerwise LR |
| `dense_only` | 0/0/0 | blocks 8/11 | 0.15 | 0.05 | Test dense objective independently |
| `hybrid` | 0.25/0.15/0.25 | blocks 8/11 | 0.15 | 0.05 | Test complementarity with v2 CLS objective |

## Dense Objective

1. Extract original student, frozen original teacher, and horizontally flipped
   student patch features from blocks 8 and 11.
2. Flip the augmented feature grid back into the original spatial coordinates.
3. Preserve each layer against its frozen teacher representation.
4. Compute soft patch-box overlap using intersection over patch area.
5. Average consistency within each object, then equally across boxes and images,
   preventing large boxes from dominating the objective.
6. Add a low-weight global consistency term so unlabeled but meaningful objects
   are not automatically treated as background.

## Decision Rules

- Primary: best medium detection AP over 24 epochs.
- Localization: AP75 and AP for medium/large objects.
- Small-object guardrail: APs must not materially regress.
- Representation checks: horizontal-flip consistency and multi-object coverage.
- FP/GP proxy remains diagnostic only; high-response foreground patches are not
  automatically spurious and should not be optimized against directly.

Runtime status is written to
`lazystrike_dense_consistency_gpu3_queue_nohup.log`. Consolidated metrics are
written to `output/lazystrike_dense_consistency/summary.tsv`.

## Results (2026-07-17)

All three GPU3 jobs completed successfully. Values below are COCO AP points at
the best epoch of each 24-epoch medium run.

| Variant | Best epoch | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current base | 24 | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 |
| Existing v2 e1 | 22 | 34.08 | 51.28 | 36.37 | 15.97 | 36.93 | 53.40 |
| Existing v2 e2 | 24 | **34.36** | **51.71** | 36.37 | 16.77 | 36.48 | 52.97 |
| `last4_control` | 24 | 34.01 | 51.49 | 35.73 | 16.95 | 35.98 | 52.78 |
| `dense_only` | 22 | 34.03 | 50.94 | **36.75** | **17.36** | **36.85** | 51.95 |
| `hybrid` | 23 | 33.72 | 51.02 | 35.78 | 16.26 | 36.06 | **53.06** |

The last-five-epoch mean AP/AP75 was 33.16/35.15 for base, 33.51/35.72 for
v2 e1, 33.71/35.87 for v2 e2, 33.44/35.48 for `last4_control`,
33.56/35.84 for `dense_only`, and 33.33/35.34 for `hybrid`.

### Diagnostic Interpretation

- `last4_control` strongly changed CLS behavior: top-100 box coverage rose from
  0.584 (base) to 0.754 and CLS flip cosine from 0.960 to 0.987. This did not
  outperform v2 e1 detection, so stronger CLS coverage is not sufficient.
- `dense_only` kept CLS diagnostics close to base (top-100 coverage 0.587,
  CLS flip cosine 0.968), but improved AP75 by 1.51 points, APs by 0.48, and
  APm by 0.83 over base. The dense objective primarily improved precise
  localization and small/medium objects rather than coarse AP50 recognition.
- Relative to `last4_control`, `dense_only` gained 1.02 AP75, 0.41 APs, and
  0.87 APm, while losing 0.55 AP50 and 0.84 APl. Its overall AP was essentially
  tied (+0.02), but the error profile changed meaningfully.
- `hybrid` had slightly stronger CLS coverage than `last4_control` but lower AP.
  During the last 250 refinement steps, its block-11 teacher-distillation loss
  reached 0.024 versus 0.004 for `dense_only`, while its object consistency loss
  increased instead of decreasing. The simultaneous CLS and dense objectives
  compete at the final block.
- FP/GP proxy ratios did not track detection quality and remain unsuitable as
  an optimization or model-selection target for complex COCO scenes.

### Decision

The best general detection checkpoint remains existing v2 conservative e2.
The dense route is nevertheless validated as a distinct localization-oriented
signal. Do not continue the current simultaneous `hybrid` recipe. The next
high-value experiment is sequential refinement: initialize from v2, preserve or
freeze block 11, and apply dense consistency mainly to block 8 (optionally block
10) with a reduced global term. This tests complementarity without forcing CLS
aggregation and dense invariance to reshape the final block at the same time.
