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

## Dense-to-Lazy Reverse Sequential Test (2026-07-24)

The previous simultaneous hybrid reached 33.72 AP because LazyStrike and dense
consistency both reshaped the final block. The previous sequential tests used
the opposite order, LazyStrike to dense consistency, and reduced the standalone
LazyStrike v2 result from 34.36 to at most 34.01 AP. Even with block 11 frozen,
changing blocks 8--10 changed the input distribution consumed by block 11.

The new test reverses the order. It initializes from the validated
`dense_only` epoch-1 checkpoint, then runs the exact FP32 LazyStrike
v2-conservative recipe for two epochs while training only blocks 10--11 and the
final norm. The frozen teacher is initialized from the same dense checkpoint.
Consequently, the layer-8 dense feature is preserved exactly, while the final
blocks can adapt to that feature distribution under LazyStrike supervision and
dense-teacher distillation.

`run_dense_to_lazy_gpu0.sh` runs on physical GPU0 and performs:

1. two-epoch Dense-to-Lazy refinement on the same seeded COCO train-5k subset;
2. matched COCO-val-200 diagnostics for epoch-1 and epoch-2 checkpoints;
3. aligned 24-epoch medium detection for epoch 2;
4. summary export to `output/dense_to_lazy/summary.tsv`.

Success is measured against standalone LazyStrike v2 rather than the original
baseline: AP should exceed 34.36 with last-five AP at least 33.71, while
retaining the dense route's AP75/small-object behavior. If epoch 2 misses this
target, epoch 1 remains available for a targeted detection follow-up without
repeating refinement.

## Dense-to-Lazy Results (2026-07-26)

The queue completed successfully. The epoch-2 checkpoint reached 33.94 AP at
epoch 24 with 50.85 AP50, 35.94 AP75, 15.61 APs, 36.48 APm, 53.65 APl, and a
33.46 last-five mean AP.

| Variant | AP | AP50 | AP75 | APs | APm | APl | Last-5 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current base | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |
| Dense-only | 34.03 | 50.94 | **36.75** | **17.36** | **36.85** | 51.95 | 33.56 |
| LazyStrike v2 e2 | **34.36** | **51.71** | 36.37 | 16.77 | 36.48 | 52.97 | **33.71** |
| Dense to Lazy e2 | 33.94 | 50.85 | 35.94 | 15.61 | 36.48 | **53.65** | 33.46 |

Reverse sequential refinement therefore did not combine the two gains. It was
0.42 AP below standalone LazyStrike and 0.08 AP below dense-only. Relative to
dense-only, it lost 0.81 AP75 and 1.75 APs while gaining 1.70 APl. Relative to
standalone LazyStrike, it lost 0.42 AP, 0.86 AP50, 0.43 AP75, and 1.16 APs,
while gaining 0.68 APl. The result is a strong redistribution toward large
objects rather than an additive improvement.

Optimization was healthy and slightly easier from the dense initialization:
average LazyStrike loss was 0.13016/0.07799 in refinement epochs 1/2 versus
0.13411/0.08065 in the original standalone v2 run. Teacher distillation was
nearly identical. Lower objective loss therefore did not predict better
detection.

Epoch-1 and epoch-2 token diagnostics reveal the transition:

| Checkpoint | Top-100 coverage | Small coverage | LAST coverage | LAST-small coverage | CLS flip cosine |
|---|---:|---:|---:|---:|---:|
| Dense-only | 0.5871 | 0.1966 | 0.7506 | 0.1614 | 0.9681 |
| Dense to Lazy epoch 1 | 0.8066 | 0.4950 | 0.7452 | 0.1569 | 0.9921 |
| Dense to Lazy epoch 2 | 0.8519 | 0.5497 | 0.7289 | 0.1357 | 0.9943 |
| LazyStrike v2 epoch 2 | 0.8366 | 0.5225 | 0.7365 | 0.1233 | 0.9929 |

The second LazyStrike epoch continued to improve CLS-score box coverage while
reducing LAST stability, especially for small boxes. This again shows that CLS
coverage is not a sufficient selection metric. Freezing layer 8 did not retain
the dense-only detector behavior because dense-only also improved layer-11
patch consistency, and changing blocks 10--11 altered cross-level compatibility
and the final representation consumed by the detector.

**Decision:** reject Dense-to-Lazy epoch 2 as a combined checkpoint. Run the
already available epoch-1 checkpoint through aligned detection once, because it
preserves dense-like LAST-small coverage and can determine whether the failure
is primarily second-epoch over-refinement. If epoch 1 also fails, stop pure
sequential combinations. The next method should use size-aware layer-11
teacher preservation: keep the dense teacher's small/medium-box patch features
strongly anchored while applying LazyStrike mainly to global and large-object
aggregation.
