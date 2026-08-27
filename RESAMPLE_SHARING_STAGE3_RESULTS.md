# Resampling Sharing and Projector Efficiency Results

All accuracy experiments use the same medium COCO protocol, selective-source
pruning, C2f depth 3/3/3, and seeds 42 or 43. AP values below are COCO AP points.

## Stage 3: Resampling Weight Sharing

| Variant | Shared kernels | Params | Seed 42 | Seed 43 | Mean AP | vs A1 prune |
|---|---|---:|---:|---:|---:|---:|
| A1 prune | None | 37.476M | 35.994 | 35.542 | 35.768 | 0.000 |
| S1 | P3 F2/F5/F11 | 36.886M | - | 35.446 | - | - |
| S2 | P5 F8/F11 | 36.813M | 35.902 | 35.537 | **35.719** | **-0.048** |
| S3 | P3 and P5 | 36.223M | 35.347 | 35.648 | 35.498 | -0.270 |

S2 is selected. It removes 0.664M parameters from A1 prune while preserving
the two-seed mean within 0.05 AP. S3's seed-43 gain did not reproduce at seed
42, showing that P3 sharing is not stable enough to retain.

## Stage 4: Efficient P5 Convolution

E1b keeps the selected S2 sharing and replaces only the F8/F11 group-2 P5
sampler with a shared depthwise 3x3 plus pointwise 1x1 path.

| Variant | Params | Seed 43 AP | Delta vs S2 |
|---|---:|---:|---:|
| S2 | 36.813M | 35.537 | 0.000 |
| E1b DW+PW | 36.300M | 35.212 | -0.325 |

RTX 5090 BF16 projector-only latency (100 measured iterations):

| Batch | S2 | E1b | E1b change |
|---:|---:|---:|---:|
| 1 | 3.460 ms | 3.531 ms | 2.0% slower |
| 8 | 4.570 ms | 4.519 ms | 1.1% faster |

The theoretical FLOP reduction does not translate into useful GPU latency, so
the DW+PW replacement is rejected.

## Stage 5: Projector Distillation

The frozen S2 seed-42 checkpoint teaches normalized P3/P4/P5 features to E1b.
Distillation stops at epoch 20 so the last four epochs optimize detection only.

| Variant | Coefficient | Seed 43 AP | Delta vs E1b | Delta vs S2 |
|---|---:|---:|---:|---:|
| E1b, no distillation | 0.00 | 35.212 | 0.000 | -0.325 |
| D1 | 0.25 | 34.997 | -0.215 | -0.540 |
| D2 | 0.50 | 35.366 | +0.154 | -0.171 |

D2 recovers about 47% of E1b's accuracy loss, but the student remains below S2
and has no meaningful latency advantage. No stronger distillation sweep is
justified for this architecture.

## Final Decision

Use A1 selective-source pruning with P5-only deep resampling-kernel sharing:

- P3: independent F2/F5/F11 transpose-convolution weights.
- P4: native-resolution paths with no resampling parameters.
- P5: F2 keeps its independent full convolution; F8/F11 share the group-2 3x3
  convolution weight while retaining source-specific normalization.
- C2f remains 3/3/3.

This is the best verified accuracy-efficiency point from stages 3-5. Projector
distillation remains available as an experimental training option, but is not
part of the selected recipe.
