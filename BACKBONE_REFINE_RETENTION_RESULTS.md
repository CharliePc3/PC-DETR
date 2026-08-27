# Backbone Refinement Retention Results

All runs use the fixed integrated S2 projector, iterative refinement, legacy
scale routing, Group DETR 6, tuned CDN, enhanced Dense O2O, data seed 42, and
detector initialization seed 1042.

## Detection results

| Variant | AP | Delta AP | Last-5 AP | Delta Last-5 |
|---|---:|---:|---:|---:|
| LazyStrike unprotected | 36.043 | - | 35.551 | - |
| lazy_lr01 | 34.760 | -1.283 | 34.070 | -1.481 |
| lazy_anchor1e3 | 35.674 | -0.369 | 35.088 | -0.463 |
| Dense unprotected | 36.228 | - | 35.637 | - |
| dense_lr01 | 33.706 | -2.522 | 33.041 | -2.596 |
| dense_anchor1e3 | 35.752 | -0.476 | 35.321 | -0.316 |

Reducing the selected blocks to 0.1x LR is too restrictive. L2-SP is less
damaging, but neither anchor run improves peak or last-five AP. The criterion
for a combined LR-plus-anchor run is therefore not met.

## Final-backbone diagnostics

The same 200-image COCO-val token protocol used for the unprotected references
was run on both anchor checkpoints.

| Lazy metric | Refined init | Unprotected epoch 24 | Anchor epoch 24 |
|---|---:|---:|---:|
| Top-100 box coverage | 83.66% | 68.20% | 70.13% |
| Medium-box coverage | 79.24% | 46.93% | 51.42% |
| Top-5 patch-in-box | 95.50% | 82.00% | 85.00% |

LazyStrike foreground aggregation is partially retained, but the extra
retention does not improve AP. For Dense consistency, global false-positive
inter-token similarity is 0.4046 for the unprotected final backbone, 0.4138
with the anchor, and 0.4141 for the official final backbone (lower is better).
Parameter anchoring does not retain the useful Dense behavior and slightly
obstructs detector adaptation.

## Decision

Do not promote low-LR retention, L2-SP retention, or their combination. Offline
refinement remains useful as an initialization and mechanism probe, but a
parameter-distance constraint is not an effective way to preserve its useful
behavior during detector training. The next attempt should regularize behavior
directly during detector training, with a short schedule and losses defined on
the exact patch features consumed by the projector.

Outputs:

- `output/backbone_refine_retention_s2_denseo2o/summary.tsv`
- `output/token_analysis/refine_retention/lazy_anchor1e3_epoch24`
- `output/token_analysis/refine_retention/dense_anchor1e3_epoch24`
