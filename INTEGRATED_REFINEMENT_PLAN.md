# Integrated Backbone, Projector, and Decoder Refinement

## Fixed detector recipe

The integration experiment keeps the current detector recipe fixed:

- DINOv3-small outputs: F2/F5/F8/F11.
- P3 sources: F2/F5/F11.
- P4 sources: F2/F5/F8/F11.
- P5 sources: F2/F8/F11.
- Source branches are physically pruned; C2f depth is 3/3/3.
- P5 F8/F11 share their group-2 resampling kernel (S2).
- Four decoder layers with shared iterative box refinement and legacy scale routing.
- Group DETR 6 and tuned CDN (DN 50, box noise 0.6, label noise 0.5, loss 0.5).
- Enhanced Dense O2O with CopyBlend N=1.
- Medium COCO, 24 epochs, total batch size 16, LR drop at epoch 20.

## Controlled backbone matrix

Only `pretrained_encoder` changes between cells:

| Cell | Backbone initialization | Purpose |
|---|---|---|
| official | Official DINOv3-small | Integrated detector reference |
| lazystrike | LazyStrike v2-conservative e2 | Test the strongest general detection refinement |
| dense_only | Detection-aware dense consistency e1 | Test the localization-oriented refinement |

The three cells use data seed 42 and detector initialization seed 1042. Peak
AP, last-five AP, AP75, and scale-specific AP are collected. An offline refined
backbone is promoted to a second seed only if it improves peak AP by at least
0.20 points, improves last-five AP, and does not merely shift performance to
large objects.

UniRefiner and combined refinement checkpoints are intentionally excluded from
this first matrix because the handoff results show weak downstream transfer or
objective interference. Projector distillation is also disabled because it was
not selected in the final S2 recipe.

## Reproduction

```bash
nohup env GPU_OFFICIAL=0 GPU_LAZYSTRIKE=1 GPU_DENSE=3 \
  bash run_integrated_refinement_s2_denseo2o.sh \
  > integrated_refinement_s2_denseo2o_controller.log 2>&1 &
```

The controller verifies both refinement checkpoint hashes, snapshots the dirty
working tree, runs all three cells in parallel, and writes:

```text
output/integrated_refinement_s2_denseo2o/summary_seed42.tsv
```

## Seed-42 result

| Backbone | AP | AP50 | AP75 | APs | APm | APl | Last-5 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official DINOv3 | 36.103 | 52.952 | 37.990 | 18.215 | 38.960 | 54.829 | 35.548 |
| LazyStrike v2 | 36.043 | 52.543 | 38.067 | 17.465 | 39.499 | 54.479 | 35.551 |
| Dense consistency | **36.228** | **53.142** | **38.469** | 16.958 | **39.953** | 54.151 | **35.637** |

Relative to the official-backbone cell, LazyStrike changes AP by -0.059 and
does not transfer its old-architecture gain. Dense consistency changes AP by
+0.125, AP75 by +0.478, and APm by +0.993, but APs drops by 1.257 and APl by
0.677. Its gain is therefore a localization/medium-object specialization, not
a broad improvement.

Neither refined checkpoint meets the predeclared second-seed promotion gate of
+0.20 peak AP with a favorable last-five and scale profile. The default remains
the official DINOv3 backbone. Dense consistency is retained as an optional
task-specific initialization, while LazyStrike is rejected for this integrated
recipe.

The previous S2 experiment did not enable Dense O2O and did not use the same
explicit detector-initialization seed. Its 35.902 seed-42 AP is useful context,
but the difference from the new 36.103 official cell is not a strict
single-factor estimate. The three rows above are the controlled backbone
comparison.
