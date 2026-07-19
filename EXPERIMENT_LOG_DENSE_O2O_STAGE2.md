# Dense O2O Stage 2

## Definition

DEIM/DEIMv2 Dense O2O keeps strict one-to-one Hungarian matching and increases
the number of objects per training image:

- `image`: Mosaic + MixUp, corresponding to the original DEIM Dense O2O.
- `enhanced`: Mosaic + MixUp + CopyBlend, corresponding to the DEIMv2 upgrade.

The cross-decoder-layer regression union in `DEIMCriterion` is a separate loss
mechanism and is not included in this stage.

## Fixed Detection Configuration

- COCO medium: 5,000 train / 2,000 val
- 24 epochs, seed 42
- batch size 8, gradient accumulation 2, total batch size 16
- DINOv3-S, P3/P4/P5, expanded multi-scale, resolution 640
- decoder layers 4, queries/select 300
- group DETR 6
- CDN: number 50, label noise 0.5, box noise 0.6, loss coefficient 0.5
- Budgeted SA disabled

## Schedule

- Mosaic + MixUp: epochs `[2, 12)`, probability 0.5 each
- CopyBlend: epochs `[2, 21)`, probability 0.5
- CopyBlend: minimum source area 100 px2, 3 objects/image, expansion ratio `[0.1, 0.25]`
- MixUp and CopyBlend are mutually exclusive for a batch, matching DEIMv2
- Epochs `[21, 24)` use clean data

## Experiments

| ID | Dense O2O mode | Purpose |
|---|---|---|
| D0 | disabled | contemporaneous strict baseline |
| D1 | image | isolate original Mosaic + MixUp Dense O2O |
| D2 | enhanced | measure the incremental value of CopyBlend over D1 |

Only after D1/D2 are evaluated independently should the winning Dense O2O mode
be combined with the S3 Budgeted SA schedule.
