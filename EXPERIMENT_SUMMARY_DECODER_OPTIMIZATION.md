# RF-DETR-DINOv3 Decoder Optimization Summary

## Scope and terminology

The experiments cover four complementary training mechanisms. They should not
all be described as Hungarian-matching modifications.

| Mechanism | Training role | Directly changes Hungarian matching |
|---|---|---|
| Group DETR | Runs independent O2O matching for several query groups | Yes |
| CDN | Adds GT-derived positive and negative denoising queries with known assignments | No |
| Budgeted SA | Adds scale-dependent positive matches on auxiliary decoder outputs | Yes |
| Dense O2O | Creates denser image targets with Mosaic, MixUp, and CopyBlend | No |

Unless noted otherwise, the aligned medium protocol uses DINOv3-S, P3/P4/P5,
feature indexes 2/5/8/11, expanded multi-scale training, four decoder layers,
300 inference queries, resolution 640, and total batch size 16.

## 1. CDN denoising

The first 12-epoch medium screen established that CDN is useful only after its
strength is tuned. Values are best COCO AP points from that screen.

| Configuration | AP |
|---|---:|
| No CDN | 25.04 |
| Initial CDN baseline | 25.13 |
| DN loss coefficient 0.5 | 25.39 |
| Loss 0.5, box noise 0.6 | 25.60 |
| Loss 0.5, box noise 0.6, DN 50 | **26.29** |
| Loss 0.25 | 24.69 |
| Loss 0.5, box noise 0.4 | 25.47 |
| Loss 0.5, box noise 0.8, DN 50 | 25.67 |
| Loss 0.5, box noise 0.6, DN 200 | 25.14 |

The selected CDN parameters are:

```text
dn_number=50
dn_label_noise_scale=0.5
dn_box_noise_scale=0.6
dn_loss_coef=0.5
positive_and_negative_queries=true
```

Later Group-6 screens confirmed that changing `dn_number` to 100 or the loss
coefficient to 0.3/0.7 was worse than DN 50 and loss 0.5.

The 24-epoch full-COCO comparison confirms that the gain transfers beyond the
medium subset:

| Full-COCO configuration | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| Group 13, no CDN | 51.10 | 69.98 | 55.03 | 29.65 | 55.83 | 70.91 |
| Group 13 + CDN | 51.50 | 70.18 | 55.65 | 29.76 | 56.24 | 71.03 |
| Group 6 + CDN | **51.55** | 70.17 | **55.85** | **30.18** | 56.12 | **71.08** |

**Decision:** keep CDN. It provides a reproducible convergence and accuracy
gain, with the strongest evidence coming from the full-COCO improvement of
about 0.40 AP at Group 13.

## 2. Group count

The aligned 18-epoch, total-batch-16 medium screen used the selected CDN
configuration.

| Group count | AP |
|---:|---:|
| 13 | 32.70 |
| 8 | 32.49 |
| 7 | 32.55 |
| 6 | **32.99** |
| 4 | 32.29 |

Reducing Group from 13 to 6 did not weaken the model. It produced the best
medium result in this screen and slightly improved full-COCO AP from 51.50 to
51.55 when combined with CDN. Group 4 lost 0.70 AP relative to Group 6.

Fresh 24-epoch Group-6 strict baselines reached 33.948 AP with seed 42 and
33.983 AP with seed 43, giving a two-seed mean of 33.965 AP.

**Decision:** use Group 6 as the accuracy baseline. Group 2 and Group 4 remain
possible efficiency variants, but not replacements for the main model.

## 3. Budgeted SA matching

Budgeted SA was first tested with Group 6 and scale budgets 6/7/9. Under the
aligned 24-epoch protocol, the two-seed mean changed from 33.97 AP for strict
Group-6 matching to 33.87 AP with SA. Alternative budgets 6/7/8, 6/8/10, and
7/8/9 also remained below the strict seed-42 baseline.

The Group-replacement screen reached the following seed-42 AP values:

| Configuration | AP | Difference from Group-6 strict |
|---|---:|---:|
| Group-6 strict | **33.948** | 0.000 |
| Group-4 + SA 4/7/9 | 33.565 | -0.382 |
| Group-2 + SA 2/7/9 | 33.509 | -0.439 |
| Group-1 + SA 1/7/9 | 32.247 | -1.701 |

Almost all requested SA matches were fulfilled, so the shortfall was not caused
by query-capacity truncation. The current budgets add no extra small-object
positives and increasingly over-allocate medium/large positives as Group is
reduced. This overlaps with Group DETR without providing a stable AP gain.

**Decision:** reject the current auxiliary-only Budgeted SA design for the
accuracy path. Do not spend more runs on budget or activation-window searches
without a materially different matching hypothesis.

## 4. Dense O2O

Dense O2O preserves strict O2O matching but changes the target distribution.
The enhanced D2 schedule is:

```text
epochs 0-1:   original images
epochs 2-11:  Mosaic + MixUp + CopyBlend
epochs 12-20: CopyBlend only
epochs 21-23: original images
```

The D2 base result is reproducible. Its two-seed mean is 34.314 AP versus
33.965 AP for the strict Group-6 + CDN baseline, a gain of 0.349 AP. Mean AP75,
APm, and APl improve by 0.432, 0.779, and 0.303 points respectively.

The seed-42 CopyBlend screen produced:

| Dense O2O configuration | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| D2 base, N=3 | 34.290 | 51.031 | 36.290 | 16.250 | **37.506** | 54.409 |
| CopyBlend N=1 | **34.897** | **51.760** | 37.378 | 17.541 | 37.089 | 53.582 |
| CopyBlend p=0.25 | 34.589 | 51.251 | 36.963 | **18.499** | 37.334 | 54.153 |
| Context 0.00-0.10 | 34.705 | 51.080 | **37.470** | 17.934 | 37.032 | **54.751** |
| CopyBlend stop=12 | 34.092 | 50.905 | 36.358 | 17.024 | 36.912 | 52.220 |
| N=1 + Context 0.00-0.10 | 34.395 | 51.064 | 36.911 | 17.470 | 36.734 | 52.656 |

N=1 is the strongest observed medium configuration, improving AP by 0.950
points over the aligned strict seed-42 baseline. It still requires seed-43
confirmation. Tight copied context is useful by itself but is not additive with
N=1. Stopping CopyBlend at epoch 12 is harmful, confirming that the long,
lighter CopyBlend tail contributes to the result.

Dense O2O also exposes a significant memory interaction with CDN and Group:

| Configuration | Peak allocated memory |
|---|---:|
| Strict Group 6 + CDN | 12,643 MiB |
| D2 base | 24,840 MiB |
| D2 N=1 | 25,912 MiB |
| D2 N=1 + Context 0.00-0.10 | 21,615 MiB |

Mosaic/MixUp raises the average number of targets in a local batch from about
107 to roughly 412-419. CDN derives its padding from the largest per-image GT
count and currently guarantees at least one denoising group, so `dn_number=50`
is not a hard upper bound. Group 6 then replicates those denoising queries six
times. This is the primary reason Dense O2O can nearly double peak memory.

**Decision:** keep D2 as a validated improvement and treat N=1 as the current
candidate. Before a full-COCO run, confirm N=1 with seed 43 and cap the number
of GT instances used to construct CDN queries without removing any GT from
ordinary Hungarian matching.

## Current decoder-side recipe

```text
decoder_layers=4
num_queries=300
group_detr=6

use_cdn=true
dn_number=50
dn_label_noise_scale=0.5
dn_box_noise_scale=0.6
dn_loss_coef=0.5

use_dense_o2o=true
dense_o2o_mode=enhanced
dense_o2o_start_epoch=2
dense_o2o_image_stop_epoch=12
dense_o2o_copyblend_stop_epoch=21
dense_o2o_mosaic_prob=0.5
dense_o2o_mixup_prob=0.5
dense_o2o_copyblend_prob=0.5
dense_o2o_copyblend_num_objects=1
dense_o2o_copyblend_expand_ratios=(0.1, 0.25)
```

This recipe is the highest observed medium configuration, not yet the final
full-COCO recipe. The validated core is Group 6 + CDN + D2 base; CopyBlend N=1
remains a single-seed promotion candidate.

## Consolidated decisions

1. **Validated and retained:** Group 6, tuned CDN, and D2 Dense O2O.
2. **Current best candidate:** D2 with CopyBlend N=1, pending seed 43.
3. **Rejected:** current Budgeted SA, CopyBlend stop at epoch 12, and N=1 plus
   tight context.
4. **Memory priority:** impose a true CDN GT/query cap before scaling Dense O2O.
5. **Next decoder experiment:** after memory control, test late path annealing
   with separate CDN-off and active-Group-1 final-phase ablations.
