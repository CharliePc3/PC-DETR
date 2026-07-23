# Dense O2O D2 and SA Group-Replacement Plan

## Fixed protocol

- COCO medium subset, 24 epochs, seed 42 unless explicitly marked seed 43.
- Batch size 8 with gradient accumulation 2, total batch size 16.
- DINOv3-S, P3/P4/P5, feature indexes 2/5/8/11, multi-scale expanded training.
- Four decoder layers, 300 inference queries, Group-aware CDN with DN 50, box noise 0.6, loss coefficient 0.5.
- Dense O2O and SA matching remain independent in this phase.

## D2 ablation

The existing D2 result is the seed-42 control: AP 34.290, APs 16.250, APm 37.506, APl 54.409.
D1 is the small-object guardrail: AP 34.204 and APs 16.809.

| Variant | Changed factor | Purpose |
|---|---|---|
| D2 seed43 | Seed only | Confirm reproducibility of the current D2 result. |
| D2 CB-N1 | CopyBlend objects 3 -> 1 | Test density and occlusion strength. |
| D2 CB-P025 | CopyBlend probability 0.5 -> 0.25 | Test augmentation frequency. |
| D2 CB-Stop12 | CopyBlend stop 21 -> 12 | Remove the long CopyBlend-only tail. |
| D2 CB-Ctx00010 | Expansion 0.1/0.25 -> 0.0/0.1 | Reduce copied context and collateral occlusion. |

Promotion rule: prefer AP >= 34.20 and APs >= 16.80. After the seed-42 screen, combine only the
best two factor changes and confirm that candidate with seed 43.

## D2 results (2026-07-23)

All values are best-epoch COCO metrics. The D2 seed-43 control was resumed
from its complete training checkpoint after an external interruption.

| Variant | Seed | Best epoch | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict baseline | 42 | 22 | 33.948 | 51.070 | 36.183 | 17.118 | 36.347 | 53.482 |
| D2 base | 42 | 24 | 34.290 | 51.031 | 36.290 | 16.250 | 37.506 | 54.409 |
| D2 base | 43 | 23 | 34.339 | 50.758 | 36.708 | 17.329 | 37.148 | 51.816 |
| CopyBlend N=1 | 42 | 24 | **34.897** | **51.760** | 37.378 | 17.541 | 37.089 | 53.582 |
| CopyBlend p=0.25 | 42 | 24 | 34.589 | 51.251 | 36.963 | **18.499** | **37.334** | 54.153 |
| CopyBlend stop=12 | 42 | 24 | 34.092 | 50.905 | 36.358 | 17.024 | 36.912 | 52.220 |
| Context 0.00/0.10 | 42 | 22 | 34.705 | 51.080 | **37.470** | 17.934 | 37.032 | **54.751** |
| N=1 + Context 0.00/0.10 | 42 | 23 | 34.395 | 51.064 | 36.911 | 17.470 | 36.734 | 52.656 |

The two-seed D2-base mean is 34.314 AP, 0.349 points above the strict
two-seed mean. The gain is concentrated in AP75 (+0.432), APm (+0.779), and
APl (+0.303), while mean AP50 and APs change by -0.168 and -0.180.

N=1 is the best observed overall configuration, gaining 0.607 AP over the D2
seed-42 control and 0.950 AP over the aligned strict baseline. Reducing copied
context is useful by itself, especially for AP75 and APl, but it does not
combine additively with N=1. The combined run loses 0.502 AP, 0.696 AP50,
0.467 AP75, and 0.926 APl relative to N=1.

**Decision:** retain `copyblend_num_objects=1`, `copyblend_prob=0.5`, and
`copyblend_expand_ratios=(0.1, 0.25)` as the current AP candidate. Confirm N=1
with seed 43 before promoting it to full COCO. Keep p=0.25 as the small-object
candidate and context 0.00/0.10 as a localization/large-object diagnostic.
Reject the early CopyBlend stop and the N=1 plus tight-context combination.

## SA as a Group replacement

The SA-Matching paper reports target replication factors (small, medium, large) = (1, 7, 9).
The previous Group-6 integration had to use (6, 7, 9), which removed strict O2O supervision for
small objects across the training groups and overlapped heavily with Group DETR.

The replacement experiment uses SA only on intermediate decoder outputs during epochs [2, 20).
The final decoder output and encoder output remain strict grouped O2O, and inference still uses 300 queries.

| Variant | Base positives per GT | SA auxiliary budget | Question |
|---|---:|---:|---|
| G6 strict | 6 | none | Current accuracy control. |
| G4 strict | 4 | none | Cost and accuracy of reducing Group alone. |
| G4+SA | 4 | 4/7/9 | Can scale allocation recover G6 with fewer groups? |
| G2 strict | 2 | none | Stronger Group reduction control. |
| G2+SA | 2 | 2/7/9 | Can two groups plus SA replace six uniform groups? |
| G1 strict | 1 | none | No-Group control. |
| G1+SA | 1 | 1/7/9 | Pure SA replacement with original scale factors. |

The first decision uses both AP and wall-clock training time. A candidate advances when it either exceeds
G6 strict AP or stays within 0.10 AP while providing a material training-time reduction. The best strict and
SA candidate then receive seed-43 confirmation before any full-COCO run.
