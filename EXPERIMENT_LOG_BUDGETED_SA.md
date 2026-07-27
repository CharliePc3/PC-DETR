# Budgeted SA Matching Ablation

## Setup

- COCO medium subset, 24 epochs, resolution 640, P3/P4/P5, group DETR 6.
- CDN enabled with 50 denoising queries.
- Budgeted SA assigns total matching budgets by target area.
- Initial budgets `[6,7,9]` were tested over three activation windows.

## Initial Window Ablation

The first comparison used an older `currentbase` run from commit `ce7c923`.
The BSA runs were made from the later `0346323` worktree, so the apparent gain
against the 33.61 AP old baseline is not a strict source-aligned result.

| Variant | Active epochs | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---:|---:|---:|---:|---:|---:|
| Old baseline | none | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 |
| BSA S1 | 0-24 | 33.77 | 50.31 | 35.95 | 16.33 | 35.72 | 53.20 |
| BSA S2 | 0-20 | 33.50 | 50.66 | 35.62 | 16.85 | 35.84 | 52.01 |
| BSA S3 | 2-20 | 33.88 | 50.74 | 35.97 | 17.33 | 36.09 | 52.28 |

S3 was selected for strict confirmation because it had the best initial AP and
avoided applying the modified matcher in the first two and last four epochs.

## Strict Confirmation (2026-07-20)

Fresh non-BSA baselines and BSA runs were made from the same current source.

| Variant | Seed | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict baseline | 42 | **33.95** | 51.07 | **36.18** | 17.12 | **36.35** | **53.48** |
| BSA `[6,7,9]` S3 | 42 | 33.88 | 50.74 | 35.97 | **17.33** | 36.09 | 52.28 |
| Strict baseline | 43 | **33.98** | **51.05** | 35.95 | **16.82** | **36.75** | 52.14 |
| BSA `[6,7,9]` S3 | 43 | 33.85 | 50.59 | **36.24** | 16.30 | 36.58 | **53.05** |

Two-seed mean AP is 33.97 for the strict baseline and 33.87 for BSA. BSA loses
about 0.10 AP and 0.40 AP50 on average; its mean AP75 gain is only 0.04. The
direction of APs and APl changes also flips between seeds. The seed-43 last-five
mean AP drops from 33.57 to 33.22.

Alternative seed-42 budgets `[6,7,8]`, `[6,8,10]`, and `[7,8,9]` reached
33.84, 33.74, and 33.66 AP, respectively, all below the 33.95 strict baseline.

## Decision

Reject the current Budgeted SA matcher as an AP improvement. It sometimes
redistributes performance toward AP75 or large objects, but the effect is not
stable across seeds and lowers mean AP/AP50. Do not run it on full COCO or
combine it with backbone refinements without a materially different matching
hypothesis.

## Group-Replacement Screen (2026-07-21)

The matcher was extended so `group_detr=1` can use SA positives on auxiliary
decoder outputs while the final output remains strict O2O. All runs used seed
42 and the same 24-epoch medium protocol as the strict Group-6 baseline.

| Variant | AP | AP50 | AP75 | APs | APm | APl | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| G6 strict | **33.948** | **51.070** | **36.183** | 17.118 | **36.347** | **53.482** | about 2:18 |
| G1 strict | 32.721 | 48.994 | 35.052 | 15.285 | 34.965 | 51.795 | 2:18 |
| G1 + SA 1/7/9 | 32.247 | 48.341 | 34.608 | 16.142 | 33.822 | 50.671 | 2:34 |
| G2 strict | 33.450 | 50.731 | 35.514 | **18.618** | 35.813 | 51.917 | 1:31 |
| G2 + SA 2/7/9 | 33.509 | 50.667 | 35.596 | 16.239 | 35.895 | 51.529 | 1:35 |
| G4 strict | 33.305 | 50.470 | 34.993 | 16.400 | 36.285 | 52.606 | 1:41 |
| G4 + SA 4/7/9 | 33.565 | 50.207 | 36.181 | 15.843 | 35.685 | 52.631 | 1:45 |

SA fulfilled effectively all requested extra matches, so the shortfall is not
query-capacity truncation. The 1/7/9, 2/7/9, and 4/7/9 budgets add no extra
small-object positives and allocate most extras to medium and large targets.
G1 therefore receives the strongest redistribution and performs worst.

G4+SA is the closest efficiency candidate: it is 0.382 AP below G6 strict but
about 24% faster and uses about 12% less peak memory. G2 strict is 0.498 AP
below G6, about 34% faster, and improves APs by 1.50 points. Neither candidate
meets the predefined 0.10-AP promotion rule.

**Decision:** SA does not replace Group-6 under the current auxiliary-only
matching design. Keep Group-6 for the accuracy path. G4+SA and G2 strict may
be retained only as explicit speed/accuracy variants.
