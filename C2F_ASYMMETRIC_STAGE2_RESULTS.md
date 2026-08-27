# C2f Asymmetric Stage 2 Results

All results use the same medium COCO protocol and the A1 selective-source prune
path. AP values in the table are COCO AP points averaged over seeds 42 and 43.

| Variant | P3/P4/P5 C2f blocks | Params | Mean AP | Seed range | vs. A1 mask | vs. A1 prune |
|---|---:|---:|---:|---:|---:|---:|
| A1 mask reference | 3/3/3 | 38.583M | 35.675 | 0.001 | 0.000 | -0.093 |
| A1 prune baseline | 3/3/3 | 37.476M | 35.768 | 0.452 | +0.093 | 0.000 |
| B1 P5 reduced | 3/3/1 | 36.820M | 35.593 | 0.151 | -0.082 | -0.174 |
| B2 P3 reduced | 1/3/3 | 36.820M | 35.472 | 0.073 | -0.203 | -0.296 |
| B4 P4 reduced | 3/1/3 | 36.820M | 35.573 | 0.021 | -0.102 | -0.195 |
| B5 all reduced | 1/1/1 | 35.507M | 35.009 | 0.043 | -0.666 | -0.759 |

## Decision

- Keep A1 prune with 3/3/3 blocks when absolute medium AP is the priority.
- B1 is the best reduced variant by mean AP. It removes 1.763M parameters
  (4.57%) from A1 mask while losing only 0.082 AP on the two-seed mean.
- B4 is the stability-oriented reduced variant. Its two seeds differ by only
  0.021 AP, but its mean is 0.020 AP below B1.
- Do not reduce P3 first and do not compose reductions across all scales. B2 is
  weaker than B1/B4, and B5 loses 0.666 AP relative to the mask reference.

The unusually high A1 prune seed-42 result makes prune-relative deltas look
harsher than mask-relative deltas. The two references are therefore retained
in the report instead of selecting candidates from one seed alone.
