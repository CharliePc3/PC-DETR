# Dense Consistency Optimization

## Motivation

The first standalone `dense_only` run reached 34.03 AP and improved AP75 by
1.51 points over the current base, with gains concentrated on small and medium
objects. The simultaneous CLS+dense hybrid dropped to 33.72 AP. Its final-block
teacher-distillation loss was about 5.5 times the standalone dense value,
showing that CLS aggregation and dense invariance competed at block 11.

## Implementation Change

- Train only blocks 8, 9, and 10.
- Freeze block 11 and the final normalization layer.
- Measure dense losses at blocks 8 and 11.
- Use distill/global layer weights `0.25/0.75` to preserve final output.
- Use object layer weights `0.75/0.25` to focus learning on detector-consumed
  block 8 features.
- Increase teacher preservation from `1.0` to `1.5`.
- Use object consistency weight `0.20`.
- Compare global consistency `0.02` against object-only `0.0`.
- Keep all CLS alignment, coverage, and CLS flip losses disabled.

The refinement remains offline and adds no detection inference module.

## Experiments

| Variant | Initialization | Global weight | Question |
|---|---|---:|---|
| `base_g002` | Original DINOv3 | 0.02 | Does localized dense refinement improve the standalone recipe? |
| `v2seq_g002` | LazyStrike v2 e2 | 0.02 | Can dense refinement add localization without destroying v2? |
| `v2seq_objonly` | LazyStrike v2 e2 | 0.00 | Is global consistency causing AP50/large-object regression? |

All variants use 5,000 shuffled COCO train2017 images, one refinement epoch,
seed 42, COCO-val200 token diagnostics, and aligned 24-epoch medium detection.
They run serially on physical GPU3. Runtime status is written to
`dense_local_sequential_gpu3_queue_nohup.log`; consolidated results are written
to `output/lazystrike_dense_optimization/summary.tsv`.

## Success Criteria

- `base_g002` should retain the previous AP75/APs/APm gains while recovering
  AP50 or APl.
- A sequential variant must exceed the v2 e2 reference of 34.36 AP, or show a
  repeatable complementary gain large enough to justify another seed.
- Block-11 distillation should remain close to standalone dense levels and far
  below the failed hybrid trajectory.

## Results (2026-07-19)

All three GPU3 jobs completed successfully. Detection values are COCO AP points
at the best epoch of each aligned 24-epoch medium run.

| Variant | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| Current base | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 |
| Previous dense-only | **34.03** | 50.94 | **36.75** | **17.36** | **36.85** | 51.95 |
| LazyStrike v2 e2 | **34.36** | **51.71** | 36.37 | 16.77 | 36.48 | 52.97 |
| `base_g002` | 33.87 | 51.36 | 36.38 | 16.32 | 36.79 | 52.73 |
| `v2seq_g002` | 34.01 | 51.06 | 36.32 | 17.41 | 36.62 | 52.69 |
| `v2seq_objonly` | 33.85 | 50.82 | 36.06 | 16.87 | 36.80 | 52.01 |

The last-five-epoch mean AP/AP75 was 33.16/35.15 for current base,
33.56/35.84 for previous dense-only, 33.71/35.87 for v2 e2, 33.58/35.98 for
`base_g002`, 33.61/35.88 for `v2seq_g002`, and 33.53/35.66 for
`v2seq_objonly`.

### Refinement Behavior

- In `base_g002`, object consistency fell from 0.213 to 0.195 and global
  consistency from 0.166 to 0.157 between the first and last 250 steps.
- Block-11 distillation stayed near 0.0033 in all localized variants, compared
  with about 0.024 in the failed simultaneous hybrid. Freezing block 11 and the
  final norm successfully removed final-layer drift.
- `v2seq_g002` and `v2seq_objonly` produced nearly identical token diagnostics.
  The small global term nevertheless improved peak AP by 0.16, AP75 by 0.26,
  APs by 0.54, and APl by 0.68 over object-only. Global consistency was not the
  source of the previous hybrid regression.

### Interpretation

- Relative to previous dense-only, `base_g002` recovered 0.42 AP50 and 0.78
  APl, but lost 0.37 AP75 and 1.04 APs. Freezing the final stage shifted the
  representation toward stable coarse/large-object recognition and away from
  the small-object localization gain that made dense-only useful.
- Although `base_g002` had a lower peak AP, its last-five-epoch mean AP75 was
  the strongest of the compared runs. The localized objective is stable but too
  conservative to maximize peak AP.
- Sequential refinement did not beat its v2 initialization: `v2seq_g002` lost
  0.35 peak AP and 0.10 mean AP. It gained 0.64 peak APs but lost 0.65 AP50.
  Current evidence does not support stacking this dense recipe after v2.
- Token coverage and flip diagnostics changed only slightly for `base_g002` and
  almost not at all between the two sequential variants, so they cannot select
  among these close detection checkpoints.

### Decision

Do not continue the fully frozen block-11 recipe or the current v2 sequential
recipe. If one final dense optimization round is pursued, use a midpoint between
previous dense-only and `base_g002`: train blocks 8-11, freeze final norm, apply
a much lower explicit LR to block 11, use object layer weights around 0.6/0.4,
and retain a small global term (`0.02`). This targets the old dense AP75/APs
gain while preserving the new recipe's AP50/APl recovery. Validate that single
candidate before spending another seed or full-COCO budget.

## Midpoint Candidate (queued 2026-07-19)

The single follow-up candidate starts from the original DINOv3-small weights,
trains blocks 8-11, and freezes the final normalization layer. Block LR scales
are `0.512/0.64/0.8/0.2` at a base LR of `2.5e-6`, giving block 11 a much
smaller `5e-7` LR. Dense object weights are `0.6/0.4` for blocks 8/11 and the
global consistency weight remains `0.02`. Refinement, COCO-val-200 diagnostics,
and aligned 24-epoch medium detection are orchestrated by
`run_dense_midpoint_candidate_gpu3.sh`.

Static checks, explicit optimizer-LR verification, and a one-step CPU smoke
test passed. The GPU3 queue was launched as PID `1573962`; it waits until used
memory is at most 4000 MiB before starting each stage. Progress is recorded in
`dense_midpoint_candidate_gpu3_queue_nohup.log`.

### Midpoint Result (2026-07-20)

The queue completed successfully. The best aligned detection checkpoint was
epoch 20:

| Variant | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| Previous dense-only | **34.03** | 50.94 | **36.75** | **17.36** | **36.85** | 51.95 |
| Frozen-block-11 `base_g002` | **33.87** | **51.36** | **36.38** | **16.32** | **36.79** | 52.73 |
| Midpoint candidate | 33.86 | 50.98 | 36.15 | 15.92 | 36.71 | **53.20** |

The midpoint's last-five-epoch mean AP was 33.72, compared with 33.58 for
`base_g002` and 33.56 for previous dense-only. Its peak was therefore not
caused by a single isolated late spike, but its benefit was concentrated on
large objects rather than the desired AP75/small-object behavior.

Refinement remained controlled: first-to-last-250-step object consistency was
0.19061 to 0.17621, global consistency was 0.16575 to 0.15561, and block-11
distillation was 0.00326 to 0.00336. The explicit `5e-7` block-11 LR prevented
the large final-layer drift seen in the failed hybrid. Nevertheless, relative
to `base_g002`, the midpoint lost 0.38 AP50, 0.23 AP75, 0.40 APs, and 0.08 APm
while gaining 0.47 APl. COCO-val-200 patch-score coverage also fell from
0.5881 to 0.5844 overall and from 0.3057 to 0.2926 for medium boxes.

This rejects the hypothesis that mildly unfreezing block 11 is sufficient to
combine the old dense-only localization gain with the frozen recipe's stable
semantics. The old dense-only signature likely depends on its complete stronger
recipe (`global=0.05`, `object=0.15`, `distill=1.0`, `0.35/0.65` layer weights,
trainable final norm), rather than block-11 adaptation alone. Stop tuning the
same block-11 LR axis; retain previous dense-only as the standalone dense
reference and LazyStrike v2 e2 as the overall backbone-refinement reference.
