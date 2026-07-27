# Unified Backbone Refinement

## Trainer Design

`tools/train_dinov3_lazystrike_refine.py` now exposes three composable
objectives through `--objectives`: `unirefiner`, `lazystrike`, and `dense`.
Legacy commands remain compatible through automatic objective inference.

- UniRefiner reuses the source implementation for register construction,
  FP/GP/AH filtering, clean-token NCE, register absorption, and SCD through
  `tools/refinement/unirefiner_objective.py`.
- LazyStrike retains box-balanced CLS aggregation, object coverage, and CLS
  flip consistency.
- Dense consistency retains multilevel patch preservation and object/global
  cross-view consistency.
- All active losses share one student, frozen stage-local teacher, optimizer,
  backward pass, checkpoint format, and CSV history.
- Refinement still exports a plain DINOv3 state dictionary and adds no
  inference-time module.

Static checks, legacy objective-inference tests, a three-objective 224-pixel
smoke test, and a 640-pixel batch-2 UniRefiner+dense memory test passed. The
640 test used about 1710 MiB and produced a useful-token ratio of 0.246.

## Experiment Order

1. **Phase A: UniRefiner + dense compatibility.** Start both variants from the
   COCO `balanced_zero3e` UniRefiner checkpoint. Compare dense-only sequential
   refinement against joint UniRefiner+dense refinement. Both use the previous
   standalone dense recipe and BF16. The joint variant uses
   `lambda_unirefiner=0.005`, which balances the initial weighted UniRefiner
   and dense loss magnitudes.
2. **Phase B: UniRefiner + LazyStrike compatibility.** Run only after Phase A
   is analyzed. Compare LazyStrike-only sequential refinement against joint
   UniRefiner+LazyStrike using the v2-conservative LazyStrike recipe.
3. **Phase C: three-objective refinement.** Run only if one pairwise joint
   variant beats both its initialization and sequential control. Separate
   LazyStrike/UniRefiner late-layer supervision from earlier dense losses.
4. Confirm any gain with seed 43 before full COCO. A difference below 0.10 AP
   without matching late-epoch stability is treated as inconclusive.

Phase A is orchestrated on GPU1 by
`run_unified_refinement_phase_a_gpu1.sh`. Each variant runs one refinement
epoch on the same seeded COCO 5k subset, COCO-val-200 token diagnostics, and an
aligned 24-epoch medium detection experiment.

## Phase A Results (2026-07-20)

Both variants completed successfully. Detection values are best-epoch COCO AP
points from aligned 24-epoch medium runs.

| Variant | AP | AP50 | AP75 | APs | APm | APl | Last-5 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| UniRefiner `balanced_zero3e` initialization | 33.88 | 50.96 | 35.95 | **17.09** | **36.58** | 51.94 | 33.46 |
| Uni to dense sequential | 33.91 | 51.21 | 35.78 | 16.78 | 36.35 | 53.04 | 33.38 |
| Uni+dense joint (`lambda_uni=0.005`) | **34.11** | **51.50** | **36.39** | 16.17 | 36.05 | **53.67** | **33.52** |
| Previous standalone dense-only | 34.03 | 50.94 | **36.75** | **17.36** | **36.85** | 51.95 | **33.56** |

The joint objective beat its sequential control by 0.20 AP and the UniRefiner
initialization by 0.23 AP. Relative to sequential it gained 0.30 AP50, 0.62
AP75, and 0.63 APl, but lost 0.61 APs and 0.30 APm. Relative to standalone
dense-only, joint gained only 0.08 peak AP and lost 0.04 last-five mean AP,
while shifting 1.72 points from small/medium-localization behavior toward APl.
This is evidence that the objectives can coexist, but not yet evidence of a
stable additive gain over the strongest component.

Optimization remained healthy. Dense trajectories were almost identical in
the sequential and joint runs. In joint training, SCD activated after step 250,
which raised raw UniRefiner loss from 5.21 to 7.79 as expected; NCE improved
from 4.85 to 4.72 and useful-token ratio rose from 0.262 to 0.282. The loss jump
was therefore scheduled SCD activation rather than divergence.

COCO-val-200 top-100 GT coverage improved from 0.5868 at the UniRefiner start
to 0.5947 sequential and 0.5945 joint. Sequential and joint token diagnostics
were nearly identical, so they do not explain the 0.20 AP difference. FP/GP
ratios from the older initialization diagnostic use fixed thresholds and must
not be compared numerically with the new dynamic-threshold diagnostics.

**Decision:** proceed to Phase B because joint training clearly outperformed
the matched sequential control without objective-level instability. Do not yet
promote Uni+dense as additive or run it on full COCO: its margin over standalone
dense-only is below the predefined 0.10 AP threshold and lacks a last-five-AP
gain. Phase B should retain matched sequential/joint controls and explicitly
track the same small-to-large redistribution.

## Phase B Setup (2026-07-23)

Phase B compares two ways of combining the strongest retained UniRefiner
initialization with LazyStrike:

1. `uni_balanced_to_lazy_v2_seq`: load the COCO `balanced_zero3e` checkpoint,
   then optimize only the LazyStrike v2-conservative objective.
2. `uni_balanced_lazy_v2_joint_u005`: load the same checkpoint and jointly
   optimize LazyStrike plus the UniRefiner objective with
   `lambda_unirefiner=0.005`.

Both variants use the same seeded COCO train-5k subset, two refinement epochs,
the final two DINOv3 blocks, learning rate `5e-6`, and the original
v2-conservative LazyStrike weights (`align=0.25`, `cover=0.15`,
`consistency=0.25`, `distill=1.0`). Both use BF16 so precision is controlled
within the pair. The joint UniRefiner configuration keeps zero-filled register
regions, three proposals, FP/GP filtering, no AH filter, uniformity strength
`0.3`, and SCD weight `0.4` starting at 10% of training.

A 640-pixel, batch-2, one-step joint calibration passed on GPU0 with 1347 MiB
peak memory. Total loss was 0.29925: the weighted LazyStrike terms contributed
about 0.27283 and weighted UniRefiner contributed 0.02642 (about 9% of the
total), so the auxiliary objective is active without dominating refinement.

`run_unified_refinement_phase_b_gpu0.sh` runs each refinement followed by a
COCO-val-200 token diagnostic and aligned 24-epoch medium detection. Results
are collected in `output/unified_refinement/phase_b_summary.tsv`. The queue is
restricted to physical GPU0 and waits whenever its used memory exceeds
4000 MiB.

## Phase B Results (2026-07-24)

The Phase B queue completed successfully. Detection values below are
best-epoch COCO AP points from aligned seed-42, 24-epoch medium runs.

| Variant | AP | AP50 | AP75 | APs | APm | APl | Last-5 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current DINOv3 baseline | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |
| UniRefiner `balanced_zero3e` initialization | 33.88 | 50.96 | 35.95 | **17.09** | 36.58 | 51.94 | 33.46 |
| Standalone LazyStrike v2-conservative | **34.36** | **51.71** | 36.37 | 16.77 | 36.48 | 52.97 | **33.71** |
| Uni to LazyStrike sequential | 34.04 | 51.23 | **36.66** | 16.24 | **36.86** | **53.32** | 33.59 |
| Uni+LazyStrike joint (`lambda_uni=0.005`) | 33.65 | 50.87 | 35.97 | 15.73 | 36.13 | 52.58 | 33.49 |

Sequential refinement beat joint refinement by 0.38 peak AP, including
0.36 AP50, 0.69 AP75, 0.51 APs, 0.72 APm, and 0.74 APl. The peak difference
overstates run-level separation, however: sequential led by only 0.10 in
last-five mean AP, while joint led by 0.05 in last-ten mean AP. Joint was
slightly stronger around epochs 15--18, but sequential rose more strongly over
epochs 19--24 and finished 0.40 AP higher.

Sequential refinement retained a small 0.16 peak-AP gain over its UniRefiner
initialization, concentrated in AP75 (+0.71), APm (+0.28), and APl (+1.38), at
the cost of 0.85 APs. It did not beat standalone LazyStrike v2-conservative:
standalone remained 0.33 AP and 0.12 last-five AP higher. Sequential did shift
quality toward stricter localization and medium/large objects, gaining 0.29
AP75, 0.38 APm, and 0.35 APl over standalone LazyStrike while losing AP50 and
APs.

Optimization was numerically healthy. The sequential and joint LazyStrike
losses were almost identical in all three measured intervals:

| Steps | Sequential Lazy loss | Joint Lazy loss | Weighted Uni loss |
|---|---:|---:|---:|
| 1--500 | 0.21832 | 0.21828 | 0.02651 |
| 501--2500 | 0.12317 | 0.12327 | 0.04022 |
| 2501--5000 | 0.08499 | 0.08505 | 0.03968 |

SCD activated at step 500, increasing raw UniRefiner loss from about 5.30 to
about 8.04. Useful-token ratio rose from 0.252 to 0.317 and NCE improved only
modestly from 4.94 to 4.87. This indicates objective coexistence without
divergence, but no positive transfer to the LazyStrike objective.

The two COCO-val-200 CLS-score maps were nearly indistinguishable.
Top-100 box coverage was 0.85277 sequential and 0.85261 joint; corresponding
small/medium/large coverage differences were all below 0.001. Joint reduced
FP/GP proxy ratio only from 0.22840 to 0.22751, while LAST-vote top-1
Point-in-Box fell from 0.425 to 0.365 and top-10 from 0.88 to 0.86. These proxy
changes are too small and internally mixed to support an additive
representation-quality claim.

Joint refinement also took about 14.1 minutes versus 5.9 minutes sequential
because it runs the UniRefiner teacher/register path, while adding no
inference-time module. Given the absent AP gain, this extra refinement cost is
not justified for the current combination.

**Decision:** do not enter the original three-objective Phase C yet. Phase B
does not satisfy the preregistered condition that a pairwise joint variant beat
both its initialization and matched sequential control. The most defensible
detector checkpoint remains standalone LazyStrike v2-conservative for peak AP,
while the Phase A Uni+dense joint and Phase B sequential checkpoints remain
useful localization/large-object alternatives. If combination work continues,
first test a narrowly scoped no-SCD or early-only UniRefiner auxiliary and
measure per-objective gradient cosine; broad three-objective weight searches
are not warranted by these results.
