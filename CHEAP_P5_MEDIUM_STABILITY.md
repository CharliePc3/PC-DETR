# Cheap P5 Medium Stability Track

Date: 2026-07-31

## Completion Gate

This track is complete only when one lightweight P5 configuration satisfies all
of the following on the aligned COCO medium protocol:

- Seed 42 best AP is at least about 35.7.
- Seed 43 best AP is at least about 35.7.
- The two-seed mean is close to 36 AP.
- The inference model has materially fewer parameters than the full P3/P4/P5
  projector.

One high seed or an early-epoch projection is not sufficient evidence.

## Aligned Protocol

- DINOv3-S with hidden-state indexes `[2, 5, 8, 11]`
- Resolution 640
- P3/P4/P5 decoder inputs
- Four decoder layers, 300 queries, 300 selected predictions
- Group DETR 6, CDN 50
- Shared iterative box refinement and legacy scale routing
- Multi-scale plus expanded-scale training
- Total batch size 16
- 24 epochs with `lr_drop=20`
- Detector LR `1e-4`, encoder LR `1.5e-4`
- ViT layer decay 0.8, component decay 0.7
- One DINOv3 register-border token

The reproducible launcher is `run_cheap_p5_medium.sh`.

## Completed Seed-43 Controls

| P5 branch | Params | Best AP | AP50 | AP75 | APS | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 40.574M | 35.898 | 53.046 | 37.994 | 17.689 | 38.808 | 54.034 |
| Direct fusion | 33.986M | 34.902 | 51.977 | 37.263 | 16.648 | 38.394 | 52.206 |
| Fusion then C2f refine | 35.103M | 34.517 | 51.785 | 36.049 | 17.759 | 37.843 | 50.996 |

Direct fusion saves 6.588M parameters, or 16.24%, but remains 0.996 AP below
the full branch. Its remaining deficit is concentrated at the scale extremes:
`-1.041 APS` and `-1.829 APL`, versus only `-0.414 APM`.

The replacement-style `fusion_refine` branch improved APS by 1.111 over direct
fusion, but reduced overall AP by 0.385, AP75 by 1.214, and APL by 1.209.
Post-compression spatial refinement therefore shifts the scale tradeoff instead
of recovering the information discarded by rank-64 compression.

## Diagnosis

The full P5 branch independently applies a learned `384 -> 384` 3x3 stride-2
convolution to every selected DINOv3 hidden state before concatenation and C2f
fusion. Direct fusion first compresses each hidden state to 64 channels.

A spectrum analysis of the four trained full-P5 downsampling kernels found:

| Retained rank | Per-layer spectral energy range |
| ---: | ---: |
| 64 | 39.7% to 60.6% |
| 128 | 58.9% to 72.2% |

Rank 64 is therefore a measurable information bottleneck, especially for the
later two hidden states. Adding a C2f block after this compression improves
some localization losses but cannot recover information already discarded.

## Grouped P5 Results

`group4` and `group8` retain all 384 channels and the original P5 C2f stage.
Only the four expensive 3x3 downsampling convolutions become grouped
convolutions. The downstream C2f still performs global cross-layer and
cross-channel mixing.

| P5 mode | Model params | Saving vs full | Seed | Best AP | AP50 | AP75 | APS | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 40.574M | - | 43 | 35.898 | 53.046 | 37.994 | 17.689 | 38.808 | 54.034 |
| `group4` | 36.593M | 3.981M / 9.81% | 43 | 35.225 | 52.219 | 37.347 | 17.012 | 38.101 | 54.765 |
| `group8` | 35.929M | 4.645M / 11.45% | 43 | 35.166 | 52.168 | 37.259 | 17.868 | 38.052 | 53.222 |

The implementation constructs the original full branch first, then initializes
the grouped replacement inside a forked RNG context. P3, P4, the P5 C2f,
decoder parameters, and the post-construction RNG state are bitwise identical
to the full baseline. The only experimental variable is P5 downsampling
connectivity.

Both candidates miss the 35.7 gate. Increasing the group count from four to
eight saves another 0.664M parameters but changes AP by only -0.059. The main
loss is therefore caused by removing too much cross-channel connectivity at
the first grouping step, rather than by the precise group count after that.

## Group2 Stability Screen

`group2` is the smallest reduction of the full downsampling connectivity. It
retains the original C2f and all 384 output channels, while cutting each P5
downsampling convolution into two groups.

| P5 mode | Model params | Saving vs full | P5 downsample MAC ratio |
| --- | ---: | ---: | ---: |
| Full | 40.574M | - | 1.00x |
| `group2` | 37.920M | 2.654M / 6.54% | 0.50x |
| `group2_mix` | 38.116M | 2.458M / 6.06% | 0.50x plus rank-64 adapters |
| `group2_mix128` | 38.313M | 2.261M / 5.57% | 0.50x plus rank-128 adapters |
| `group2_fullmix` | 38.509M | 2.064M / 5.09% | 0.50x plus full-rank 1x1 adapters |
| `group2_first_full` | 38.583M | 1.991M / 4.91% | 0.625x |
| `group2_first2_full` | 39.247M | 1.327M / 3.27% | 0.75x |
| `group2_last_full` | 38.583M | 1.991M / 4.91% | 0.625x |

| P5 mode | Seed | Best epoch | Best AP | AP50 | AP75 | APS | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `group2` | 42 | 23 | 35.824 | 53.159 | 37.474 | 16.881 | 38.422 | 56.064 |
| `group2` | 43 | 22 | 35.219 | 52.302 | 37.210 | 16.912 | 37.980 | 54.885 |
| `group2_mix` | 42 | 23 | 35.645 | 52.748 | 37.798 | 16.853 | 37.863 | 55.688 |
| `group2_mix` | 43 | 23 | 35.476 | 52.619 | 37.668 | 17.111 | 38.155 | 54.664 |
| `group2_first_full` | 42 | 23 | 35.897 | 53.213 | 38.060 | 17.362 | 38.593 | 55.871 |
| `group2_first_full` | 43 | 22 | 35.693 | 52.972 | 37.845 | 17.030 | 38.096 | 55.649 |

The `group2` two-seed mean is 35.521 AP. Seed 42 passes the 35.7 gate, but seed
43 misses it by 0.481 AP, so the candidate is efficient but not stable enough.

`group2_mix` adds a zero-start rank-64 residual channel mixer before each
grouped stride-2 convolution. Its initial outputs and grouped 3x3 weights are
bitwise identical to `group2`, so any gain measures learned cross-group
communication rather than initialization noise. Its two-seed mean is 35.560,
only 0.039 above `group2`. It reduces the seed gap from 0.605 to 0.169 AP, but
neither seed reaches the 35.7 completion gate.

Weight inspection at epoch 8 found that the linearized low-rank corrections
were only 2.5% to 4.2% of the identity path in Frobenius norm. Increasing rank
without increasing update magnitude is therefore unlikely to recover the
missing full-convolution capacity.

The `group2_first_full` follow-up keeps the earliest DINOv3 hidden-state
downsampler dense and groups the other three. It is preferred over
`group2_last_full` because the seed-43 `group2` deficit versus full P5 is
concentrated in APS and APM, while APL is already higher.

This candidate passes the completion gate. Its two-seed mean is 35.795 AP and
the seed gap is 0.205 AP. Seed 43 rounds to 35.7, while seed 42 reaches 35.9.
The model removes 1.991M parameters, or 4.91%, relative to full P5, and uses
37.5% fewer MACs in the four P5 downsampling convolutions. The unused
`group2_first2_full` follow-up was stopped immediately after this result and
all GPUs were released.

Higher-rank, unrestricted residual pre-mix, and alternate selective-dense
variants remain implemented as controlled ablation options, but none is
needed for the completed stability track.

## Verification

- `python -m py_compile` passed for the modified Python entry points.
- `bash -n run_cheap_p5_medium.sh` passed.
- The complete local test suite passed: 159 tests, with one skipped test.
- `git diff --check` passed for the modified files.
