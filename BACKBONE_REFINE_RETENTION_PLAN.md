# Backbone Refinement Retention Plan

## Fixed reference

All runs retain the integrated S2 MultiScaleProjector, iterative box refinement,
legacy scale routing, Group DETR 6, tuned CDN, and enhanced Dense O2O with
CopyBlend N=1. Data seed 42 and detector initialization seed 1042 are fixed.

The paired unprotected references are:

| Backbone | AP | Last-5 AP |
|---|---:|---:|
| LazyStrike v2 | 36.043 | 35.551 |
| Dense consistency | 36.228 | 35.637 |

## Stage 1

| Variant | Protected parameters | LR scale | L2-SP coefficient |
|---|---|---:|---:|
| lazy_lr01 | blocks 10-11 + final norm | 0.1 | 0 |
| dense_lr01 | blocks 8-11 + final norm | 0.1 | 0 |
| lazy_anchor1e3 | blocks 10-11 + final norm | 1.0 | 0.001 |
| dense_anchor1e3 | blocks 8-11 + final norm | 1.0 | 0.001 |

The LR and L2-SP factors are tested separately. A combined run is justified
only if at least one single factor improves peak AP or last-five AP without
worsening the refinement-specific metric profile.

The controller waits until GPU0/1 memory usage is at most 2500 MiB before each
run, snapshots the source once, verifies refinement checkpoint hashes, resumes
valid partial runs, and writes a compact TSV summary.
