# ICLR 2027 H-Validation Protocol Addendum

This addendum is a narrow validation protocol based on frozen commit
`7287b89ac3af7c33e616b90be411b25b856ab61d`. It does not modify or supersede
the `paper/iclr27-freeze-v1` paper freeze. The H recipe in
`paper/ICLR27_RECIPE.json` remains unchanged.

## Pre-registered axes

The H-Validation matrix permits exactly two scientific axes:

- projector: `sdsr_v40` (`sdsr_v40_p4_learnable`) or `msp` (the original
  `MultiScaleProjector`);
- exposed detector levels: `p4` or `p345`.

The paper launcher exposes the second axis only as
`--scale-interface {p4,p345}`. It resolves deterministically:

| Interface | `projector_scale` | `dec_level_n_points` | MSP C2f blocks |
|---|---|---|---|
| `p345` | `P3 P4 P5` | `2 3 1` | `3 3 3` |
| `p4` | `P4` | `3` | `3` |

The P4 sampling count is selected from the P4 component of the frozen
P3/P4/P5 mapping. It is not separately tuned. The launcher does not expose
arbitrary levels, sampling-point lists, C2f depths, learning rates, or other
scientific overrides. P4-only is limited to the 24-epoch `analysis` track.

## H0-H5 preregistration

| ID | Projector | Interface | Seed | Detector init seed | Purpose |
|---|---|---|---:|---:|---|
| H0 | SDSR-v40 | `p345` | 43 | 1043 | clean reproduction of the development H result |
| H1 | MSP | `p345` | 43 | 1043 | seed43 H-LR projector control |
| H2 | MSP | `p4` | 44 | 1044 | seed44 scale-interface control |
| H3 | MSP | `p345` | 44 | 1044 | seed44 scale-interface control |
| H4 | SDSR-v40 | `p4` | 44 | 1044 | seed44 scale-interface control |
| H5 | SDSR-v40 | `p345` | 44 | 1044 | seed44 scale-interface control |

All six runs retain every analysis invariant from the paper experiment
contract: H detector/encoder learning rates, 24 epochs, LR drop 20, Group 6,
DN25 with CDN budget 300, Dense O2O without MixUp, register border 0, EMA
0.993/tau100, and random multi-scale training through the final epoch.

No P3-only, P5-only, P34, P45, architecture search, LR search, or seed45 run
is part of this protocol.
