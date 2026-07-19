# Scale-Decoupled Semantic Reassembly Projector

## Goal

SDSR replaces `MultiScaleProjector` while preserving its detector-facing contract:

- inputs: four stride-16 DINOv3 intermediate feature maps;
- outputs: any ordered subset of P3, P4, and P5 with 256 channels;
- decoder, positional encoding, masks, CDN, and Group DETR remain unchanged.

The design avoids applying the same resampling-and-fusion recipe to every scale. Each output level receives an
operator matched to its role, while the image detail branch only predicts reassembly weights and never injects raw
CNN features into the detector feature maps.

## Architecture

### 1. Layer-channel routing

Each DINOv3 hidden layer is independently normalized and passed through a shared `1x1` projection. Two learnable
channel-wise softmax routers then produce separate semantic sources for P3 and P4:

`S_s(c) = sum_l softmax(a_s(l,c)) * Proj(LN(H_l))(c)`, where `s` is P3 or P4.

This lets shallow and deep transformer layers contribute differently to detail-oriented and semantic levels without
replicating a large projection tower for every layer and scale.

### 2. P3 semantic local reassembly

The P3 source predicts a 3x3 sampling distribution for each of the four stride-8 child positions. A lightweight
stride-8 directional stem supplies horizontal, vertical, local, and dilated image cues to the weight predictor.
The learned weights reassemble neighboring semantic tokens with `unfold` and `pixel_shuffle`.

The predictor starts from the exact `align_corners=False` bilinear phase prior. Its final layer has a tiny non-zero
initialization, so the initial result remains close to bilinear interpolation while both semantic and directional
paths receive gradients from the first optimizer step.

### 3. P4 semantic anchor

P4 retains the native stride-16 ViT resolution. It uses its own routed layer mixture and a small residual depthwise
refinement block. This is the least destructive path and serves as the pyramid semantic anchor.

### 4. P5 anti-aliased phase downsampling

P5 fuses two stride-2 paths:

- a fixed binomial low-pass filter followed by decimation;
- a per-channel learned mixture of the four 2x2 sampling phases.

A high-frequency response predicts the mixing gate. The gate starts biased toward the low-pass path, limiting aliasing
early in training while retaining the ability to preserve useful boundaries later.

## Parameters

For DINOv3-S intermediate channels `[384, 384, 384, 384]`, output channels 256, and P3/P4/P5:

| Projector | Parameters |
|---|---:|
| Existing MultiScaleProjector | 10,629,888 |
| SDSR, detail channels 32 | 556,752 |

SDSR uses approximately 5.2% of the existing projector parameters. This comparison covers the projector only, not the
shared DINOv3 backbone or decoder.

## Configuration

Select SDSR with:

```bash
--projector-scale P3 P4 P5 \
--projector-type sdsr \
--sdsr-detail-channels 32
```

Independent ablation flags are:

```bash
--no-sdsr-use-local-reassembly
--no-sdsr-use-directional-guide
--no-sdsr-use-phase-downsample
```

When local reassembly is disabled, P3 falls back to bilinear interpolation. When phase downsampling is disabled, P5
falls back to depthwise stride-2 plus pointwise projection. Disabling the directional guide leaves a semantic-only
local reassembly predictor.

## First Ablation Matrix

Keep seed, total batch size, epochs, augmentation, DINOv3 features, CDN, Group DETR, and decoder settings aligned.

| ID | Local reassembly | Directional guide | Phase downsample | Purpose |
|---|---:|---:|---:|---|
| MSP | n/a | n/a | n/a | Existing baseline |
| S0 | off | off | off | Lightweight routed pyramid baseline |
| S1 | on | off | off | Isolate semantic P3 reassembly |
| S2 | on | on | off | Measure image-guided boundary recovery |
| S3 | on | on | on | Complete SDSR |

The first decision should be based on COCO AP, APs/APm/APl, parameter count, peak memory, and iteration time. If S1
improves APs but S2 does not, simplify or regularize the directional guide. If S3 hurts AP while S2 helps, tune the P5
gate before changing the P3 path.
