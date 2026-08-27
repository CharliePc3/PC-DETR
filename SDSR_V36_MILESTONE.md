# SDSR-v36 Medium Milestone

Date: 2026-08-03

## Outcome

SDSR-v36 is a functionally equivalent execution reorganization of SDSR-v23.
It preserves the full grouped 3x3 detail operator that produced the best SDSR
accuracy, while packing repeated low-pass, shallow depthwise, and normalization
operations across the four DINOv3 hidden states.

The converted v23 checkpoint independently reproduced the following medium
COCO metrics with the current source:

| Model | AP | AP50 | AP75 | APS | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP low-LR finish, no scale routing | 35.898 | 53.019 | 38.085 | 17.787 | 38.004 | 55.208 |
| Tuned MSP low-LR finish | 36.142 | 53.620 | 38.494 | 17.293 | 38.823 | 55.469 |
| **SDSR-v36 / v23 weights** | **36.059** | **53.232** | **37.979** | **17.759** | **38.670** | **55.978** |

SDSR-v36 therefore clears the 36 AP target and improves over the no-routing MSP
reference by 0.161 AP. It remains 0.083 AP below the separately tuned routed
MSP low-LR finish, so the two MSP rows must not be conflated.

## Native training controls

The v34 and v36 execution variants were also trained natively for 28 epochs
with the same seed 43, detector initialization seed 1043, and learning-rate
drop at epoch 22. The matching v23 seed-43 run is the correct control for this
table; it must not be compared directly with the 36.059-AP seed-42 checkpoint
used for conversion above.

| Native seed-43 model | Best epoch | AP | AP50 | AP75 | APS | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SDSR-v23 | 24 | 35.778 | 52.630 | 37.840 | 18.065 | 38.806 | 54.535 |
| SDSR-v34 | 27 | 35.635 | 52.710 | 37.830 | 17.443 | 38.331 | 52.776 |
| SDSR-v36 | 27 | 35.501 | 52.297 | 37.815 | 17.150 | 38.390 | 54.325 |

The native runs are close but not bitwise training-equivalent: packing changes
the order of floating-point accumulation and therefore the optimization path.
Relative to the same-seed v23 control, v34 is lower by 0.143 AP and v36 by
0.277 AP. This does not invalidate the checkpoint conversion, whose forward
equivalence and independent evaluation are measured separately.

The recommended workflow is therefore train with v23 and convert to v36 for
evaluation or deployment. V34 remains an optional lighter approximation, but
factorizing the deepest grouped 3x3 detail operator particularly reduced APL.

## Efficiency

All end-to-end measurements below were collected consecutively on one RTX 5090
with the same eager PyTorch benchmark, 640x640 input, 80 warmup iterations, and
300 measured iterations.

| Model | Parameters | GFLOPs | FP32 mean | FP32 p50 | AMP mean | AMP p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 40.574M | 90.564 | 23.415 ms | 22.983 ms | 29.330 ms | 28.487 ms |
| SDSR-v23 | 35.757M | 88.644 | 24.354 ms | 23.587 ms | 29.682 ms | 28.925 ms |
| SDSR-v34 | 35.686M | 88.618 | 23.450 ms | 23.064 ms | 30.098 ms | 28.562 ms |
| **SDSR-v36** | **35.757M** | **88.646** | **22.229 ms** | **21.674 ms** | **29.871 ms** | **28.237 ms** |

Relative to MSP, SDSR-v36 reduces total parameters by 11.9% and measured total
FLOPs by 2.1%. Its FP32 mean latency is 5.1% lower. AMP mean latency is 1.8%
higher, while AMP median latency is 0.9% lower. AMP peak memory was 246.44 MiB
for SDSR-v36 and 253.90 MiB for MSP in this benchmark.

The isolated projector remains slower than MSP at batch size one (3.134 ms
versus 2.783 ms), but its lower parameter and FLOP cost makes the complete
detector competitive. Both module-level and end-to-end numbers should be
reported.

## Equivalence

With the same seed, newly initialized SDSR-v23 and SDSR-v36 projectors have:

- identical RNG state after construction;
- identical P3, P4, and P5 outputs in the direct test;
- the same 5,812,608 projector parameters;
- the same full grouped 3x3 deepest-detail capacity.

The checkpoint conversion verification measured a maximum projector output
difference of `2.86102294921875e-06`. The converted detector loaded with
`strict=True`, and an independent COCO evaluation reproduced rounded AP=0.361.

## Artifacts

Converted checkpoint:

```text
output/coco_medium_v36_converted_from_v23_ap36p059/checkpoint_best_regular.pth
```

Independent evaluation log:

```text
eval_coco_medium_v36_converted_ap36p059_current_code_gpu2.log
```

Same-pass benchmark records:

```text
output/benchmarks/samepass_gpu2_msp.json
output/benchmarks/samepass_gpu2_v23.json
output/benchmarks/samepass_gpu2_v34.json
output/benchmarks/samepass_gpu2_v36.json
```

## Conversion

```bash
PYTHONPATH=src \
python tools/convert_sdsr_v23_to_v36.py \
  --input output/coco_medium_seed42_detinit1042_iterref_scaleroute_v23_exactp3p4_twobasis_deepgroupdetail16_p5_lrd22_frome18_e28/checkpoint_best_regular.pth \
  --output output/coco_medium_v36_converted_from_v23_ap36p059/checkpoint_best_regular.pth
```

The converter removes optimizer and scheduler state because tensor packing
changes optimizer parameter identities. The converted checkpoint is intended
for evaluation, inference, or loading as model weights, not direct optimizer
resume.

Native full-run records:

```text
output/coco_medium_seed43_detinit1043_iterref_scaleroute_v34_exactp3p4_packedtwobasis_deepseparablegroupdetail16_p5_lrd22_e28_full
output/coco_medium_seed43_detinit1043_iterref_scaleroute_v36_exactp3p4_packedtwobasis_deepfullgroupdetail16_p5_lrd22_e28_full
```
