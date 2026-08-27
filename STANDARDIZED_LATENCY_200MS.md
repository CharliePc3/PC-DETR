# Local 200 ms Buffered Latency Comparison

Date: 2026-08-24

## Protocol

- GPU: one otherwise-idle NVIDIA GeForce RTX 5090 (GPU 0).
- Batch size: 1.
- Input: synthetic tensor; preprocessing and postprocessing excluded.
- Runtime: PyTorch eager.
- Warmup: 20 forwards.
- Measurement: 100 forwards per precision.
- Buffer: 200 ms after every synchronized forward; the buffer is excluded from
  the reported latency.
- Timing: CUDA events around model forward only.
- Precision: FP32 with TF32 enabled, and PyTorch AMP FP16.
- FLOPs: `torch.utils.flop_counter`; multiply and add count as two operations.
  This reproduces RF-DETR Medium's paper value (78.8249 measured versus 78.8
  reported at 576).

## Aligned 640 Input

| Model | Parameters | GFLOPs | FP32 mean | FP32 p50 | AMP mean | AMP p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF-DETR Medium | 33.687M | 100.204 | 14.222 ms | 13.525 ms | 19.161 ms | 17.777 ms |
| DEIMv2 DINOv3-L | 32.551M | 146.011 | 26.284 ms | 23.597 ms | 30.661 ms | 27.615 ms |
| RF-DETR-DINOv3 S2 | 36.813M | 172.985 | 32.892 ms | 31.826 ms | 28.791 ms | 28.600 ms |

RF-DETR is loaded at its native 576 architecture and receives a 640 tensor via
the backbone's positional interpolation. The other two checkpoints are native
640 models.

## RF-DETR Native Resolution

| Model | Input | GFLOPs | FP32 mean | FP32 p50 | AMP mean | AMP p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF-DETR Medium | 576 | 78.825 | 14.772 ms | 13.682 ms | 20.216 ms | 21.483 ms |

The buffered protocol measures isolated request latency rather than sustained
throughput. On this RTX 5090, the 200 ms idle period can lower GPU clocks, so a
smaller input is not guaranteed to have a lower mean in a single run. Median
and distribution statistics should be retained with the mean.

## Artifacts

- RF-DETR-DINOv3: S2 projector, official DINOv3 backbone, scale routing, four
  decoder layers, checkpoint `integrated_refinement_s2_denseo2o/official_seed42`.
- DEIMv2: `deimv2_dinov3_l_coco_24e_budget.yml`, `best_stg2.pth`, EMA weights.
- RF-DETR: official `rf-detr-medium.pth`.

## Interpretation Limits

These results apply the paper's 200 ms buffering idea but are not reproductions
of its TensorRT latency. The paper uses TensorRT 10.4, FP16, CUDA Graphs, and a
T4. No TensorRT executable or Python package is installed locally, and the
current RF-DETR-DINOv3 export path is incompatible with shared iterative bbox
refinement because it returns one decoder state where four are expected.

The three existing environments also use different PyTorch builds:

- RF-DETR-DINOv3: PyTorch 2.11.0, CUDA 13.0.
- DEIMv2: PyTorch 2.11.0 development build, CUDA 12.8.
- RF-DETR: PyTorch 2.8.0, CUDA 12.8.

Consequently, this is a useful local engineering comparison, but a paper table
should first fix RF-DETR-DINOv3 export and benchmark all three FP16 engines with
one TensorRT/CUDA stack. AMP being slower for DEIMv2 and RF-DETR here is an
eager-runtime result and should not be interpreted as FP16 hardware being
slower than FP32.
