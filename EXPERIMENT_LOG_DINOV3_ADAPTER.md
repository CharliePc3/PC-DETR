# RF-DETR-DINOv3 Adapter Experiment Log

Date: 2026-07-10

## Baselines Already Completed

| Experiment | Setup | Best AP50:95 |
| --- | --- | ---: |
| RF-DETR Medium official eval | official `rf-detr-medium.pth`, COCO val, `maxDets=100` | 54.5 |
| DINOv3 swap + RF-DETR detector weights | P4, 576, multi-scale, expanded scales, EMA, 24e | 50.86 |
| DINOv3 + reinit projector + load decoder/head | P4, 576, multi-scale, expanded scales, EMA, 24e | 50.49 |
| DINOv3 + reinit projector/decoder/head | P4, 576, multi-scale, expanded scales, EMA, 24e | 48.99 |
| DINOv3 swap + RF-DETR detector weights | P4, 576, no multi-scale, no expanded scales, no EMA, 24e | 50.14 |

Working interpretation:

- The main gap is not only projector random init. Reinitializing the projector costs about 0.37 AP versus loading all matchable RF-DETR detector weights.
- Reinitializing decoder/head costs about 1.5 AP versus loading decoder/head.
- Current DINOv3-native setup is still about 3.6 AP behind the official RF-DETR Medium evaluation.

## Code Change: Feature Adapter

Implemented optional DINOv3 feature adapters before the RF-DETR projector:

- `feature_adapter=none`: default, preserves old behavior.
- `feature_adapter=residual_ln_1x1`: per-level LayerNorm + 1x1 conv residual branch, zero-initialized, starts as exact identity.
- `feature_adapter=ln_1x1`: per-level LayerNorm + 1x1 conv without residual, stronger remapping option.

Touched files:

- `src/rfdetr/models/backbone/dinov3.py`
- `src/rfdetr/models/backbone/backbone.py`
- `src/rfdetr/models/backbone/__init__.py`
- `src/rfdetr/models/lwdetr.py`
- `src/rfdetr/config.py`
- `src/rfdetr/main.py`
- `run_coco_subset.py`

Static check:

```bash
python -m py_compile \
  src/rfdetr/models/backbone/dinov3.py \
  src/rfdetr/models/backbone/backbone.py \
  src/rfdetr/models/backbone/__init__.py \
  src/rfdetr/models/lwdetr.py \
  src/rfdetr/main.py \
  src/rfdetr/config.py \
  run_coco_subset.py
```

Result: passed.

## Smoke Test

The default `/opt/anaconda3` Python could not access CUDA inside the managed sandbox. The project environment works when run outside the sandbox:

```text
/home/cpc/.conda/envs/rfdetr-dinov3/bin/python
torch=2.11.0+cu130
cuda_available=True
device_count=4
```

Smoke command:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
PYTHONPATH=$PWD/src:$PWD \
CUDA_VISIBLE_DEVICES=0 \
/home/cpc/.conda/envs/rfdetr-dinov3/bin/python run_coco_subset.py \
  --subset smoke \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --num-workers 0 \
  --resolution 576 \
  --dec-layers 4 \
  --num-queries 300 \
  --num-select 300 \
  --group-detr 13 \
  --projector-scale P4 \
  --multi-scale \
  --expanded-scales \
  --use-ema \
  --detector-pretrain-weights /data/cpc/root/project/RF-DETR/fast/rf-detr-medium.pth \
  --feature-adapter residual_ln_1x1 \
  --feature-adapter-init-scale 1.0 \
  --lr 1e-4 \
  --lr-encoder 1.5e-4 \
  --lr-drop 1 \
  --warmup-epochs 0.0 \
  --output-dir /data/cpc/root/project/RF-DETR-DINOv3/output/smoke_dinov3_adapter_resln1x1_load_rfdetr_detector
```

Smoke result:

- Completed successfully.
- Train set: 16 images.
- Val set: 16 images.
- Trainable parameters: 33.81M.
- Best AP50:95: 0.0010, only used as a pipeline sanity check.
- Output: `output/smoke_dinov3_adapter_resln1x1_load_rfdetr_detector`.

## Running Experiment A1

Purpose: isolate whether a conservative DINOv3 feature adapter helps the strongest 24e COCO-only aligned baseline.

Comparison target:

- Previous baseline: `coco_full_dinov3_swap_rfdetr_medium_detector_p4_ms_exp_ema_lrd24_e24`, best AP 50.86.

Setup:

- COCO full train/val.
- DINOv3-small backbone.
- Load RF-DETR Medium detector checkpoint, excluding only original DINOv2 encoder keys.
- Keep projector/decoder/head loaded when shape-compatible.
- P4, resolution 576.
- Multi-scale and expanded scales enabled.
- EMA enabled.
- CDN disabled.
- 24 epochs.
- 2 GPUs.
- Total batch size target: 16 (`batch_size=8` per GPU, grad accumulation 1).
- New variable: `feature_adapter=residual_ln_1x1`.

Launch command:

```bash
cd /data/cpc/root/project/RF-DETR-DINOv3
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH=$PWD/src:$PWD
export CUDA_VISIBLE_DEVICES=0,1

nohup /home/cpc/.conda/envs/rfdetr-dinov3/bin/torchrun \
  --nproc_per_node=2 \
  --master_port=29631 \
  run_coco_subset.py \
  --subset full \
  --epochs 24 \
  --batch-size 8 \
  --grad-accum-steps 1 \
  --num-workers 2 \
  --resolution 576 \
  --dec-layers 4 \
  --num-queries 300 \
  --num-select 300 \
  --group-detr 13 \
  --projector-scale P4 \
  --multi-scale \
  --expanded-scales \
  --use-ema \
  --detector-pretrain-weights /data/cpc/root/project/RF-DETR/fast/rf-detr-medium.pth \
  --feature-adapter residual_ln_1x1 \
  --feature-adapter-init-scale 1.0 \
  --lr 1e-4 \
  --lr-encoder 1.5e-4 \
  --lr-drop 24 \
  --warmup-epochs 0.0 \
  --eval-max-dets 100 \
  --output-dir /data/cpc/root/project/RF-DETR-DINOv3/output/coco_full_dinov3_adapter_resln1x1_load_rfdetr_detector_p4_ms_exp_ema_lrd24_e24 \
  > coco_full_dinov3_adapter_resln1x1_load_rfdetr_detector_p4_ms_exp_ema_lrd24_e24_nohup.log 2>&1 &
```

Launch status:

- PID: `425238`.
- PID file: `coco_full_dinov3_adapter_resln1x1_load_rfdetr_detector_p4_ms_exp_ema_lrd24_e24.pid`.
- Log: `coco_full_dinov3_adapter_resln1x1_load_rfdetr_detector_p4_ms_exp_ema_lrd24_e24_nohup.log`.
- Started at about 2026-07-10 12:02 Asia/Shanghai.
- Distributed initialized with `world_size=2`.
- Dataset loaded: 118287 train images, 5000 val images.
- Total batch size: 16.
- Trainable parameters: 33.81M.
- First training logs reached epoch 1 iteration 70 without failure.
- GPU usage after entering training: GPU0 about 11.5GB, GPU1 about 10.8GB.
- Early step time after warmup: about 0.44-0.46 s/iter.
- Warning observed: DDP grad stride warning for 1x1 conv parameters. This is a performance warning, not a correctness failure. Current decision: continue the run and judge from AP/time unless it becomes a clear bottleneck.

Status sample:

- 2026-07-10 12:04 Asia/Shanghai: still running, reached epoch 1 iteration about 180.
- GPU usage: GPU0 about 10.8GB, GPU1 about 10.8GB.
- Recent step time: about 0.41-0.46 s/iter.
- 2026-07-10 12:23 Asia/Shanghai: still running on GPU0/GPU1, reached epoch 1 iteration about 2560/7393. A requested B0 DINOv3-only run was not started on GPU0/GPU1 because those GPUs are occupied by this A1 run.

## Planned Experiment B0

Purpose: establish a clean DINOv3-only COCO baseline without RF-DETR detector weights and without feature adapter.

Setup:

- Load only DINOv3-small image pretrained backbone.
- Random-init projector, decoder, and detection heads.
- No feature adapter.
- P4, resolution 576.
- Multi-scale and expanded scales enabled.
- EMA enabled.
- CDN disabled.
- COCO full train/val.
- Proposed length: 60 epochs unless resource pressure requires 50 epochs.
- 2 GPUs, total batch size 16.

Planned command once GPU0/GPU1 are available:

```bash
cd /data/cpc/root/project/RF-DETR-DINOv3
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH=$PWD/src:$PWD
export CUDA_VISIBLE_DEVICES=0,1

nohup /home/cpc/.conda/envs/rfdetr-dinov3/bin/torchrun \
  --nproc_per_node=2 \
  --master_port=29641 \
  run_coco_subset.py \
  --subset full \
  --epochs 60 \
  --batch-size 8 \
  --grad-accum-steps 1 \
  --num-workers 2 \
  --resolution 576 \
  --dec-layers 4 \
  --num-queries 300 \
  --num-select 300 \
  --group-detr 13 \
  --projector-scale P4 \
  --multi-scale \
  --expanded-scales \
  --use-ema \
  --lr 1e-4 \
  --lr-encoder 1.5e-4 \
  --lr-drop 50 \
  --warmup-epochs 0.0 \
  --eval-max-dets 100 \
  --output-dir /data/cpc/root/project/RF-DETR-DINOv3/output/coco_full_dinov3_only_noadapter_p4_ms_exp_ema_lrd50_e60 \
  > coco_full_dinov3_only_noadapter_p4_ms_exp_ema_lrd50_e60_nohup.log 2>&1 &
```

Actual launch:

- Started on GPU3 instead of GPU0/GPU1 because A1 is still using GPU0/GPU1.
- Staged freeze/unfreeze was not enabled. This B0 run is intentionally a clean long-training extension of the DINOv3-only scratch-detector baseline.
- Effective total batch size remains 16 via `batch_size=8` and `grad_accum_steps=2`.
- PID: `742805`.
- PID file: `coco_full_dinov3_only_noadapter_p4_ms_exp_ema_lrd50_e60_gpu3.pid`.
- Log: `coco_full_dinov3_only_noadapter_p4_ms_exp_ema_lrd50_e60_gpu3_nohup.log`.
- Output: `output/coco_full_dinov3_only_noadapter_p4_ms_exp_ema_lrd50_e60`.

Actual command:

```bash
cd /data/cpc/root/project/RF-DETR-DINOv3
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH=$PWD/src:$PWD
export CUDA_VISIBLE_DEVICES=3

nohup /home/cpc/.conda/envs/rfdetr-dinov3/bin/python run_coco_subset.py \
  --subset full \
  --epochs 60 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --num-workers 2 \
  --resolution 576 \
  --dec-layers 4 \
  --num-queries 300 \
  --num-select 300 \
  --group-detr 13 \
  --projector-scale P4 \
  --multi-scale \
  --expanded-scales \
  --use-ema \
  --feature-adapter none \
  --lr 1e-4 \
  --lr-encoder 1.5e-4 \
  --lr-drop 50 \
  --warmup-epochs 0.0 \
  --eval-max-dets 100 \
  --output-dir /data/cpc/root/project/RF-DETR-DINOv3/output/coco_full_dinov3_only_noadapter_p4_ms_exp_ema_lrd50_e60 \
  > coco_full_dinov3_only_noadapter_p4_ms_exp_ema_lrd50_e60_gpu3_nohup.log 2>&1 &
```

Initial status:

- Started at about 2026-07-10 12:31 Asia/Shanghai.
- `detector_pretrain_weights=None`.
- `feature_adapter=none`.
- `world_size=1`.
- Trainable parameters: 33.21M.
- Dataset loaded: 118287 train images, 5000 val images.
- Training config: `grad_accum_steps=2`, `total_batch_size=16`, `dataloader_length=7392`.
- Reached epoch 1 iteration 30 without failure.
- GPU3 memory after entering training: about 10.2GB.

## Final Results: A1 and B0

Both runs completed successfully.

### A1: residual adapter + RF-DETR Medium detector initialization

- Best AP50:95: **0.51095** at epoch 23, EMA.
- Best-epoch metrics: AP50 0.70000, AP75 0.54983, APs 0.28011, APm 0.56632, APl 0.71532.
- Trainable parameters: 33.81M. The four residual LN + 1x1 adapters add about 0.59M parameters.
- Training time: 1 day, 1:59:39 on two GPUs.
- Reference without adapter, with the same RF-DETR detector initialization: 0.50863 AP at epoch 23, EMA.
- Adapter delta: +0.00232 AP (+0.23 AP points). The scale deltas are APs +0.24, APm +0.63, and APl -0.31 points.
- Interpretation: the adapter gives a small positive result, mainly on medium objects and AP75, but it does not explain or close the main DINOv3-to-RF-DETR gap.

### B0: DINOv3-only, random detector, no adapter

- Best AP50:95: **0.49690** at epoch 56, regular weights.
- Best-epoch metrics: AP50 0.68789, AP75 0.53358, APs 0.27356, APm 0.55091, APl 0.70518.
- Best EMA AP50:95: 0.49657 at epoch 53.
- Trainable parameters: 33.21M.
- Training time: 4 days, 3:11:43 on one GPU.
- At epoch 23, EMA AP was 0.48921. The aligned 24-epoch random-detector run reached 0.48988, a difference of only -0.07 AP points.
- Extending training from 24 to 60 epochs therefore adds about +0.77 AP points.
- The run plateaued around 0.49 AP through much of epochs 24-49. The clear second improvement occurred immediately after the scheduled LR drop at epoch 50.
- Final gap to the no-adapter run initialized from the RF-DETR detector is 1.17 AP points; gap to A1 is 1.41 points; gap to official RF-DETR Medium at maxDets=100 (54.5 AP) is 4.81 points.

### Conclusion

- Random projector/decoder/head modules do learn detection from DINOv3 features on COCO, and longer training helps, but insufficient training length accounts for less than one AP point.
- Reusing the RF-DETR detector remains worth about 1.2-1.4 AP points after long COCO training, so detector pretraining is useful but is not the dominant explanation for the official-model gap.
- The lightweight residual adapter is not a high-impact solution in its current form. Its +0.23 point gain is useful as a diagnostic, but it should not become the main architecture direction without repeat-seed confirmation.
- The next high-value work should target the DINOv3 detection interface and optimization: feature-layer selection/statistics, projector inputs, backbone LR and layer decay, followed by full-model Objects365 detection pretraining. A staged freeze/unfreeze run is a secondary optimization ablation, not the primary missing ingredient.

## Comparison: DEIMv2-DINOv3 COCO 24e

The existing DEIMv2 24-epoch run was inspected as a reference for DINOv3 adaptation.

- Initialization: only `dinov3_vits16_pretrain_lvd1689m` was loaded. `resume=None` and `tuning=None`; the spatial adapter, HybridEncoder, decoder, and heads were randomly initialized.
- Hardware: 4 GPUs, local batch size 8, total batch size 32.
- Model size: 32.55M trainable parameters, close to RF-DETR-DINOv3 B0 at 33.21M.
- Final/best EMA result at epoch 23: AP 0.54411, AP50 0.72017, AP75 0.59153, APs 0.35163, APm 0.59848, APl 0.73898.
- Training time: 9:39:58. This is not directly comparable to B0's one-GPU runtime; DEIMv2 used four GPUs and half as many DDP iterations per epoch as the two-GPU RF runs.

This is not a raw DINOv3-to-random-detector baseline. `DINOv3STAs` contains a detection-specific spatial interface:

- DINOv3 semantic features are selected from blocks `[5, 8, 11]`.
- A parallel convolutional Spatial Prior Module extracts local-detail maps at strides 8, 16, and 32 directly from the image.
- Semantic ViT features are resized and concatenated with the spatial-prior maps, then projected and normalized into three feature levels.
- A HybridEncoder applies transformer encoding plus top-down FPN and bottom-up PAN fusion before the three-level deformable decoder.

The training recipe also differs substantially:

- DINOv3 LR is `1.25e-5`; detector LR is `5e-4`, a 40x ratio.
- RF B0 uses detector LR `1e-4`; its top DINOv3 layers receive about `7.35e-5` after component decay, only about a 1.36x detector/backbone ratio.
- DEIMv2 uses 1000-iteration warmup, flat LR through epoch 12, cosine decay, and a final three-epoch no-augmentation stage.
- It uses Mosaic, MixUp, CopyBlend, RandomIoUCrop, strong multi-scale training, 100 denoising queries, a matcher switch at epoch 18, and an EMA checkpoint refresh before the final stage.
- Evaluation resolution is 640 instead of 576. The model reports 96.74 GFLOPs, so the result is not a compute-matched comparison with RF-DETR Medium.

Main lesson: DINOv3 itself is capable of exceeding 54 AP with randomly initialized detection modules. The strongest transferable hypotheses for RF-DETR-DINOv3 are a much lower backbone LR relative to the random detector, a schedule adapted to the 24-epoch budget, and a detection-specific local-detail/multi-level feature interface. The simple residual 1x1 adapter does not provide the spatial prior that DEIMv2's STA supplies.

## Planned B0 Optimization and Feature Ablations

Decision update:

- Do not copy DEIMv2's Mosaic/MixUp/CopyBlend pipeline or its final no-augmentation stage.
- Preserve RF-DETR's current augmentation policy: horizontal flip plus its existing multi-scale/expanded image resizing.
- Preserve RF-DETR's step scheduler rather than introducing DEIMv2's cosine schedule.
- Test optimization, decoder feature levels, and DINOv3 intermediate block indexes as separate variables.

Shared 24-epoch setup:

- DINOv3-S image-pretrained backbone only; projector, decoder, and heads randomly initialized.
- No feature adapter, no CDN, resolution 576, EMA enabled.
- Two GPUs, local batch size 8, total batch size 16.
- Base detector/head LR: `5e-4`.
- Decoder LR after component decay: `3.5e-4`.
- Nominal encoder LR: `2.5e-5`; top DINOv3 block LR after `0.7^2` component decay: about `1.225e-5`.
- ViT layer decay: `0.8`.
- Linear warmup: `0.15` epoch. Step LR drop: epoch 20.

Experiment matrix:

| ID | Projector levels | DINOv3 block indexes | Variable under test |
|---|---|---|---|
| E0 | P4 | `[2,5,8,11]` | Optimized-LR reference |
| E1 | P3/P4/P5 | `[2,5,8,11]` | Stride 8/16/32 decoder inputs |
| E2 | P4 | `[1,4,7,11]` | Earlier intermediate features |
| E3 | P4 | `[3,6,9,11]` | Later intermediate features |

Implementation and validation:

- `run_coco_subset.py` now exposes `--out-feature-indexes`, validates the indexes, logs them, and includes them in automatic output names.
- `DinoV3Backbone` also validates indexes against the actual number of encoder blocks.
- A CPU forward smoke test with indexes `[1,4,7,11]` and P3/P4/P5 produced feature shapes `(1,256,8,8)`, `(1,256,4,4)`, and `(1,256,2,2)` for a 64x64 input, confirming strides 8/16/32.
- These levels are produced by RF-DETR's MultiScaleProjector from stride-16 DINOv3 intermediate features. They do not include DEIMv2's separate raw-image spatial-prior branch.
- Queue script: `run_b0_lr_multilevel_index_ablation.sh`.

Launch status on 2026-07-15:

- Sequential queue started on physical GPU1/GPU2.
- Queue PID: `182129`; PID file: `b0_lr_multilevel_index_ablation_queue.pid`.
- Queue log: `b0_lr_multilevel_index_ablation_queue_nohup.log`.
- E0 loaded only the DINOv3-S image-pretrained checkpoint; `detector_pretrain_weights=None` confirms the detector is random initialized.
- E0 entered distributed training with world size 2, total batch size 16, 33.21M trainable parameters, and the intended `[2,5,8,11]`/P4 configuration.
- At iteration 80, loss remained finite, warmup LR had reached about `3.7e-5`, peak model memory was about 8.45GB per process, and no training error was present.
- E1, E2, and E3 will start automatically after each preceding experiment exits successfully. The queue uses `set -e`, so a failed experiment stops the sequence rather than silently contaminating later results.

## Final Results: B0 Optimization and Feature Ablations

All four experiments completed successfully on 2026-07-19. Metrics below are the best EMA COCO `maxDets=100` results.

| ID | Levels | Block indexes | Epoch | AP | AP50 | AP75 | APs | APm | APl | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | P4 | `[2,5,8,11]` | 23 | 48.72 | 68.10 | 52.24 | 24.62 | 54.21 | 70.42 | 33.21M |
| E1 | P3/P4/P5 | `[2,5,8,11]` | 22 | **51.60** | **70.38** | **55.68** | **29.43** | **56.13** | **72.05** | 42.60M |
| E2 | P4 | `[1,4,7,11]` | 23 | 48.95 | 68.43 | 52.30 | **25.50** | 54.55 | 70.31 | 33.21M |
| E3 | P4 | `[3,6,9,11]` | 23 | 48.96 | 68.39 | **52.44** | 24.64 | **54.57** | **70.68** | 33.21M |

Controlled deltas relative to E0:

- E1: AP +2.88, AP50 +2.28, AP75 +3.44, APs +4.81, APm +1.91, APl +1.63.
- E2: AP +0.23, APs +0.88, APl -0.11.
- E3: AP +0.23, APs +0.03, APl +0.27.
- E2 and E3 differ by only 0.003 AP. One seed cannot establish either index set as superior.

Recall confirms that E1's gain is not only score calibration:

- AR100 increases from 66.13 to 70.19 (+4.06 points).
- Small-object AR increases from 41.07 to 49.23 (+8.16 points).
- AP75 grows more than AP50, indicating improved localization as well as recall.

Optimization interpretation:

- The complete E0 optimization recipe did not improve P4. The old 24-epoch B0 reached 48.99 EMA AP, while E0 reached 48.72 (-0.26 points).
- Before the epoch-20 step drop, E0 also trailed the old B0 throughout most of training. The detector-`5e-4`/low-backbone-LR recipe is therefore not validated for RF's P4 architecture.
- The step drop itself was useful: E0 rose from 47.84 AP at epoch 19 to 48.40 at epoch 20 and 48.72 at epoch 23. However, E0 simultaneously changed detector LR, encoder LR, warmup, and LR drop, so this run cannot isolate which LR component caused the weaker early trajectory.
- A better next optimization control is the original RF learning rates with the same epoch-20 step drop, especially on the stronger P3/P4/P5 architecture.

Feature-level interpretation:

- P3/P4/P5 is the only decisive improvement in this matrix. It closes about half of the old B0-to-official-RF gap and raises a randomly initialized detector above the P4 A1 run that loaded RF detector weights (51.60 versus 51.10 AP).
- The best E1 remains 2.92 AP below official RF-DETR Medium at `maxDets=100` (54.52) and 2.81 below the inspected DEIMv2-DINOv3 run (54.41).
- This is not a parameter-matched Medium comparison: E1 adds 9.38M parameters (+28.2%) and raises measured peak training memory from about 9.3GB to 15.5GB per process.
- P3/P4/P5 supplies 6,804 decoder source positions at resolution 576, versus 1,296 for P4 alone, a 5.25x increase. This explains both the recall gain and the extra memory/compute.

Projector parameter decomposition:

- P4 projector: 1.445M parameters.
- P3 branch: 2.429M parameters.
- P5 branch: 6.756M parameters, dominated by four dense 3x3 stride-2 convolutions.
- A P3/P4 projector is expected to yield about 35.64M total model parameters, only about 7.3% above E0, while removing most of E1's parameter overhead.

Recommended next controls:

1. Run P3/P4 with `[2,5,8,11]` under the E1 recipe to test how much of the 2.88-point gain survives without the expensive P5 branch.
2. Run P3/P4 with original RF learning rates (`lr=1e-4`, `lr_encoder=1.5e-4`) and an epoch-20 step drop to separate the useful late decay from the unsuccessful DEIM-like LR ratio.
3. Keep `[2,5,8,11]` as the default index set. Treat E2's small-object tendency and E3's large-object tendency only as hypotheses unless repeated with multiple seeds.
4. Benchmark params, GFLOPs, and latency before presenting E1 as a comparison to RF-DETR Medium; its AP is currently an accuracy-oriented, higher-cost result.

## P3/P4 LR Controls and Complexity Benchmark

Two P3/P4 experiments were prepared as a controlled learning-rate comparison on 2026-07-20.

Shared configuration:

- P3/P4 projector, DINOv3 blocks `[2,5,8,11]`, resolution 576.
- DINOv3 image-pretrained weights only; random projector, decoder, and heads.
- No adapter, no CDN, no budgeted SA, and no dense O2O.
- RF default horizontal flip plus multi-scale/expanded resizing.
- EMA, 24 epochs, epoch-20 step LR drop, seed 42, total batch size 16.
- Measured trainable parameters: 35.74M.

LR controls:

| ID | Detector/head LR | Nominal encoder LR | Warmup | Purpose |
|---|---:|---:|---:|---|
| P3P4-LR-A | `5e-4` | `2.5e-5` | 0.15 epoch | Match E1 optimization recipe |
| P3P4-LR-B | `1e-4` | `1.5e-4` | none | Restore RF learning rates while retaining the epoch-20 drop |

Runtime records:

- Queue script: `run_p3p4_lr_ablation_queue.sh`.
- The original P3P4-LR-A launch used two GPUs with `batch_size=8` and
  `grad_accum_steps=1`. It was stopped during epoch 6, then resumed from the
  epoch-5 checkpoint on one GPU with `batch_size=8` and
  `grad_accum_steps=2`. The global batch size remained 16.
- The single-GPU launcher now supports `RUN_A=0` or `RUN_B=0`, so either
  control can be run independently without occupying a second GPU.
- Peak training memory for the resumed single-GPU P3P4-LR-A run was about
  14.6GB.
- Each completed run is benchmarked from `checkpoint_best_total.pth`.

P3P4-LR-A completed on 2026-07-27:

| Source | Epoch | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Regular | 23 | 51.01 | - | - | - | - | - | - |
| EMA | 23 | **51.08** | 69.78 | 55.12 | 28.89 | 56.05 | 71.15 | 69.50 |

- The final EMA result is 0.52 AP below the P3/P4/P5 E1 result (51.60), while
  using 35.74M rather than 42.60M trainable parameters.
- Removing P5 therefore retains 82% of the full three-level improvement over
  the P4 E0 baseline: P3/P4 gains 2.36 AP over E0, versus 2.88 AP for
  P3/P4/P5.
- The small-object gain remains substantial: APs improves from 24.62 to 28.89
  (+4.27), compared with 29.43 for P3/P4/P5.

P3P4-LR-B completed on 2026-07-29:

| Source | Epoch | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Regular | 23 | **51.12** | 69.89 | 55.10 | 28.34 | 56.02 | 71.24 | 69.74 |
| EMA | 23 | 51.11 | - | - | - | - | - | - |

- LR-B improves the final best AP over LR-A by only 0.04 points. This is too
  small to establish a learning-rate winner from one seed.
- LR-B is consistently 0.3-0.8 AP ahead through much of the middle and late
  training trajectory, but LR-A nearly catches up after the epoch-20 step
  drop.
- The step drop is the robust optimization result. From epoch 19 to epoch 20,
  regular AP rises by 1.64 points for LR-A and 1.13 points for LR-B; EMA AP
  rises by 0.74 and 0.55 points, respectively.
- LR-B trades APs (-0.55 versus LR-A) for slightly higher AP50, APl, and
  AR100. These scale differences also require a repeat seed before being
  treated as real.

Unified benchmark protocol:

- Script: `tools/benchmark_detector.py`.
- Fixed input: 576x576, batch size 1, RTX 5090, PyTorch eager mode.
- 50 warmup iterations and 200 measured iterations using CUDA events.
- Reports FP32/TF32 and AMP FP16 separately; postprocessing is excluded.
- FLOPs use the RF-DETR JIT counter plus QK-transpose and attention-V costs for `scaled_dot_product_attention`.
- One multiply-add is counted as one FLOP. `grid_sampler` remains ignored, matching the RF-DETR utility convention.
- These timings are PyTorch eager results and must not be compared directly with TensorRT latency from a paper.

P3/P4 and P3/P4/P5 benchmarks at 576:

| Levels | Params | GFLOPs | FP32/TF32 mean | FP32 p50 | FP32 peak memory | Peak train memory |
|---|---:|---:|---:|---:|---:|---:|
| P3/P4 | 35.74M | 60.51 | 15.85 ms | 15.85 ms | 222.53 MiB | 14,629 MiB |
| P3/P4/P5 | 42.60M | 62.89 | 19.95 ms | 19.27 ms | 249.37 MiB | 15,527 MiB |

Adding P5 to P3/P4 changes the measured cost as follows:

- Parameters: +6.86M (+19.18%).
- GFLOPs: +2.38 (+3.93%).
- FP32/TF32 eager p50 latency: +3.42 ms (+21.59%).
- Peak inference allocation: +26.84 MiB (+12.06%).
- Peak per-process training allocation: about +898 MiB (+6.14%).
- Accuracy under the matched LR-A recipe: +0.52 AP, including +0.54 APs and
  +0.89 APl.

P3/P4 is therefore the efficiency-oriented baseline. It retains about 82% of
the P3/P4/P5 gain over P4 while avoiding the expensive P5 projector branch.
P3/P4/P5 remains the accuracy-oriented option, but its extra 0.52 AP should be
reported with its parameter and latency cost.

AMP is slower in this eager-mode benchmark, most likely because autocast and mixed-kernel dispatch overhead dominate at batch size 1. It should not be treated as TensorRT/FP16 deployment performance.

### Final Results (2026-07-19)

All four jobs completed successfully. Values below use each run's best EMA
checkpoint and are COCO AP points.

| ID | Levels | Indexes | AP | AP50 | AP75 | APs | APm | APl | Epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | P4 | `[2,5,8,11]` | 48.72 | 68.10 | 52.24 | 24.62 | 54.21 | 70.42 | 24 |
| E1 | P3/P4/P5 | `[2,5,8,11]` | **51.60** | **70.38** | **55.68** | **29.43** | **56.13** | **72.05** | 23 |
| E2 | P4 | `[1,4,7,11]` | 48.95 | **68.43** | 52.30 | **25.50** | 54.55 | 70.31 | 24 |
| E3 | P4 | `[3,6,9,11]` | **48.96** | 68.39 | **52.44** | 24.64 | **54.57** | **70.68** | 24 |

Interpretation:

- E1 improves over E0 by 2.88 AP, 3.44 AP75, 4.81 APs, 1.91 APm, and 1.69 APl. The gap is already about 2.8 AP after epoch 1 and remains about 2.6 AP before the epoch-20 LR drop, so it is a feature-level effect rather than a best-epoch artifact.
- E2 and E3 are separated by only 0.003 AP. Both are about 0.23 AP above E0, within the scale where a repeat seed is needed. No intermediate-index set is established as superior.
- The optimized P4 LR recipe is 0.26 AP below the previous aligned random-detector P4 run (48.99 AP). Directly copying DEIMv2's detector/backbone LR ratio is therefore rejected for the current RF projector.
- The epoch-20 step drop is useful: all runs gain roughly 0.6-0.8 AP immediately. It should be separated from the rejected extreme LR ratio in future experiments.
- P3/P4/P5 increases trainable parameters from 33.21M to 42.60M, an extra 9.38M or 28.25%. Its checkpoint grows from about 532MB to 682MB. FLOPs and latency were not measured, so E1 is an accuracy-oriented result, not yet a compute-matched RF-DETR Medium result.
