# SDSR Projector 全过程交接文档

> 项目：`/data/cpc/root/project/RF-DETR-DINOv3`
> 分支：`exp/scale-decoupled-projector`
> 更新时间：2026-08-31
> 范围：SDSR 从 v1 到 v46、匹配策略整合、Register 纠正与 Full COCO 验证

## 1. 文档定位

本文是 SDSR 方向的正式全流程 handoff，不只记录最终 v40。它重点保存以下过程：

1. 为什么最初尝试完整替换 `MultiScaleProjector`；
2. v1 为什么轻量却明显掉点；
3. v2-v23 如何逐步恢复多层融合能力；
4. v24-v36 如何优化 P5 和执行效率；
5. v37-v46 如何验证路由、C2f 替代和空间条件残差；
6. 另一会话的 iterative box refinement、scale routing 和 P5 轻量化怎样影响实验环境；
7. v40 与 Dense O2O、Group、CDN 的整合结果；
8. 为什么历史 36.9718 AP 不能作为纯 SDSR 最终数字。

项目根目录下的 `SDSR_PROJECTOR.md` 和 `SDSR_V36_MILESTONE.md` 是重要原始记录，
但本文不依赖读者先阅读它们。

## 2. 项目背景

RF-DETR-DINOv3 的主路径是：

```text
image
  -> DINOv3 intermediate hidden states
  -> MultiScaleProjector / SDSR
  -> P3/P4/P5 detector memories
  -> deformable transformer decoder
  -> class and box heads
```

DINOv3 patch token 默认约为 stride 16。Projector 必须同时完成：

- 多个 ViT hidden layers 的通道对齐；
- 单一空间尺度到 P3/P4/P5 的重采样；
- 多层语义融合；
- 小目标、边界和大目标语义的平衡。

RF-DETR 的 `MultiScaleProjector` 在每个目标尺度对多层特征重采样，并通过 C2f 融合。
DEIMv2-DINOv3 则把尺度适配、STA 局部细节和 HybridEncoder 融合分散在多个模块中。
两者都说明 ViT 多层融合是检测性能的重要来源。

## 3. 形成 SDSR 的相关工作

设计过程参考并讨论了：

- DEIMv2 的 STA 细粒度分支；
- UPLiFT 的局部注意力上采样；
- ViT-Up 的 faithful feature upsampling；
- RaysUp 的 geometry-aware ray representation；
- Spatial Frequency Modulation；
- CROWn 的 anti-aliased downsampling 和 phase-calibrated fusion。

最终没有把全部方法直接拼接，而是提炼出三个尺度职责：

- **P3**：恢复局部、高分辨率语义，重视相位、边界和小目标；
- **P4**：保持原生 stride16 语义，作为最稳定的语义锚点；
- **P5**：抗混叠下采样，同时保留大目标结构和最深层细节。

SDSR 的核心创新是尺度职责解耦和语义重组，不是单独更换一个插值函数。

## 4. SDSR-v1：近乎完整替换 MultiScaleProjector

### 4.1 结构

v1 几乎完全删除了原 Projector 的 C2f 融合，包含四个核心组件。

**Layer-channel routing**

每层 hidden state 独立 LayerNorm，经共享 1x1 projection。P3 和 P4 使用独立的
channel-wise softmax router：

```text
S_s(c) = sum_l softmax(a_s(l,c)) * Proj(LN(H_l))(c)
```

**P3 semantic local reassembly**

P3 source 为四个 stride8 child positions 预测 3x3 sampling distribution，通过
`unfold` 和 `pixel_shuffle` 重组。初始权重接近双线性相位先验。

**Directional image guide**

轻量 CNN 提供水平、垂直、局部和 dilated image cues，只预测重组权重，不直接把原始
CNN feature 拼入 detector memory。

**P5 anti-aliased phase downsample**

融合 fixed binomial low-pass decimation 与四个 2x2 sampling phases；高频响应预测 gate，
初始偏向 low-pass。

P4 仅保留 routed semantic source 和小型 depthwise residual refine。

### 4.2 参数量和结果

| Projector | Parameters |
|---|---:|
| MultiScaleProjector | 10,629,888 |
| SDSR-v1 detail32 | 556,752 |

v1 Projector 参数约为 MSP 的 5.2%。首轮 Medium 消融：

| Version | AP |
|---|---:|
| MSP baseline | **34.0308** |
| S0 lightweight routed pyramid | 30.8808 |
| S1 + semantic local reassembly | 30.6227 |
| S2 + directional guide | **31.3523** |
| S3 + phase downsample | 30.7520 |

结论：尺度解耦思想没有被否定，但同时删除 C2f、多层 identity 和高容量跨通道融合过于
激进。S2 相对更好，说明图像条件局部信息有效，但无法弥补主语义融合容量损失。

## 5. v2-v9：定位性能损失来源

这一阶段很多版本只训练 3-6 epoch，是早期筛选而非正式最终 AP。

### 5.1 v2：恢复 layer identity

v2 使用 layer-preserving `LightweightScaleFusion`，避免先把四层压成单一 source。

| Variant | AP |
|---|---:|
| v2 current baseline | 33.7428 |
| v2 core rank64 | 31.9491 |
| v2 directional rank64 | **32.4241** |
| v2 directional rank96 | 31.9215 |

恢复 layer identity 明显修复 v1，但轻量 fusion 仍不足。单纯提高 rank 没有收益。

### 5.2 v3-v6：空间条件重组

- v3：target-scale layer fusion、rank downsample、spatial-confidence detail；
- v4：每层先 local reassembly，再做目标尺度融合；
- v5：joint spatial reassembly，同时保留 layer channels；
- v6：shallow-token phase detail，比较 anti-alias on/off。

早期 screen：

| Version | Epochs | Screen AP |
|---|---:|---:|
| v3 core | 6 | 14.6249 |
| v3 spatial | 3 | 9.2530 |
| v4 guided | 3 | 9.6490 |
| v4 semantic | 4 | 11.6524 |
| v5 guided | 6 | 15.0876 |
| v5 semantic | 3 | 9.9474 |
| v6 anti-alias | 4 | 10.9306 |
| v6 no anti-alias | 4 | **11.2714** |

复杂空间自适应使优化更困难。过早强化低通会损失有用 detail。局部重组必须建立在稳定
语义主干上，不能替代主干。

### 5.3 v7-v9：scale-first 和 CSP mixing

- v7：先构造目标尺度，再进行 full-channel P3 phase up/downsample；
- v8：CSP semantic mixing 和 inverted depthwise bottleneck；
- v9：P4 anchor preserving、P3 layer-phase residual、P5 semantic residual。

| Version | Epochs | Screen AP |
|---|---:|---:|
| v7 b0 | 6 | **16.7887** |
| v8 CSP rank64 | 6 | 15.3026 |
| v8 CSP rank96 | 6 | 15.6357 |
| v9 anchor | 6 | 15.0663 |

v7 表明先对齐尺度再融合比先压缩层再重组更可靠，但仍低于 MSP。

## 6. v10-v23：恢复融合容量并达到 36 AP

### 6.1 ExactScaleFusion 转折

v10 引入 `ExactScaleFusion`：每个 hidden state 独立对齐目标尺度，拼接后使用 C2f 融合。
这是从“完全删除原组件”转向“保留已验证融合骨架，只创新重采样和层贡献”的关键转折。

- v10：P3/P4 ExactScaleFusion，P5 alias-aware；
- v11：P5 TwoBasisDownsample；
- v12：P5 FourBasis；
- v13：高通初始化 TwoBasis。

v11 完整结果：

| Run | AP |
|---|---:|
| seed42, lr_drop100 | 34.1341 |
| seed43, lr_drop100, e28 | 35.6191 |
| seed42, lr_drop22, e28 | **35.6432** |

### 6.2 v14-v22

- v14：layer-adaptive P5，一条 dense 与三条 two-basis；
- v15：zero-init bottom-up cross-scale；
- v16：zero-init P4-to-P5 pool residual；
- v17：提高 P4-to-P5 gate 初始梯度；
- v18：learnable binomial low-pass bases；
- v19：额外低分辨率 P5 fusion；
- v20：per-layer low-rank semantic adapters；
- v21：low-rank spatial-semantic residual；
- v22：expandable grouped detail sampling。

v14-v16 的 6e screen 约为 19.42、19.39、19.23。简单跨尺度 residual 和通用 adapter
没有形成稳定突破，搜索因此转向只扩展最深层 P5 detail。

### 6.3 v23 成熟骨架

```text
P3: ExactScaleFusion(scale=2) + C2f
P4: ExactScaleFusion(scale=1) + C2f
P5: TwoBasisDownsample per hidden layer
    + expanded grouped 3x3 detail on deepest feature only
    + C2f
```

| Run | AP | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| seed42, 24e | 35.6114 | - | - | - | - | - |
| seed42, low-LR finish | **36.0590** | 53.2316 | 37.9790 | 17.7589 | 38.6705 | **55.9784** |
| seed43, e28 | 35.7784 | 52.630 | 37.840 | 18.065 | 38.806 | 54.535 |

v23 比无 routing MSP low-LR reference 35.898 高 0.161 AP，但比单独调优的 routed MSP
36.142 低 0.083 AP。这两个 MSP 对照不能混用。

## 7. v24-v36：细节路径和执行效率

### 7.1 v24-v33

- v24：earliest-layer grouped detail；
- v25：dense early-layer detail；
- v26：endpoint-adaptive grouped detail；
- v27：8 groups / 48 channels；
- v28：32 groups / 12 channels；
- v29：deep semantic basis residual，35.5906 AP；
- v30：conservative image-detail residual at P3；
- v31：grouped-detail + semantic residual P5，35.8457 AP；
- v32：deepest-layer phase-calibrated P5；
- v33：off-diagonal grouped detail residual。

这些版本没有稳定超过 v23。最深层 grouped detail 已接近合理容量，继续向早层或所有路径
扩展 detail 容易增加噪声和优化难度。

### 7.2 v34-v36

- v34：packed separable TwoBasis；
- v35：single fused-kernel TwoBasis；
- v36：打包重复 low-pass、shallow depthwise 和 normalization，同时保留 v23 完整 deepest
  grouped 3x3 operator。

seed43 native training：

| Model | AP | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| v23 | 35.778 | 52.630 | 37.840 | 18.065 | 38.806 | 54.535 |
| v34 | 35.635 | 52.710 | 37.830 | 17.443 | 38.331 | 52.776 |
| v36 | 35.501 | 52.297 | 37.815 | 17.150 | 38.390 | 54.325 |

native v36 因浮点累加顺序改变，不保证训练轨迹等价。v23 checkpoint 转为 v36 后：

- strict load 成功；
- projector 最大输出误差 `2.86102294921875e-06`；
- 独立评测复现 36.059 AP；
- optimizer/scheduler state 被移除，只用于评测、推理或重新加载权重。

效率：

| Model | Params | GFLOPs | FP32 mean | AMP mean |
|---|---:|---:|---:|---:|
| MSP | 40.574M | 90.564 | 23.415 ms | 29.330 ms |
| v23 | 35.757M | 88.644 | 24.354 ms | 29.682 ms |
| v34 | 35.686M | 88.618 | 23.450 ms | 30.098 ms |
| v36 | **35.757M** | **88.646** | **22.229 ms** | 29.871 ms |

v36 相比 MSP 总参数少 11.9%，GFLOPs 少 2.1%，FP32 mean 快 5.1%。Projector 单模块
batch1 仍略慢，因此必须同时报告模块和端到端效率。

## 8. 另一会话改动的影响边界

共享工作树中，另一会话实现或完善了：

- iterative box refinement；
- scale routing；
- MultiScaleProjector P5 轻量化；
- 分割/INSID3 相关代码。

这些功能影响了后期可用 recipe，但并不属于 SDSR Projector 内部。SDSR-v1 的尺度解耦、
P3 local reassembly 和 P5 phase-aware downsampling 在这些工作之前已提出；v10-v23 的
ExactScaleFusion、TwoBasis 和 grouped detail 也是 SDSR 自身迭代。

v23 之后的强基线通常同时启用 iterative refinement 和 scale routing。因此报告时必须区分：

1. Projector strict 对照；
2. decoder/attention recipe；
3. Dense O2O/CDN/Group 的训练策略增益。

后期关键实验使用 `experiment_snapshots/` 固定源码，避免共享工作树继续变化。snapshot 是
本地复现产物，不再重复提交到 GitHub。

## 9. v37-v46：后期结构搜索

统一主要协议为 Medium、24e、lr_drop20、total batch16。

| Version | Seed | AP | AP50 | AP75 | Conclusion |
|---|---:|---:|---:|---:|---|
| v37 R1 routed C2f | 42 | 35.4644 | 52.7464 | 37.4281 | 不优于基线 |
| v37 R2 routed C2f | 42 | 35.5850 | 52.5864 | 38.0183 | v37 最好 |
| v37 R3 routed C2f | 42 | 35.1996 | 52.0696 | 37.0467 | 路由过强 |
| v38 P3 local residual | 42 | 35.4078 | 52.5533 | 37.8640 | 无明显收益 |
| v38 P4 dynamic | 42 | 35.4638 | 52.5369 | 37.6466 | 收益有限 |
| v38 P4 dynamic | 43 | 35.4999 | 52.5917 | 37.3605 | 未跨 seed 突破 |
| v39 P4 static | 43 | 35.3803 | 52.3689 | 37.5331 | 不够灵活 |
| v40 P4 fixed prior | 43 | 35.7905 | 52.8548 | 38.1473 | 强先验有效 |
| v40 P4 learnable | 42 | 35.8411 | 52.8678 | 38.3695 | 强候选 |
| v40 P4 learnable | 43 | **35.9690** | **53.1171** | **38.3228** | strict 最佳 |
| v40 bounded dynamic | 43 | 35.6071 | 52.4729 | 37.7550 | 动态自由度无收益 |
| v41 P3 uniform | 43 | 35.6444 | 52.7959 | 38.1525 | 不超 v40 |
| v41 P3 shallow | 43 | 35.5694 | 52.3204 | 38.0218 | 浅层偏置不稳 |
| v42 P5 uniform | 43 | 35.8212 | 52.7448 | 38.2189 | 接近 v40 |
| v42 P5 deep | 43 | 35.5558 | 52.7786 | 37.8092 | 过度偏深层 |
| v43 spatial 0.10 | 43 | 35.5544 | 52.8152 | 37.8849 | 无收益 |
| v43 spatial 0.20 | 43 | 35.8568 | 52.8093 | 38.2264 | APL 好但不超 v40 |
| v44 C3k2 | 43 | 35.4063 | 52.5067 | 37.8094 | 替换 C2f 后退化 |
| v44 RepC3 | 43 | 34.5526 | 51.5115 | 36.5184 | 明显退化 |
| v45 centered spatial | 43 | 35.7549 | 52.7350 | 38.3068 | 更稳但不超 v40 |
| v46 annealed centered | 43 | 35.6206 | 52.4764 | 37.9500 | 退火无收益 |

结论：

- v37 回答了“P3/P4 使用新方法同时保留 C2f”的问题，折中可行但未超越 v40；
- P4 全局可学习深度先验比逐图动态路由可靠；
- P3/P5 不需要同时复杂化；
- C2f 在 DINOv3-small 和 24e budget 下优于 C3k2/RepC3；
- 空间 residual、中心化和退火增加了自由度，却没有形成稳定 AP。

## 10. 最终选定的 SDSR-v40

```text
4 DINOv3 hidden states
  -> P3 ExactScaleFusion(scale=2) + C2f
  -> P4 ExactScaleFusion(scale=1) + C2f + learnable depth prior
  -> P5 TwoBasisDownsample per layer
       + deepest expanded grouped detail
       + C2f
```

P4 prior 初始化：

```text
(0.2485995, 0.4129705, 0.7943345, 2.5440950)
```

内部保存为 logits，经 softmax 后乘 4，使平均 gate 为 1。最终配置：

```text
projector_type=sdsr_v40_p4_learnable
projector_scale=P3 P4 P5
sdsr_rank_channels=64
sdsr_detail_channels=32
sdsr_use_local_reassembly=False
sdsr_use_directional_guide=False
sdsr_use_phase_downsample=True
sdsr_cross_scale_mode=none
projector_p5_mode=full
```

v40 与 v1 的关系：v1 是概念上最纯的完整替换；v40 保留尺度解耦思想，但接受 C2f 和
多层独立尺度对齐是必要容量。创新集中在非对称尺度处理、P5 two-basis/detail 和 P4 depth
prior，而不再追求删除原模块内部所有组件。

## 11. 与 Dense O2O、Group、CDN 整合

### 11.1 Dense O2O

| Config | AP | AP50 | AP75 | Peak memory |
|---|---:|---:|---:|---:|
| strict, no Dense | 35.9690 | 53.1171 | 38.3228 | 13,432 MiB |
| Mosaic/MixUp/CopyBlend=.5/.5/.5 | 36.1239 | 52.7063 | 38.6949 | 22,633 MiB |
| no-MixUp=.5/0/.5 | **36.6764** | **53.8563** | **39.3384** | 15,063 MiB |
| CopyBlend-only=0/0/.5 | 36.4538 | 53.3681 | 39.1693 | 13,344 MiB |

删除 MixUp 同时提高 AP 并显著减少显存。选定调度为 epoch2 开始，image augmentation
epoch12 停止，CopyBlend epoch21 停止，Mosaic/CopyBlend 均为 0.5，CopyBlend 每图 1 个目标。

### 11.2 Group

| Group | AP | Peak memory |
|---:|---:|---:|
| 6 | **36.6764** | 15,063 MiB |
| 4 | 36.5252 | 11,954 MiB |
| 2 | 36.1016 | 10,248 MiB |

Group6 为精度默认，Group4 为显存备选。

### 11.3 CDN

| CDN | AP | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| disabled | 36.1761 | 53.3948 | 38.7153 | 18.2973 | 39.2829 | 54.5929 |
| DN25, loss0.5 | **36.9718** | **53.8691** | **39.7993** | 18.2784 | 40.4331 | 54.7916 |
| DN50, loss0.3 | 36.7024 | 53.8330 | 39.2941 | 17.9703 | **40.4426** | 54.0024 |
| DN50, loss0.5 | 36.6764 | 53.8563 | 39.3384 | 17.9813 | 40.0817 | 53.0949 |

该表属于历史 Reg1 搜索，说明 DN25 更适合 v40，但不能直接视为 Reg0 下已穷举最优。

### 11.4 EMA 和 total batch

EMA 0.993/tau100 为 36.6606 AP，同次 regular 36.6445，均低于 no-EMA 36.9718。
total batch32 降至 33.8250 AP。最终不使用 EMA，也不提高 total batch。

## 12. Register border 污染与纠正

历史搜索意外保留：

```text
backbone_register_border_tokens=1
```

它给 token grid 增加一圈 register/noise patch，参与 backbone self-attention，在进入 Projector
前裁掉。最终方法未采用 inference-time register，因此纯 SDSR 必须设为 0。

| Metric | Reg1 | Reg0 clean | Delta |
|---|---:|---:|---:|
| AP | 36.9718 | **36.4706** | -0.5012 |
| AP50 | 53.8691 | 53.6406 | -0.2285 |
| AP75 | 39.7993 | 38.9451 | -0.8542 |
| APS | 18.2784 | **19.0819** | +0.8034 |
| APM | 40.4331 | 39.7308 | -0.7023 |
| APL | 54.7916 | 53.6951 | -1.0965 |

`36.4706 AP` 是当前纯 SDSR-v40 integrated Medium baseline；`36.9718 AP` 只能作为包含
register border 的 ablation。

## 13. 当前 Full COCO 配置

```text
run=coco_full_s43_detinit1043_sdsr_v40_dense_nomix_g6_cdn25_l05_reg0_lrd20_e24
epochs=24
batch_size=8
grad_accum_steps=2
total_batch_size=16
resolution=640
multi_scale=True
expanded_scales=True
lr=1e-4
lr_encoder=1.5e-4
lr_drop=20
seed=43
detector_init_seed=1043
projector=sdsr_v40_p4_learnable
iterative_bbox_refinement=True
scale_routing=True
group_detr=6
CDN=DN25, box_noise0.6, label_noise0.5, loss0.5
Dense=Mosaic0.5, MixUp0, CopyBlend0.5
backbone_register_border_tokens=0
EMA=False
```

相关入口：

```text
run_sdsr_v40_best_full_gpu1.sh
output/coco_full_s43_detinit1043_sdsr_v40_dense_nomix_g6_cdn25_l05_reg0_lrd20_e24/
```

此前 Reg1 Full 已停止，不能与 Reg0 checkpoint 拼接或续训。

## 14. 关键代码

```text
src/rfdetr/models/backbone/semantic_reassembly_projector.py
src/rfdetr/models/backbone/semantic_reassembly_projector_v2.py
...
src/rfdetr/models/backbone/semantic_reassembly_projector_v46.py
src/rfdetr/models/backbone/backbone.py
tests/test_semantic_reassembly_projector.py
run_coco_subset.py
```

v23 转 v36：

```text
tools/convert_sdsr_v23_to_v36.py
output/coco_medium_v36_converted_from_v23_ap36p059/checkpoint_best_regular.pth
```

## 15. 不应重复的方向

1. 不再把全部 C2f 删除作为主线；
2. 不把 2-6 epoch screen AP 当最终结果；
3. 不认为参数更少必然更快，必须实测 kernel 和端到端 latency；
4. 不继续用 C3k2/RepC3 替代当前全部 C2f；
5. 不继续堆 P4 dynamic/spatial routing，已有多轮负结果；
6. 不默认开启 EMA、register border 或 total batch32；
7. 不把 iterative refinement/scale routing 的收益全部算给 SDSR；
8. 不把 Reg1 36.9718 写成纯 SDSR 最终结果；
9. 不提交新的完整 `experiment_snapshots` 源码副本到 GitHub。

## 16. 下一步

1. 完成 Reg0 Full COCO，报告 best epoch、AP/AP50/AP75/APS/APM/APL、显存和时间；
2. 补 Reg0 seed42 Medium；
3. Reg0 下最小邻域复测 no-CDN、DN25、DN50 与 Group4/6；
4. 论文表中分开报告 Projector strict、decoder recipe 和 Dense/CDN 增益；
5. 若继续 Projector，优先研究 P3 faithful boundary reconstruction；
6. 若继续效率，围绕 v23->v36 等价打包和导出 kernel；
7. 若接分割，重新建立任务内 baseline，不从 detection AP 直接外推。

## 17. 最终阶段结论

SDSR 不是一次完成的“完全替换 MultiScaleProjector”。v1 用极轻路由、P3 local
reassembly 和 P5 anti-aliased phase downsampling 将 Projector 参数降至约 5.2%，但因
删除 C2f 和高容量多层融合而损失约 2.7-3.4 AP。v2-v23 逐步证明 layer identity、
目标尺度独立对齐和 C2f 是 DINOv3 检测的关键，并形成 P3/P4 ExactScaleFusion、P5
TwoBasis 与 deepest grouped detail 的成熟骨架。v23 达到 36.059 AP，v36 可等价转换并
减少 11.9% 总参数。v37-v46 又证明 P4 learnable depth prior 比复杂动态路由和替换 C2f
更可靠，最终选择 v40。关闭遗留 register border 后，v40 与 Dense no-MixUp、Group6、
CDN25 的可信 Medium 结果为 36.4706 AP。
