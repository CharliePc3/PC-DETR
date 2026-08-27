# RF-DETR-DINOv3 Backbone Refinement 跨会话交接文档

> 更新时间：2026-08-22  
> 适用项目：`/data/cpc/root/project/RF-DETR-DINOv3`  
> Refinement 主线提交：`8c3af42 Unify DINOv3 backbone refinement experiments`  
> 原 refinement 分支：`exp/lazystrike-refinement`  
> 当前项目分支：`exp/scale-decoupled-projector`

## 1. 这份文档的用途

本文档用于把本会话完成的 DINOv3-small backbone refinement 工作交接给
其他模型改进会话。它回答以下问题：

1. 已经研究了哪些 refinement 方向；
2. 每个方向具体优化什么、如何实现；
3. 哪些结果可以认为有效，哪些结论仍不稳定；
4. 最佳权重、代码和结果文件在哪里；
5. 如何把 refinement 与 projector、decoder、训练策略等其他方向组合；
6. 哪些实验已经证明不值得重复。

最重要的边界是：本文中的 33--34 AP 结果来自当时的基础
`MultiScaleProjector + group 6 + CDN` 检测器。当前 projector/decoder 主线已经达到
约 36 AP，因此 refinement 收益不能直接与新结果相加，必须在当前最强架构上重新做
受控验证。

## 2. 模型与实验背景

RF-DETR-DINOv3 以 RF-DETR/LW-DETR 为检测框架，将 backbone 从 DINOv2 换成
DINOv3-small。Backbone 输出的隐藏层特征经过 projector 形成 P3/P4/P5，再送入
DETR decoder。Refinement 是离线进行的：单独微调 DINOv3 backbone，导出普通
state dict，再通过 `--pretrained-encoder` 加载到检测训练中。

所有已实现的 refinement：

- 不增加检测推理模块；
- 不增加推理延迟；
- 不要求检测时继续运行 teacher；
- 输出可以直接被 RF-DETR-DINOv3 backbone loader 加载。

### 2.1 旧统一检测协议

除特别说明外，文中的主要对比采用：

- 数据：`COCO_RFDETR_TEST/medium`
- 训练：24 epochs
- 分辨率：640
- batch size：8
- gradient accumulation：2
- total batch size：16
- seed：42
- group DETR：6
- projector：P3/P4/P5
- multi-scale + expanded scales
- decoder layers：4
- queries/select：300/300
- CDN：开启
- DN number：50
- DN box noise：0.6
- DN label noise：0.5
- DN loss coefficient：0.5

该协议的原始 DINOv3-small baseline 为：

| AP | AP50 | AP75 | APs | APm | APl | Last-5 AP |
|---:|---:|---:|---:|---:|---:|---:|
| 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |

## 3. 三条独立 refinement 路线

## 3.1 UniRefiner：异常 token 清理与 register 吸收

### 原理

UniRefiner 使用 frozen teacher 识别 clean/spurious token，并在输入图像周围构造
register 区域。训练过程中通过 FP/GP/AH 过滤、clean-token NCE、register absorption、
uniformity 和 spatial consistency distillation（SCD），让异常信息迁移到 register，
同时尽量保持有用 dense token。

本项目做过：

- 原论文实现接入 DINOv3-small；
- zero/rand/randn register；
- SCD on/off；
- AH on/off；
- 1/2/3 epoch；
- COCO、CC3M5k、ADE20K 图像 refinement；
- abnormal-token ratio、PCA、token heatmap、COCO detection、ADE20K linear segmentation。

### 最佳保留配置

`balanced_zero3e`：

- COCO train2017 images
- image size：640
- batch size：8
- epochs：3
- BF16
- LoRA：`linear-r8`
- optimizer LR：`1e-4`
- weight decay：0.1
- register factor：37
- register type：zero
- FP/GP threshold：0.5
- adaptive register threshold：0.55
- AH filter：关闭
- proposals：3
- uniformity：0.3
- SCD start：0.1
- SCD weight：0.4

完整配置：

```text
/data/cpc/root/project/UniRefiner/outputs/
dinov3_small_640_unirefiner_coco_balanced_zero3e/resolved_config.yaml
```

### 结果

| 模型 | AP | AP50 | AP75 | APs | APm | APl | Last-5 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |
| balanced_zero3e | 33.88 | 50.96 | 35.95 | 17.09 | 36.58 | 51.94 | 33.46 |

- Peak AP：`+0.27`
- Last-5 AP：`+0.30`
- AP75：`+0.71`
- fixed abnormal-token ratio：`9.62% -> 5.75%`，相对下降约 40%

但下游证据不稳定：

- author-recipe seed 43 的 peak AP 没有超过对应 baseline；
- ADE20K linear segmentation：base 40.89 mIoU，balanced_zero3e 40.75；
- COCO-refined UniRefiner 权重在 ADE20K 上没有带来 dense feature 提升；
- PCA/heatmap 中的异常区域变化通常比 abnormal ratio 数值变化更弱。

### 当前判断

UniRefiner 的 token redistribution 机制在本项目上得到验证，但对 DINOv3-small 的
检测/分割收益较弱，不应作为当前默认 backbone。它更适合保留为机制研究和未来
更大 DINOv3 backbone 的候选。

## 3.2 LazyStrike：前景感知的 CLS 聚合

### 原理

LazyStrike 路线用 GT boxes 约束 CLS 应聚合前景相关 patch，并针对多目标检测增加：

- box-balanced 前景聚合；
- object coverage；
- 水平翻转 CLS consistency；
- frozen-teacher distillation；
- 对复杂 COCO 场景避免把所有未标注区域当成背景。

该路线主要改变最终两层的 CLS/patch 聚合关系，收益偏向总体 AP 和大目标。

### 最佳配置：v2-conservative e2

- COCO train2017 随机 5k images
- resolution：640
- epochs：2
- batch size：2
- FP32 refinement
- train last blocks：2（blocks 10--11 + final norm）
- LR：`5e-6`
- weight decay：0.05
- max boxes/image：12
- lazy top-k：1
- target weight：0.1
- cover margin/temperature：0.1/0.1
- align/cover/consistency/distill：`0.25/0.15/0.25/1.0`
- seed：42

配置文件：

```text
output/lazystrike_refine/
dinov3_small_coco5k_lazystrike_v2_conservative/args.json
```

### 结果

| 模型 | AP | AP50 | AP75 | APs | APm | APl | Last-5 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |
| LazyStrike v2 e2 | **34.36** | **51.71** | 36.37 | 16.77 | 36.48 | **52.97** | **33.71** |

- Peak AP：`+0.75`
- AP75：`+1.13`
- APl：`+0.98`
- Last-5 AP：`+0.55`
- APs：`-0.10`

seed 43 仍为小幅正收益，但收益幅度明显缩小，说明方向有效但方差较大。

### 当前判断

LazyStrike v2-conservative e2 是旧检测架构上最强的通用 refinement checkpoint，
也是与其他新架构整合时优先验证的权重。弱点是小目标收益不足，且 ADE20K linear
segmentation 与 base 基本持平，不能宣称改善通用 dense representation。

## 3.3 Detection-aware Dense Consistency

### 原理

该路线不优化 CLS，而直接优化 RF-DETR 实际使用的 patch features：

1. 提取原图 student、水平翻转 student 和 frozen teacher 的多层 patch features；
2. 将翻转后的 feature grid 映射回原坐标；
3. 对 block 8/11 做 teacher preservation；
4. 用 patch-box overlap 构造 object-aware consistency；
5. 先在 box 内平均，再按 box/image 平均，避免大框主导；
6. 加低权重 global consistency，避免把未标注但有意义的物体强制当背景。

### 最佳配置：dense_only e1

- COCO train2017 随机 5k images
- resolution：640
- epochs：1
- batch size：2
- FP32
- train blocks：8--11 + final norm
- LR：`2.5e-6`
- layer LR decay：0.8
- dense layers：8/11
- layer weights：0.35/0.65
- teacher distill：1.0
- dense object：0.15
- dense global：0.05
- LazyStrike align/cover/consistency：全部为 0
- seed：42

### 结果

| 模型 | AP | AP50 | AP75 | APs | APm | APl | Last-5 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |
| dense_only | 34.03 | 50.94 | **36.75** | **17.36** | **36.85** | 51.95 | 33.56 |

- Peak AP：`+0.42`
- AP75：`+1.51`
- APs：`+0.48`
- APm：`+0.83`
- Last-5 AP：`+0.40`
- AP50/APl 基本不变

### 当前判断

Dense consistency 是独立有效的定位信号，优势与 LazyStrike 互补：它改善小中目标和
严格定位，而不是增强粗粒度分类或大目标。当前仍缺少 dense-only seed 43，因此其
稳定性证据弱于 LazyStrike。

## 4. 三条路线的横向结论

| 方向 | 最佳 AP | 相对 base | 最明显收益 | 最明显问题 |
|---|---:|---:|---|---|
| UniRefiner | 33.88 | +0.27 | abnormal ratio、AP75/APm | 跨 seed/任务不稳定 |
| LazyStrike | **34.36** | **+0.75** | 总体 AP、AP50、APl | APs 不提升 |
| Dense consistency | 34.03 | +0.42 | AP75、APs、APm | 缺 seed 43，APl 不提升 |

可以认为：

- LazyStrike：检测收益最明确；
- Dense consistency：定位收益明确；
- UniRefiner：机制证据明确，但下游收益弱；
- 三者不是可以简单相加的独立增益项。

## 5. 已完成的组合实验

| 组合 | AP | AP75 | APs | APm | APl | Last-5 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Lazy+dense simultaneous | 33.72 | 35.78 | 16.26 | 36.06 | 53.06 | 33.33 | 最终层目标冲突 |
| Lazy -> dense | 34.01 | 36.32 | 17.41 | 36.62 | 52.69 | 33.61 | 擦除部分 Lazy 收益 |
| Dense -> Lazy e2 | 33.94 | 35.94 | 15.61 | 36.48 | 53.65 | 33.46 | 明显转向大目标 |
| Uni+dense joint | 34.11 | 36.39 | 16.17 | 36.05 | 53.67 | 33.52 | 仅比 dense peak +0.08 |
| Uni -> Lazy | 34.04 | 36.66 | 16.24 | 36.86 | 53.32 | 33.59 | 低于 standalone Lazy |
| Uni+Lazy joint | 33.65 | 35.97 | 15.73 | 36.13 | 52.58 | 33.49 | 无正向迁移 |

### 为什么没有叠加

1. LazyStrike 与 dense consistency 都会改写 block 11；前者强调 CLS 前景聚合，后者
   强调局部 patch 等变/一致性，联合时 final-block teacher drift 明显增大。
2. 顺序精修即使冻结某一层，也会改变上游 feature distribution，导致后续层与多尺度
   projector 的输入统计变化。
3. Dense -> Lazy 的第二个 Lazy epoch 提高了 CLS box coverage，却降低 LAST stability，
   APs 从 dense-only 的 17.36 降到 15.61，APl 则升到 53.65。
4. UniRefiner 约束可以数值稳定地与其他 loss 共存，但并未产生检测正迁移。

因此没有执行原计划的 Uni+Lazy+dense 三目标 Phase C，也不建议重复大范围 loss sweep。

## 6. Token diagnostic 的正确用法

主工具：

```text
tools/analyze_dinov3_tokens.py
```

已实现指标包括：

- CLS-patch cosine Point-in-Box；
- LAST-style patch stability vote；
- foreground/background score ratio；
- top-k 多目标 box coverage；
- small/medium/large box coverage；
- largest-box dominance；
- flip consistency；
- FP/GP proxy；
- fixed-threshold abnormal-token ratio；
- PCA/heatmap visualization。

必须注意：

1. COCO 是复杂多目标场景，前景上的强响应不能自动算 spurious token；
2. dynamic FP/GP proxy 与 fixed-threshold abnormal ratio 不能横向混用；
3. CLS top-k coverage 提高不代表 detection AP 提高；
4. FP/GP proxy 在多次实验中与 AP 相关性很弱；
5. 模型选择应以 detection AP、AP75、APs/APm/APl 和 last-5 稳定性为主，token
   diagnostic 用于解释而不是替代下游评测。

关键汇总：

```text
output/token_analysis/unirefiner3e_full_comparison.csv
output/token_analysis/balanced_zero3e_before_after/before_after_summary.csv
output/unified_refinement/phase_a_summary.tsv
output/unified_refinement/phase_b_summary.tsv
output/dense_to_lazy/summary.tsv
output/lazystrike_dense_consistency/summary.tsv
```

## 7. ADE20K linear segmentation 验证

工具：

```text
tools/linear_seg_ade20k.py
```

20-epoch frozen-backbone linear segmentation 结果：

| Backbone | ADE20K mIoU |
|---|---:|
| Original DINOv3-small | 40.89 |
| UniRefiner author COCO | 40.77 |
| UniRefiner balanced_zero3e | 40.75 |
| UniRefiner randn3e | 40.27 |
| ADE20K-image UniRefiner | 40.76 |
| LazyStrike v2 | 40.83 |

这些结果说明当前 refinement 主要是检测任务专门化，而不是通用语义分割 feature 的
一致提升。未来接入实例/语义分割时必须重新验证，不能仅凭 detection AP 推断分割收益。

## 8. 代码结构与复现入口

### 核心实现

```text
tools/train_dinov3_lazystrike_refine.py
tools/refinement/unirefiner_objective.py
tools/refinement/__init__.py
tools/analyze_dinov3_tokens.py
tools/linear_seg_ade20k.py
```

统一 trainer 支持：

```bash
--objectives unirefiner
--objectives lazystrike
--objectives dense
--objectives unirefiner dense
--objectives unirefiner lazystrike
--objectives lazystrike dense
```

所有 objective 共用 student、stage-local frozen teacher、optimizer、backward 和 checkpoint
格式。旧命令仍可通过正 loss 权重自动推断 objective。

### 实验编排脚本

```text
run_unified_refinement_phase_a_gpu1.sh
run_unified_refinement_phase_b_gpu0.sh
run_dense_to_lazy_gpu0.sh
```

详细实验记录：

```text
BACKBONE_REFINEMENT_STAGE_SUMMARY.md
EXPERIMENT_LOG_LAZYSTRIKE_DENSE.md
EXPERIMENT_LOG_UNIFIED_REFINEMENT.md
```

## 9. 推荐 checkpoint 与校验哈希

### 9.1 检测首选：LazyStrike v2-conservative e2

```text
output/lazystrike_refine/
dinov3_small_coco5k_lazystrike_v2_conservative/checkpoints/model_final.pt
```

SHA-256：

```text
d5e16ae010841a8060927f981db610be0f1918a8648c0f6de60973b7ab5d6333
```

### 9.2 定位/小中目标候选：dense-only e1

```text
output/lazystrike_refine/
dinov3_small_coco5k_lazystrike_dsdr_dense_only_e1/checkpoints/model_epoch1.pt
```

SHA-256：

```text
397fb9637e1d15e40378dad4c710e78aa460838671635c4783c6af0488664605
```

### 9.3 UniRefiner 机制候选：balanced_zero3e

```text
/data/cpc/root/project/UniRefiner/outputs/
dinov3_small_640_unirefiner_coco_balanced_zero3e/checkpoints/model_final.pt
```

SHA-256：

```text
e92002ccb5e6f5d720203c234921023ccb5b568d78df93f2055e591ffa0fc7ca
```

### 9.4 待判别但不推荐直接使用：Dense -> Lazy epoch 1/2

```text
output/lazystrike_refine/dinov3_small_coco5k_dense_to_lazy_v2_e2/
checkpoints/model_epoch1.pt
output/lazystrike_refine/dinov3_small_coco5k_dense_to_lazy_v2_e2/
checkpoints/model_epoch2.pt
```

Epoch 2 已被检测结果否定；epoch 1 只有 token diagnostic，尚未做对齐 detection。

## 10. 与当前其他改进方向整合时的建议

当前 `exp/scale-decoupled-projector` 上 projector/decoder 路线已达到约 36 AP，典型里程碑见：

```text
MEDIUM_AP36_MILESTONE.md
SDSR_V36_MILESTONE.md
```

Refinement 旧增益不能直接加到新架构。建议另一个会话采用以下最小矩阵：

### Phase I：只测两个独立最强 checkpoint

保持新架构、训练 schedule、augmentation、seed 和 detector initialization 完全不变：

1. 当前新架构 + original DINOv3-small；
2. 当前新架构 + LazyStrike v2 e2；
3. 当前新架构 + dense-only e1。

通过：

```bash
--pretrained-encoder <checkpoint>
```

加载 refined backbone。不要同时改变 projector、decoder、LR 或数据增强。

### Phase II：确认收益是否稳定

若某个 checkpoint 在当前架构上满足：

- peak AP 至少 `+0.20`；
- last-5 AP 同方向提升；
- 不只是 APl 单项迁移；

再用第二个 seed 复测。当前主线已有 `--detector-init-seed`，应同时固定 data seed 和
detector initialization seed，减少模型初始化噪声。

### Phase III：只在必要时研究融合

如果 Lazy 与 dense 在新 projector 上仍分别有效，再研究结构性融合，不要直接 joint loss：

- dense teacher 强约束 block-11 的 small/medium-box patches；
- LazyStrike 主要作用于 global/large-object aggregation；
- 先记录每个 block 的 objective gradient cosine；
- 发现负梯度冲突后再做 gradient routing 或分层 optimizer；
- 不再以 CLS coverage 或 FP/GP proxy 作为主优化目标。

## 11. 不应重复的实验

- 不要再次大范围搜索 UniRefiner register/SCD/AH 参数；DINOv3-small 收益已证明有限。
- 不要直接运行 Uni+Lazy+dense 三目标联合 refinement。
- 不要重复当前形式的 Lazy+dense simultaneous hybrid。
- 不要假设先后顺序可以自然叠加；Lazy -> dense 和 Dense -> Lazy e2 都已失败。
- 不要用 dynamic FP/GP proxy 与 fixed abnormal ratio 比较模型清理能力。
- 不要因为 token heatmap 更集中或 CLS coverage 更高就宣称检测更好。
- 不要把旧 33.61 baseline 上的提升直接叠加到当前 36 AP 模型。

## 12. 当前最合理的阶段结论

1. Backbone offline refinement 在 RF-DETR-DINOv3 上是有效方向，但收益规模小于
   projector/decoder 的结构改进。
2. LazyStrike v2 是当前最值得带入新架构复测的 refinement。
3. Dense consistency 是最值得保留的定位专项路线，尤其适合关注 AP75、APs、APm。
4. UniRefiner 对 abnormal token 的机制验证成立，但 DINOv3-small 下游收益不足。
5. 多个 refinement 目标目前存在 final-layer 和 cross-level feature compatibility 冲突。
6. 下一次整合应从“当前新架构 × original/Lazy/dense”三组严格对照开始，而不是继续
   refinement 参数搜索。

## 13. 给下一个会话的建议开场信息

可直接告诉下一个会话：

> 请先阅读 `BACKBONE_REFINEMENT_HANDOFF_CN.md`。当前需要把已验证的
> LazyStrike v2-conservative e2 和 dense-only e1 backbone checkpoint，分别加载到
> 当前最强 projector/decoder 架构中做严格同配置对照。不要把旧 AP 增益直接相加，
> 不要同时修改其他变量，并同时比较 peak AP、last-5 AP、AP75、APs/APm/APl。
