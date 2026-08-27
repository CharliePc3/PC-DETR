# RF-DETR-DINOv3 检测优化会话交接文档

> 更新时间：2026-08-25（Asia/Shanghai）  
> 项目：`/data/cpc/root/project/RF-DETR-DINOv3`  
> 主题：架构梳理、COCO 分层实验、Decoder/Projector 优化、Backbone refinement 整合、效率测量

## 1. 文档范围

本文总结本会话从项目起步到当前检测主线的工作、实验结论和可复现入口。目标是让下一会话不需要重新阅读大量日志，就能回答：

1. 当前模型到底由哪些模块和训练策略组成；
2. 哪些改动已经证明有效，哪些只是单 seed 候选；
3. 36 AP 是如何达到的；
4. Projector 为什么变轻后没有掉点，以及最终保留了什么；
5. 离线 backbone refinement 为什么难以叠加，在线约束目前做到什么程度；
6. 下一轮应从哪里继续，而不重复已否定实验。

工作树中还存在其他会话并行开发的分割代码和结果。它们不属于本会话的检测优化贡献，因此本文不把 `SEGMENTATION_DIRECTION_AND_INITIAL_RESULTS.md`、EoMT、INSID3 等内容并入主线结论。

## 2. 执行摘要

本会话完成的核心贡献如下。

1. 系统梳理了 LW-DETR、RF-DETR、RF-DETR-DINOv3 和 DEIMv2 的特征流、Projector/Encoder/Decoder 接口以及 DINOv3 中间层使用方式。
2. 建立 deterministic COCO 分层测试集和统一实验 runner，将 smoke、overfit、quick、medium、strong、full 串成由低成本到高可信度的验证漏斗。
3. 将训练、评估和模型结构的关键配置显式暴露，补齐多卡 DDP、标准 COCO `maxDets=100`、断点续训和实验日志记录。
4. 证明简单替换或 BiFPN/Gated 化 MultiScaleProjector 不能稳定优于原结构，随后转向更细粒度的尺度选择、P5 结构分析和受控压缩。
5. 在 decoder/训练侧确定 Group DETR 6、tuned CDN、iterative box refinement、legacy scale routing、正确的 `lr_drop` 和 Dense O2O 路线。
6. medium 单次最佳首次达到 `36.1423 AP`，并完成独立复评；随后形成约 `35.7-35.8 AP` 的双 seed 轻量 Projector 配置。
7. 完成 Projector 五阶段研究：尺度选择、物理剪枝、C2f 非对称精简、P5 重采样共享、高效卷积、蒸馏。最终选择的 S2 方案为 `36.813M` 参数。
8. 将外部会话得到的 LazyStrike / Dense consistency refinement 严格整合到当前最强 detector，证明旧架构收益不能直接叠加。
9. 实现检测训练期短程行为约束。当前最佳在线方案在 medium seed 42 达到 `36.2768 AP`，但尚未做第二 seed，因此仍是候选而非稳定默认。
10. 统一测量 RF-DETR-DINOv3、RF-DETR-Medium、DEIMv2-L 的本机 GFLOPs 和 latency，并保留完整 JSON 与逐层 FLOPs。

## 3. 架构理解与关键结论

### 3.1 RF-DETR-DINOv3

当前 backbone 固定为最小 DINOv3-S/16，检测主线使用中间层 `F2/F5/F8/F11`。这些 feature 的原始空间步长均为 16，MultiScaleProjector 负责通道映射、重采样和跨层融合，生成 P3/P4/P5，再由 Transformer decoder 消费。

Projector 输出 P3/P4/P5 和 decoder layer 数是两件事。Projector 的多尺度 feature 是每个 decoder layer cross-attention 的 memory；它不是“给每个 decoder layer 各输出一张 feature”。

### 3.2 RF-DETR

RF-DETR 不是只把 LW-DETR backbone 换成 DINOv2。其公开训练/权重还包含 windowed-attention DINOv2、NAS/elastic subnet 训练形成的预训练权重、训练配方和部署选择。源码发布版不等价于论文内部完整 NAS 炼制系统，因此官方 Medium checkpoint 的 54 AP 不能解释为“随机 DINOv2 权重 + 当前公开训练脚本”的自然结果。

RF-DETR 主流 P4 配置是一个 decoder 输入 feature level，不代表 MultiScaleProjector 内部没有利用多个 hidden states。它强调单尺度 memory 的速度/精度平衡；本项目的实验表明，DINOv3-S 从 P4 扩展到 P3/P4/P5 有明确收益。

### 3.3 DEIMv2

DEIMv2 先按 ViT 深度把多个中间层映射到不同尺度，并结合轻量 CNN 细粒度分支，再由 HybridEncoder 做跨尺度交互。RF-DETR-DINOv3 则在 MultiScaleProjector 内让多个 hidden states 先经尺度分支和 C2f 交互。

两者都得到“多中间层交互后的 P3/P4/P5”，但归纳偏置不同：

- DEIMv2 的深度到尺度分工更明确，HybridEncoder 的跨尺度建模更强，但模块更多、训练配方更重。
- RF-DETR-DINOv3 的 Projector 更直接，decoder 路径更轻，但容易在所有尺度重复处理同一组 hidden states，产生参数冗余和尺度语义同质化。

## 4. 实验基础设施

### 4.1 COCO 分层数据

目录：`/data/cpc/root/dataset/COCO_RFDETR_TEST`

| 子集 | Train | Val | 用途 |
|---|---:|---:|---|
| smoke | 16 | 16 | shape、加载和前后向检查 |
| overfit | 128 | 64 | 检查可学习性和明显退化 |
| quick | 1,000 | 500 | 低成本方向筛选 |
| medium | 5,000 | 2,000 | 主要消融与 seed 对照 |
| strong | 20,000 | 4,952 | Projector 主候选筛选 |
| full | COCO 全量 | COCO val2017 | 最终训练/迁移验证 |

数据由固定 seed 过滤 COCO annotation，图像使用符号链接，不复制原图。构建说明见 `/data/cpc/root/dataset/COCO_RFDETR_TEST/README.md`。

### 4.2 统一 runner

主入口：`run_coco_subset.py`

已显式暴露的重要参数包括：

- 模型：`resolution`、`dec_layers`、`num_queries`、`num_select`；
- Projector：`projector_type`、`projector_scale`、hidden-state indexes、各尺度 source indexes、source mode、C2f blocks、P5 mode、resample sharing；
- 训练尺度：`multi_scale`、`expanded_scales`、augmentation preset；
- 优化器：`lr`、`lr_encoder`、`weight_decay`、`lr_drop`、ViT layer decay、component decay；
- Decoder：Group DETR、CDN、iterative bbox refinement、scale routing、Dense O2O；
- 评估：`eval_max_dets`，COCO 默认固定为 100；
- Backbone refinement：refined initialization、LR scale、L2-SP anchor、online refinement mode/schedule/loss；
- 复现：data seed 与 `detector_init_seed` 分离。

### 4.3 分布式和评估修复

- `run_coco_subset.py` 已支持 `torchrun`/DDP，正确读取 `LOCAL_RANK/WORLD_SIZE`。
- 有效 batch 为 `batch_size * grad_accum_steps * world_size`；历史主协议固定 total batch 16。
- 修复了分布式训练/评估中的 rank、同步和 COCO 汇总路径。
- `maxDets=500` 是早期 PUMCH-AA 遗留值，COCO 对照必须显式或默认使用 100。早期非标准 maxDets 结果不能与标准 COCO AP 混用。
- 每轮运行记录完整 args，checkpoint 恢复时需保持结构参数一致。

### 4.4 实验快照

由于当前工作树长期处于 dirty 状态，关键批量实验会把 `src/` 与 runner 快照到 `experiment_snapshots/<experiment>/`，并记录 SHA-256。复现历史结果时应优先使用对应 snapshot，不能默认当前源码与历史训练源码完全相同。

## 5. 初期 Projector 对照

最初创建了 baseline、SimpleDINOv3Pyramid 和 Gated/BiFPN-style Projector 三条实验路径，并依次跑 smoke、overfit、quick、medium、strong。

结论：

- smoke/overfit 只能证明实现可运行和可学习，不能用来选最终结构；
- SimpleFPN 证明“强 backbone 下减少 hidden states”可行，但 strong 结果没有稳定保持 baseline 性能；
- Gated/BiFPN 增加了融合自由度，但收益不稳定，不能证明原 MultiScaleProjector 是主要瓶颈；
- 因此没有直接替换原 Projector，而是保留其高容量融合范式，转向受控地删除冗余 source、缩减 P5 和共享权重。

这一步的重要贡献是及时停止宽泛架构替换，建立后续单因素 Projector 消融的基线。

## 6. Decoder 与训练策略

本会话整合并继续使用 decoder 优化路线。详细来源见 `EXPERIMENT_SUMMARY_DECODER_OPTIMIZATION.md`。

### 6.1 Group DETR

对齐 medium screen 中，Group 6 优于 Group 13/8/7/4。Group 6 后续作为精度默认值，既降低训练冗余，也没有牺牲 AP。

### 6.2 Tuned CDN

选定参数：

```text
dn_number=50
dn_label_noise_scale=0.5
dn_box_noise_scale=0.6
dn_loss_coef=0.5
```

DN 100/200、更低或更高 loss coefficient、box noise 0.4/0.8 均没有超过该组合。Full COCO 中 CDN 对 Group 13 带来约 `+0.40 AP`，Group 6 + CDN 达到约 `51.55 AP`。

### 6.3 Iterative box refinement、scale routing 与 lr_drop

最关键的训练诊断是：早期 24 epoch medium 运行使用 `lr_drop=100`，实际上全程没有降学习率。许多旧实验因此受到了 schedule 影响，但相对方向仍有筛选价值，不能把绝对 AP 当作最终上限。

将 seed-43 epoch-22 最佳 checkpoint 用 `1e-5` 做一个低 LR finish epoch：

| 配置 | AP |
|---|---:|
| iterative refinement，epoch 22 | 35.3489 |
| iterative refinement + routing，epoch 22 | 35.4019 |
| iterative refinement + low-LR finish | 35.8974 |
| iterative refinement + routing + low-LR finish | **36.1423** |

最高结果 checkpoint：

```text
output/coco_medium_iterref_scaleroute_seed43_best22_lrd22_finish_e28_regb1/checkpoint_best_regular.pth
```

该结果已独立复评，详见 `MEDIUM_AP36_MILESTONE.md`。`backbone_register_border_tokens=1` 是该 checkpoint 的结构行为组成，复评时不能漏掉。

### 6.4 Dense O2O

Dense O2O 不改变普通 Hungarian O2O 的定义，而是通过 Mosaic、MixUp、CopyBlend 改变目标密度。当前增强 D2 路线两 seed 相对 strict Group6+CDN 平均提升约 `+0.349 AP`。CopyBlend `N=1` 是当前最好候选，但曾是单 seed promotion candidate；它后来被用于 integrated recipe。

Dense O2O 与 CDN/Group 会显著抬高显存，因为图像合成增加 GT 数量，CDN padding 再被 Group 复制。正式 full COCO 前仍应实现真正的 CDN GT/query cap，普通 Hungarian matching 不应丢 GT。

## 7. Projector 系统优化

### 7.1 尺度选择 A1

从所有尺度融合 `F2/F5/F8/F11` 改为：

```text
P3 <- F2/F5/F11
P4 <- F2/F5/F8/F11
P5 <- F2/F8/F11
```

功能 mask 实验确认非对称 source routing 可行，随后改为物理 prune，真正删除无用分支参数。直觉是：

- P3 保留浅层细节、过渡语义和最深语义；
- P4 作为中心尺度保留全部信息；
- P5 保留浅层定位、较深过渡和最终语义，删除冗余中间层。

A1 prune 两 seed mean 为 `35.768 AP`，参数 `37.476M`。它证明 P3/P4/P5 不需要对四个 hidden states 做完全相同的融合。

### 7.2 C2f 非对称精简

| P3/P4/P5 C2f blocks | 参数 | 两 seed mean AP | 结论 |
|---|---:|---:|---|
| 3/3/3 | 37.476M | 35.768 | 绝对精度默认 |
| 3/3/1 | 36.820M | 35.593 | 最佳轻量 C2f 候选 |
| 1/3/3 | 36.820M | 35.472 | P3 不宜优先精简 |
| 3/1/3 | 36.820M | 35.573 | 稳定但略低 |
| 1/1/1 | 35.507M | 35.009 | 明显过度压缩 |

最终精度主线保留 3/3/3。C2f 是参数的重要来源，但不是可以在三个尺度上一刀切缩减的纯冗余。

### 7.3 P5 grouped downsampling

P5 完整分支的主要参数来自四条 `384 -> 384`、3x3、stride-2 卷积和后续 C2f，不是 Upsample 本身。rank-64 先压缩再融合只保留约 39.7%-60.6% 的卷积谱能量，导致 APs/APl 明显下降。

最终先得到 `group2_first_full`：F2 P5 sampler 保持 dense，其余 sampler 使用 group 2。两 seed 为 `35.897/35.693 AP`，mean `35.795`，相对 full P5 减少 `1.991M` 参数。

### 7.4 Stage 3：P5 重采样权重共享

在 A1 prune 上进一步让 P5 的 F8/F11 group-2 3x3 kernel 共享，但保留 source-specific normalization；F2 保持独立 full convolution。

| 方案 | 参数 | Seed 42 | Seed 43 | Mean AP |
|---|---:|---:|---:|---:|
| A1 prune | 37.476M | 35.994 | 35.542 | 35.768 |
| S2，P5 F8/F11 sharing | **36.813M** | 35.902 | 35.537 | **35.719** |
| P3+P5 sharing | 36.223M | 35.347 | 35.648 | 35.498 |

S2 只损失 `0.048 AP` mean，删除 `0.664M` 参数，因此被选为当前 accuracy-efficiency 默认 Projector。

### 7.5 Stage 4/5：高效卷积和蒸馏

- DW+PW P5 虽进一步减参到 36.300M，但 seed 43 下降 `0.325 AP`；RTX 5090 batch1 Projector latency 还慢 2%。拒绝。
- Projector feature distillation `coef=0.5` 只恢复约 47% 的 DW+PW 损失，仍低于 S2，也无有意义 latency 优势。拒绝作为默认。

最终决策见 `RESAMPLE_SHARING_STAGE3_RESULTS.md`。

### 7.6 SDSR 支线

SDSR-v23/v36 证明另一种 semantic reassembly Projector 可以达到 36 AP。v36 是 v23 的前向等价执行重排，转换 checkpoint 达到 `36.059 AP`、`35.757M` 参数，并比旧 40.574M MSP 少 11.9% 参数。

但 native v36 训练路径受浮点累加顺序影响，seed 43 低于 v23；推荐“v23 训练 -> v36 转换部署”，不把 v36 native training 当当前默认。详见 `SDSR_V36_MILESTONE.md`。

## 8. Backbone refinement 整合

### 8.1 旧离线 refinement 的结论

外部 refinement 会话提供三条路线：

| 初始化 | 旧架构 AP | 相对旧 baseline | 主要收益 |
|---|---:|---:|---|
| UniRefiner balanced_zero3e | 33.88 | +0.27 | abnormal token ratio、AP75/APm |
| LazyStrike v2 e2 | 34.36 | +0.75 | 总体 AP、AP75、APl |
| Dense consistency e1 | 34.03 | +0.42 | AP75、APs、APm |

它们改变的是 backbone initialization，不增加推理模块。多目标 joint/sequential refinement 没有叠加，主要原因是 block 11 目标冲突和跨层 feature distribution 漂移。

完整背景见：

```text
handoff/BACKBONE_REFINEMENT_HANDOFF_CN.md
handoff/BACKBONE_REFINEMENT_STAGE_SUMMARY.md
```

### 8.2 与当前 S2 + decoder recipe 的严格整合

固定 S2 Projector、iterative refinement、legacy routing、Group6、tuned CDN、Dense O2O、data seed 42、detector init seed 1042，只改变 backbone initialization：

| Backbone init | AP | AP75 | APs | APm | APl | Last-5 AP |
|---|---:|---:|---:|---:|---:|---:|
| Official DINOv3 | 36.103 | 37.990 | 18.215 | 38.960 | 54.829 | 35.548 |
| LazyStrike | 36.043 | 38.067 | 17.465 | 39.499 | 54.479 | 35.551 |
| Dense consistency | **36.228** | **38.469** | 16.958 | **39.953** | 54.151 | **35.637** |

结论：

- LazyStrike 的旧 `+0.75 AP` 没有迁移到新架构；
- Dense consistency 仅 `+0.125 AP`，主要转向 AP75/APm，同时 APs/APl 下降；
- 二者均未达到预先设定的 `+0.20 AP + last5 + scale profile` promotion gate；
- 默认 backbone 仍是官方 DINOv3，dense-only 只是定位/中目标专项候选。

### 8.3 性质保留实验

针对“refined 特质在正常检测训练中是否丢失”，实现并验证：

- selected block LR 设为 0.1x；
- 对 refined initialization 做 L2-SP parameter anchor `1e-3`；
- 最终 backbone token diagnostic。

结果：低 LR 分别损失 `1.283/2.522 AP`；L2-SP 分别损失 `0.369/0.476 AP`。LazyStrike 的前景聚合只被部分保留，Dense 的有用行为没有被 parameter distance 留住。

结论：离线 refine 更像 initialization / preconditioning。参数距离不等于行为距离，强行冻结或拉回 refined 参数会阻碍 detector adaptation。

### 8.4 训练完成后再 refine

从 official integrated detector 最佳 backbone 出发，分别做 post-hoc LazyStrike 和 Dense refinement，再替换回 detector 且不重训其余模块：

| 方案 | Medium AP |
|---|---:|
| 原 detector | 36.103 |
| post-hoc Dense | 35.8 |
| post-hoc LazyStrike | 35.6 |

直接 post-hoc refine 破坏了 backbone-projector-decoder 的协同，不能作为“训练后免费加点”方法。相关目录：`output/posthoc_backbone_refinement/`。

### 8.5 检测训练期间的短程行为约束

为避免离线性质随训练消失，实现三种 online refinement mode，默认只在 epoch `[0,6)` 线性衰减：

- `object_token`：直接在检测训练期间约束 GT box 内 token 与 object aggregate；
- `object_local`：更强调对象内部局部一致性；
- `feature_teacher`：以 refined/frozen teacher 对实际被 Projector 使用的 dense feature 做前景/背景加权保持。

Stage 1：

| 方案 | AP | Last-5 |
|---|---:|---:|
| official + object_token c=0.1 | 36.227 | 35.646 |
| dense init + feature_teacher c=1.0 | 36.066 | 35.279 |

Stage 2：

| 方案 | AP | AP75 | APl | Last-5 |
|---|---:|---:|---:|---:|
| object_local c=0.1 | 35.753 | 37.794 | 54.222 | 35.294 |
| object_token c=0.05 | **36.2768** | **38.7647** | **54.9547** | **35.5887** |

当前判断：

- 在线 object-token 的低权重短程约束比离线 refine + 强 retention 更合理；
- `c=0.05` 相对固定 official integrated reference `36.103` 约 `+0.174 AP`；
- 该结果只有 seed 42，未达到“稳定默认”证据；
- object-local 和 feature-teacher 当前均不值得继续做宽 sweep；
- 下一步优先复测 `object_token c=0.05` 的 seed 43，再考虑调短 epoch 0-4 或只约束 F8/F11。

当前在线候选 checkpoint：

```text
output/online_backbone_refinement_stage2/
object_global_e0_6_c0p05_seed42/checkpoint_best_regular.pth
```

## 9. 当前推荐 medium 配方

以下是当前最可信的 detector 主线，而不是所有可选功能的并集：

```text
Backbone: DINOv3-S/16 official weights
Resolution: 640
Hidden states: F2/F5/F8/F11

P3 sources: F2/F5/F11
P4 sources: F2/F5/F8/F11
P5 sources: F2/F8/F11
Projector source mode: prune
C2f blocks: 3/3/3
P5 mode: group2_first_full
P5 resample sharing: F8/F11 (S2)

Decoder layers: 4
Queries/select: 300/300
Group DETR: 6
Deformable attention points: 2
Iterative bbox refinement: shared
Scale routing: legacy

CDN: enabled
DN number: 50
DN box/label noise: 0.6/0.5
DN loss coefficient: 0.5

Dense O2O: enhanced
Mosaic/MixUp phase: epochs 2-11
CopyBlend tail: epochs 2-20
CopyBlend objects: 1

Multi-scale + expanded scales: enabled
LR detector/encoder: 1e-4 / 1.5e-4
LR drop: epoch 20 for ordinary 24e runs
Weight decay: 1e-4
ViT layer/component decay: 0.8 / 0.7
Total batch size: 16
Register border tokens: 1
COCO eval maxDets: 100
```

Online `object_token c=0.05, epochs 0-6` 暂列候选开关，尚未并入稳定默认。

## 10. 关键 checkpoint 与结果

### 10.1 单次最高低 LR finish

```text
output/coco_medium_iterref_scaleroute_seed43_best22_lrd22_finish_e28_regb1/
checkpoint_best_regular.pth
```

AP `36.1423`。适合证明 iterative refinement + routing + 正确 schedule 的上限。

### 10.2 当前 integrated official reference

```text
output/integrated_refinement_s2_denseo2o/official_seed42/
checkpoint_best_regular.pth
```

AP `36.1027`，参数 `36.813M`。这是离线 refinement 对照的固定 reference。

### 10.3 Integrated dense-only candidate

```text
output/integrated_refinement_s2_denseo2o/dense_only_seed42/
checkpoint_best_regular.pth
```

AP `36.2276`，但 scale profile 偏向 AP75/APm，不是默认 backbone。

### 10.4 当前 online object-token candidate

```text
output/online_backbone_refinement_stage2/
object_global_e0_6_c0p05_seed42/checkpoint_best_regular.pth
```

AP `36.2768`。这是当前观察到的 integrated recipe 最高 seed-42 候选，也是本次效率测量使用的 RF-DETR-DINOv3 checkpoint。

### 10.5 SDSR-v36 converted checkpoint

```text
output/coco_medium_v36_converted_from_v23_ap36p059/
checkpoint_best_regular.pth
```

AP `36.059`，参数 `35.757M`，适合研究另一条部署效率路线。

## 11. 本机 GFLOPs 与 latency

最新统一测量目录：

```text
output/local_benchmarks_20260824/
```

协议：RTX 5090、batch 1、PyTorch eager、纯 forward、不含预处理/后处理、20 warmup、100 repeats、每次 forward 后 200 ms buffer。DEIMv2 在 `deploy()` 后测量。

| 模型 | 输入 | 参数 | GFLOPs，FMA=1 | GFLOPs，FMA=2 | FP32/TF32 mean | AMP FP16 mean |
|---|---:|---:|---:|---:|---:|---:|
| RF-DETR-DINOv3 online c0.05 | 640 | 36.813M | 86.493 | 172.985 | 22.82 ms | 28.78 ms |
| DEIMv2-L | 640 | 32.178M | 72.247 | 144.495 | 20.68 ms | 29.36 ms |
| RF-DETR-Medium | 640 | 33.687M | 50.102 | 100.204 | 16.67 ms | 20.62 ms |
| RF-DETR-Medium native | 576 | 33.687M | 39.412 | 78.825 | 13.43 ms | 20.53 ms |

解释边界：

- 这是本机 eager 工程对照，不是 RF-DETR 论文 TensorRT/T4 latency 的复现；
- 三项目使用不同 PyTorch/CUDA build；
- 200 ms buffer 会让 GPU 降频，mean/P50/P90 应一起保留；
- eager autocast FP16 较慢不能推断 TensorRT FP16 也较慢；
- 正式论文效率表应统一 TensorRT/CUDA stack，并修复当前 shared iterative refinement 的 export 兼容性。

工具：`tools/benchmark_standardized_latency.py`。最新 JSON 比旧 Markdown 中的早期同协议表更权威。

## 12. 已否定或不应重复的路线

1. 不再直接用 SimpleFPN 或现有 Gated/BiFPN 替换 MultiScaleProjector；strong 证据不支持。
2. 不把 smoke/overfit/quick 的高点当作结构收益，Projector 候选至少看 medium/strong。
3. 不再忘记 `lr_drop`；旧 `lr_drop=100` 运行可用于相对筛选，不能用于最终上限结论。
4. 不使用 COCO `maxDets=500` 与标准结果比较。
5. 不把 RF-DETR 官方 detection-pretrained checkpoint 与仅加载 DINOv3 image-pretrain 的模型解释为完全公平训练对照。
6. 不对 P3/P4/P5 全部缩成 C2f 1 block；P3 尤其不宜先缩。
7. 不对 P3 和 P5 同时共享 resampling kernel；该收益不稳定。
8. 不使用当前 DW+PW P5 作为默认；既掉 AP，也没有 batch1 latency 优势。
9. 不继续对已失败的 DW+PW student 做大范围 Projector distillation sweep。
10. 不假设 LazyStrike、Dense consistency、UniRefiner 的旧增益可直接相加。
11. 不使用 0.1x backbone LR 或 L2-SP 参数 anchor 保留 refinement；二者都降低 AP。
12. 不在 detector 训练完成后直接 refine backbone 并无训练替换回去；会破坏模块协同。
13. 不以 token heatmap、CLS coverage 或 abnormal-token ratio 代替 detection AP/last-5/scale AP。
14. 不把 online c0.05 的单 seed `36.2768` 写成已稳定提升。

## 13. 下一步路线

按投入产出排序：

### Priority 1：确认在线短程约束

1. 用完全相同的 S2 + decoder recipe 跑 `object_token c=0.05` seed 43；
2. 同 seed 必须有 official/no-online control，固定 detector init seed；
3. 比较 peak、last-5、AP75、APs/APm/APl，而不仅看最高 AP；
4. 若仍为正，再做 epoch `[0,4)` 对 `[0,6)` 单因素；
5. 只有两 seed mean/last-5 均改善，才并入默认 recipe。

### Priority 2：行为约束而非参数约束

若 Priority 1 仅有微弱或不稳定收益，下一版应：

- 直接约束 Projector 实际消费的 F8/F11 patch features；
- 按 small/medium/large box 分配损失，不让大目标主导；
- 对前景做局部 equivariance/teacher preservation，背景只给很低权重；
- 在前 4-6 epoch 衰减到零，避免后期阻碍 task adaptation；
- 先测 objective gradient cosine，再决定是否做分层 gradient routing。

### Priority 3：Full COCO 晋级门槛

只有 medium 两 seed 通过后才上 full COCO。Full 前先完成：

- CDN GT/query hard cap，控制 Dense O2O 显存；
- 统一标准 `maxDets=100`；
- 固定训练总 batch、epochs、lr_drop 和 augmentation；
- 记录 params、FLOPs、FP32/FP16 latency，不只报告 AP。

### Priority 4：效率部署

- 修复 shared iterative bbox refinement 的 ONNX/export path；
- 用统一 TensorRT 10.x + CUDA Graphs 测三模型 FP16；
- 同时报 end-to-end 与 model-only latency；
- 保留 S2 MSP 和 SDSR-v23->v36 两条候选，不再扩展无边界的 v37+ 搜索。

## 14. 复现入口

| 任务 | 入口 |
|---|---|
| COCO 分层训练/评估 | `run_coco_subset.py` |
| 当前 integrated 三 backbone 对照 | `run_integrated_refinement_s2_denseo2o.sh` |
| Refinement retention | `run_backbone_refine_retention_stage1.sh` |
| Post-hoc refine | `run_posthoc_backbone_refinement.sh` |
| Post-hoc corrected eval | `run_posthoc_backbone_refinement_eval.sh` |
| Online refine Stage 1 | `run_online_backbone_refinement_stage1.sh` |
| Online refine Stage 2 | `run_online_backbone_refinement_stage2.sh` |
| P5 轻量稳定性 | `run_cheap_p5_medium.sh` |
| Scale-selective Projector | `run_scale_selective_msp_*.sh` |
| C2f asymmetric | `run_c2f_asymmetric_*.sh` |
| Resample sharing | `run_resample_sharing_*.sh` |
| Projector benchmark | `tools/benchmark_sdsr_projector.py` |
| 三模型统一 benchmark | `tools/benchmark_standardized_latency.py` |

## 15. 关键文档索引

- `MEDIUM_AP36_MILESTONE.md`：首次 36.1423 AP 与 lr_drop 诊断；
- `CHEAP_P5_MEDIUM_STABILITY.md`：P5 grouped 结构与双 seed 稳定性；
- `C2F_ASYMMETRIC_STAGE2_RESULTS.md`：C2f 3/3/3、3/3/1 等消融；
- `RESAMPLE_SHARING_STAGE3_RESULTS.md`：S2 sharing、DW+PW、Projector distillation；
- `SDSR_V36_MILESTONE.md`：SDSR-v23/v36 精度、等价转换和效率；
- `EXPERIMENT_SUMMARY_DECODER_OPTIMIZATION.md`：Group/CDN/Budgeted-SA/Dense-O2O；
- `INTEGRATED_REFINEMENT_PLAN.md`：当前架构 × official/Lazy/Dense 严格矩阵；
- `BACKBONE_REFINE_RETENTION_RESULTS.md`：LR scale、L2-SP 和 token retention；
- `handoff/BACKBONE_REFINEMENT_HANDOFF_CN.md`：离线 refinement 完整历史；
- `handoff/BACKBONE_REFINEMENT_STAGE_SUMMARY.md`：refinement 阶段英文摘要；
- `STANDARDIZED_LATENCY_200MS.md`：早期统一 latency 协议说明；
- `output/local_benchmarks_20260824/*.json`：最新 benchmark 原始结果。

## 16. 工作树注意事项

当前仓库存在大量未提交修改和未跟踪实验文件。不要执行 `git reset --hard`、`git checkout -- .` 或批量清理；其中混有用户和其他会话的工作。

开始新实验前：

1. 阅读本文件和目标阶段文档；
2. 检查 `git status --short`；
3. 为实验创建独立 `experiment_snapshots/<name>`；
4. 保存 runner、`src/`、启动参数、checkpoint hash 和环境信息；
5. 检查 GPU 上已有进程，不终止无关任务；
6. medium 至少固定 data seed 和 detector init seed；
7. 结果汇总同时保留 best epoch、last-5 和 APs/APm/APl。

## 17. 最终阶段判断

1. 将 DINOv2 换成 DINOv3 后的性能缺口不是单一模块造成的，而是 backbone feature distribution、Projector、多尺度 memory、decoder refinement 和训练 schedule 的耦合问题。
2. 最大的 medium 增益来自正确的训练/decoder 配方与多尺度利用，而不是盲目增加参数。
3. 参数减少后性能提高，是因为删除了尺度不匹配和重复 source 路径，同时保留 P4 全信息中心、P3 细节与 P5 语义所需容量；这是一种更合适的归纳偏置，不是“越小越强”的普遍规律。
4. 当前 S2 Projector 在 36.813M 参数下是最可信的 accuracy-efficiency 主线。
5. Offline backbone refinement 的可视化性质可以改善，但正常检测训练会重塑这些性质；其价值更接近初始化和机制探针。
6. 当前最有希望的新方向是训练前期、低权重、直接作用于检测所消费 patch feature 的行为约束。online object-token c0.05 已给出单 seed 正信号，下一步必须先做第二 seed。
7. 在第二 seed 之前，默认发布/对照模型仍应使用 official DINOv3 + S2 Projector + 当前 decoder recipe，而不是把最新单次最高点写成稳定结论。

