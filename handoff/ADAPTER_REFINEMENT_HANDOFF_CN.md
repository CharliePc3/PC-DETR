# RF-DETR-DINOv3 全会话工作与贡献交接文档

> 更新时间：2026-08-31（Asia/Shanghai）
> 项目：`/data/cpc/root/project/RF-DETR-DINOv3`
> 当前分支：`exp/scale-decoupled-projector`
> 写入时 HEAD：`4aec8c0c21a2c2ea9f6e3b8e2c999477729a8783`
> 文档定位：从项目架构梳理、CDN/Group、Backbone refinement 到 SDSR-v1-v46、匹配策略整合和 COCO Full 的完整会话交接

## 1. 文档用途和权威边界

这份文档记录本次长会话从最初理解 RF-DETR-DINOv3 开始，到当前 SDSR-v40
干净配置 Full COCO 训练为止的完整工作贡献。它不是后半程摘要，也不假设接手者已经
阅读项目根目录下的临时记录。

以下文档仍然有价值，但在本文中只作为事实核对来源，不替代本文正文：

- `SDSR_PROJECTOR.md`：SDSR-v1 最初的设计说明；
- `SDSR_V36_MILESTONE.md`：v23 权重转换为 v36、等价性和效率测试；
- `handoff/MATHCER_REFINEMENT_HANDOFF_CN.md`：CDN、Group、SA-Matching、Dense O2O 专项；
- `handoff/BACKBONE_REFINEMENT_HANDOFF_CN.md`：UniRefiner、LazyStrike、dense consistency 专项；
- `handoff/ARCHITECTURE_REFINEMENT_HANDOFF_CN.md`：并行架构实验的汇总；
- `handoff/SDSR_V40_MATCHING_INTEGRATION_HANDOFF_CN.md`：SDSR 后半程和匹配策略专项。

本文的特殊价值是完整保留 SDSR 的因果链：为什么最初选择彻底替换
`MultiScaleProjector`，为什么 v1 明显掉点，怎样从 v2 一路恢复到 v23，为什么又在
v37-v46 中回到 v40，以及哪些结果受到共享工作树和另一会话实现的影响。

## 2. 十分钟执行摘要

本会话完成的主要工作如下。

1. 系统阅读和对比 LW-DETR、RF-DETR、RF-DETR-DINOv3 与 DEIMv2-DINOv3，明确
   ViT 单尺度 token 到检测多尺度 memory 的接口问题。
2. 在 RF-DETR-DINOv3 Decoder 中完整接入 DINO 风格 CDN，支持正负去噪 Query、
   attention mask、辅助层 DN loss、Group DETR、分布式训练和恢复。
3. 建立 unique output directory、COCO subset/full runner、实验快照、日志解析和后台
   队列，减少消融相互覆盖和并行会话污染。
4. 建立 DINOv3 token diagnostic：CLS-patch、LAST stability、前背景比、多目标覆盖、
   abnormal ratio、PCA 和热力图，并验证这些指标只能解释模型，不能代替下游 AP。
5. 完成 UniRefiner、LazyStrike 和 detection-aware dense consistency 三条离线 backbone
   refinement 路线。旧统一检测协议下，三者最佳单独增益分别约 `+0.27`、`+0.75`、
   `+0.42 AP`，但组合通常不叠加。
6. 设计 SDSR，将不同尺度的语义重组问题解耦：P3 负责局部/高分辨率恢复，P4 作为
   语义锚点，P5 负责抗混叠和相位感知下采样。
7. SDSR-v1 几乎完全替换原 `MultiScaleProjector`，参数仅约为原 Projector 的 5.2%，
   但 Medium AP 从 `34.0308` 降至 `30.6-31.35`，证明过度删除多层高容量融合不可行。
8. v2-v23 通过保留 layer identity、scale-first fusion、恢复 C2f、P5 two-basis 和深层
   grouped detail，逐步把性能恢复到 `36.0590 AP`；v23 成为 SDSR 第一代成熟骨架。
9. v36 将 v23 的重复算子打包，转换后的 v23 checkpoint 前向最大误差
   `2.861e-6`，复现 `36.059 AP`，总参数比 MSP 少 11.9%，GFLOPs 少 2.1%。
10. v37-v46 系统验证 routed C2f、尺度深度先验、空间条件路由、RepC3/C3k2 和退火。
    最终选择 `sdsr_v40_p4_learnable`，strict seed43 为 `35.9690 AP`。
11. 在 v40 上联合搜索 Dense O2O、Group 和 CDN。历史 Reg1 配置达到 `36.9718 AP`，
    但发现它误保留 `backbone_register_border_tokens=1`；严格关闭后可信结果为
    `36.4706 AP`。
12. 当前有效 Full COCO 使用 Reg0、SDSR-v40、Dense no-MixUp、Group6、CDN25、
    total batch16、24e、lr_drop20，不使用 EMA。

最重要的结论不是“新模块越复杂越好”，而是：

> DINOv3 多层特征的高容量融合是检测性能的核心；有效的轻量化应围绕各尺度职责做
> 结构化替换，而不是把 C2f、多层 identity 和跨通道 mixing 一次性全部删除。

## 3. 项目继承关系与初始架构认知

### 3.1 模型继承关系

```text
LW-DETR
  -> RF-DETR
       - DINOv2 ViT backbone
       - Group DETR / DETR decoder
       - MultiScaleProjector
  -> RF-DETR-DINOv3
       - 将 backbone 替换为 DINOv3
       - 继续保留和改造 projector、decoder、matching/training recipe
```

RF-DETR-DINOv3 的主数据流为：

```text
image
  -> DINOv3 intermediate hidden states
  -> backbone-to-decoder projector
  -> P3/P4/P5 detector memories
  -> deformable transformer encoder/decoder
  -> class and box heads
```

ViT 的 patch grid 默认约为输入的 stride 16。检测器需要多尺度 memory，因此必须同时
解决三个问题：

1. 把 DINOv3 多个隐藏层的通道对齐到 decoder hidden dimension；
2. 从同一空间分辨率构造 P3/P4/P5；
3. 在重采样过程中保留语义、边界和小目标信息。

### 3.2 RF-DETR 与 DEIMv2 的多尺度差异

RF-DETR 路线：

- 从多个 ViT hidden layers 取特征；
- `MultiScaleProjector` 分别重采样到目标尺度；
- 通过 C2f 融合多层特征；
- 结果直接交给后续 transformer/decoder。

DEIMv2-DINOv3 路线：

- 把多个 ViT 层映射或 Resize 到多个尺度；
- 使用极轻量 CNN/STA 分支补充局部细节；
- 由 HybridEncoder 继续做多层多尺度融合；
- 再把编码结果送入 decoder。

因此，“DEIMv2 没有 projector”只在命名上成立。它把通道/尺度适配和多层融合分散在
backbone bridge、STA 和 HybridEncoder 中；RF-DETR 则把这项职责集中在
`MultiScaleProjector`。两者都不是简单双线性 Resize。

### 3.3 一开始得到的结构判断

1. 多层 ViT feature fusion 很重要，不能只看最终层。
2. P3、P4、P5 的任务不同，不适合复制同一种上/下采样块。
3. P3 对小目标和边界敏感，需要局部、相位和高分辨率恢复能力。
4. P4 保持原生 token stride，是最稳定的语义锚点。
5. P5 下采样必须控制 aliasing，但不能过度低通而丢失大目标边界。
6. CNN 细节可以作为重组权重的条件，而不一定直接拼接到 detector feature。
7. Projector 的轻量化不能只看参数量；融合容量、优化稳定性和实际 GPU kernel 效率同样重要。

## 4. 实验工程和版本控制贡献

### 4.1 分支演进

本会话涉及的主要分支顺序为：

```text
exp/cdn-denoising
  -> exp/backbone-refine
  -> exp/lazystrike-refinement
  -> exp/scale-decoupled-projector
```

用户采用分支而不是复制整个目录，保证能够回退和比较。后续将前面分支的必要提交合并
到 `exp/scale-decoupled-projector`，因此当前分支设计目标是同时包含：

- CDN/Group；
- backbone refinement 的加载和诊断支持；
- LazyStrike/dense refinement 工具；
- SDSR Projector 全系列；
- iterative box refinement、scale routing 和 Dense O2O 的调用入口。

### 4.2 统一 runner 和输出隔离

`run_coco_subset.py` 被扩展为统一入口，支持：

- quick/medium/full COCO；
- projector scale/type；
- multi-scale 和 expanded scales；
- Group、CDN、iterative box refinement、scale routing；
- Dense O2O；
- refined backbone checkpoint；
- register border；
- EMA；
- seed 与 detector initialization seed；
- distributed/background/resume；
- unique output directory。

unique output directory 是早期必须补的工程项。它避免多组消融覆盖同一个 checkpoint、
`log.txt` 或可视化目录，也使后续脚本队列可以按运行名自动判断完成状态。

### 4.3 冻结实验快照

因为项目被多个会话共享，直接以当前工作树复现实验并不可靠。后半程采用
`experiment_snapshots/` 固定源码和配置，关键快照包括：

```text
experiment_snapshots/sdsr_v40_strong_prior_medium_seed43
experiment_snapshots/sdsr_v46_denseo2o_medium_seed43
experiment_snapshots/sdsr_v40_match_final_seed43
```

后续引用结果时，应优先检查运行目录中的 `args.json`、日志和对应 snapshot，而不是只看
当前 `git diff`。

## 5. Decoder 路线：CDN 与 Group DETR

### 5.1 原理澄清

本会话阅读和讨论 DN-DETR、DINO 与 Stable-DINO 后，明确了以下边界：

- DN-DETR：向 decoder 注入带噪 GT query，提供已知匹配的直接分类/回归路径；
- DINO CDN：同时构造较小噪声正样本和较大噪声负样本，正样本重建 GT，负样本学习
  no-object；
- DINO Mixed Query Selection：encoder top-K 只提供 anchor/reference boxes，content
  query 仍是可学习 embedding；
- Stable-DINO：通过更稳定的匹配和位置质量监督减少多优化路径冲突；
- Mixed Query Selection 与 Stable-DINO 都关心优化路径质量，但不是同一种机制。

因为 DN 与 CDN 的接入接口基本一致，项目直接实现 CDN，没有先维护一个临时 DN 分支。

### 5.2 CDN 代码实现

关键文件：

```text
src/rfdetr/models/dn_components.py
src/rfdetr/models/lwdetr.py
src/rfdetr/models/transformer.py
src/rfdetr/config.py
src/rfdetr/main.py
run_coco_subset.py
```

实现内容：

1. 根据 batch targets 构造正/负 CDN query；
2. 支持 label noise 和 box noise；
3. 构造 DN groups 之间、DN 与 matching query 之间的 self-attention mask；
4. 兼容 Group DETR 的 query 排布和 reshape；
5. decoder 输出后拆分 DN 与正常 query；
6. 为最终层和所有 auxiliary decoder layers 计算分类、L1、GIoU DN loss；
7. 支持独立 `dn_loss_coef` 和负样本分类项；
8. 推理时完全移除 DN query，不增加推理开销；
9. 支持日志、checkpoint 和分布式 resume。

### 5.3 早期 CDN 消融

12-epoch Medium 初筛：

| 配置 | AP |
|---|---:|
| no CDN | 25.04 |
| 初始 CDN | 25.13 |
| DN loss 0.5 | 25.39 |
| loss 0.5 + box noise 0.6 | 25.60 |
| loss 0.5 + box noise 0.6 + DN 50 | **26.29** |
| loss 0.25 | 24.69 |
| box noise 0.4 | 25.47 |
| box noise 0.8 + DN 50 | 25.67 |
| DN 200 | 25.14 |

早期结论是 `dn_number=50, box_noise=0.6, loss_coef=0.5`。这个结论适用于当时的
Projector 和训练协议，不应自动视为后续 SDSR-v40 的最优值；v40 联合搜索最终选择
DN25。

### 5.4 Group 兼容和数量搜索

Group DETR 在训练期把 query 分成多个独立组，每组做严格 O2O Hungarian matching；
推理只使用第一组。它增加训练正匹配和收敛速度，但提高训练计算/显存，不改变推理 query 数。

CDN 最初只在 Group1/Medium smoke，随后完成多 Group 的 query 布局、loss 归一化、
分布式训练和 checkpoint resume。

旧 18e、total batch16 消融：

| Group | AP |
|---:|---:|
| 13 | 32.70 |
| 8 | 32.49 |
| 7 | 32.55 |
| 6 | **32.99** |
| 4 | 32.29 |

Group6 因此替代官方常用的 Group13，既降低训练成本，也在当时协议下提高 AP。

旧 Full COCO 对照：

| 配置 | AP | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| Group13，无 CDN | 51.10 | 69.98 | 55.03 | 29.65 | 55.83 | 70.91 |
| Group13 + CDN | 51.50 | 70.18 | 55.65 | 29.76 | 56.24 | 71.03 |
| Group6 + CDN | **51.55** | 70.17 | **55.85** | **30.18** | 56.12 | **71.08** |

这证明 CDN 和 Group6 在完整 COCO 上有效，但这些数字属于早期模型，不是 SDSR 的结果。

## 6. Backbone 诊断工具链

### 6.1 为什么先做诊断

引入 UniRefiner 和 LazyStrike 前，需要先判断 DINOv3-small 的 patch token 是否真的存在：

- 高幅值或全局传播的异常 token；
- CLS 只聚焦最大目标；
- 多目标覆盖不足；
- 翻转或局部扰动不稳定；
- foreground/background 聚合失衡。

因此实现：

```text
tools/analyze_dinov3_tokens.py
```

### 6.2 已实现指标

1. **CLS-patch cosine Point-in-Box**：CLS 与每个 patch 的 cosine score，检查最强或
   top-k 响应是否落入任一 GT box。
2. **LAST-style patch stability vote Point-in-Box**：比较多视图/翻转下稳定响应 patch，
   检查稳定区域是否落在目标内。
3. **Foreground/background score ratio**：GT box 内外平均响应比；只能衡量标注前景，
   未标注真实物体不能简单当背景。
4. **多目标覆盖**：top-k 覆盖多少不同 GT box，并分 small/medium/large 统计。
5. **Largest-box dominance**：响应是否几乎全部被最大 GT box 吸收。
6. **Flip consistency**：翻转后映射回原坐标的 patch feature/score 一致性。
7. **FP/GP proxy**：基于当前 feature 分布的简化动态异常代理。
8. **Fixed abnormal-token ratio**：固定阈值口径，用于 refine 前后横向比较。
9. **PCA/heatmap visualization**：最后层或指定层 patch features 的共享 PCA 和响应热力图。

### 6.3 多目标 COCO 的解释限制

厨房示例揭示了一个重要问题：最强 CLS 响应和 top-10 可以全部落入 GT box，却集中在
最大的炉灶/操作台区域，忽略桌面水果等其他目标。LAST-ViT 常用的单目标、干净背景图
无法充分暴露这个问题。

因此：

- Point-in-Box 高不等于多目标覆盖好；
- 前景强响应也可能被 FP/GP proxy 判异常，因为 proxy 检测的是 feature 统计异常，不是
  语义背景；
- COCO 未标注物体不能强制视作 negative/background；
- dynamic proxy 和 fixed ratio 不能混用；
- token diagnostic 只能用于解释，模型选择必须依赖 AP/AP75/尺度 AP 和训练稳定性。

## 7. UniRefiner 路线

### 7.1 接入思路

UniRefiner 采用离线 teacher-student refinement：

```text
frozen DINOv3 teacher
  + trainable DINOv3 student
  + image border register patches
  + FP/GP/AH clean-spurious filtering
  + NCE/register absorption/uniformity/SCD
  -> refined backbone checkpoint
  -> 加载回 detector
```

它不是在 RF-DETR 推理图中永久增加一个 decoder 模块。Gaussian/zero/rand/randn 外圈
主要在 refinement 中作为异常信息的吸收区域；是否在 detection 输入继续保留 border
是另一项独立消融，不能与权重 refinement 混为一谈。

### 7.2 完成的实验范围

本会话完成：

- DINOv3-small registry 和 UniRefiner 训练适配；
- zero/rand/randn register；
- SCD on/off；
- AH on/off；
- 1/2/3 epoch；
- COCO train、CC3M5k、ADE20K images；
- COCOval1k、miniImageNet1k 和 CC3M 测试；
- abnormal-token ratio、PCA、heatmap；
- COCO Medium detection；
- ADE20K frozen-backbone linear segmentation。

### 7.3 最佳保留配置与结果

`balanced_zero3e`：

```text
images=COCO train2017
resolution=640
batch_size=8
epochs=3
precision=BF16
LoRA=linear-r8
lr=1e-4
weight_decay=0.1
register_factor=37
register_type=zero
fp_gp_threshold=0.5
adaptive_register_threshold=0.55
AH=False
num_proposals=3
uniformity=0.3
SCD_start=0.1
SCD_weight=0.4
```

旧统一 detector 结果：

| Backbone | AP | AP50 | AP75 | APS | APM | APL | Last-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original DINOv3-S | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |
| balanced_zero3e | 33.88 | 50.96 | 35.95 | 17.09 | 36.58 | 51.94 | 33.46 |

fixed abnormal ratio 从 `9.62%` 降到 `5.75%`，但可视化变化弱于数值变化；检测 AP
提高约 `0.27`，ADE20K linear segmentation 反而从 `40.89` 降到 `40.75 mIoU`。

### 7.4 为什么论文现象没有完全复现

1. 自定义早期 FP/GP proxy 与论文 Figure 2 的筛选口径不同，不能直接比较“50%”与
   论文小于 10% 的数字。
2. DINOv3-small 本身可能比更大 ViT 少极端异常 token，能清理的空间有限。
3. COCO 图像复杂、多目标且存在未标注物体，异常判定比 CC3M/单目标可视化更困难。
4. 1 epoch 只是快速验证，不等于论文的 few epochs；后续已补 2/3 epoch。
5. abnormal ratio 下降只表示 feature redistribution，不保证 detector 或分割 linear probe
   一定提高。

### 7.5 Detection register border 的教训

曾实现 detection 输入外圈 register：它参与 backbone self-attention，在进入 Projector 前
裁掉，不直接参与检测 prediction。这个实验说明 border 可能改变 feature distribution，
但最终方法没有采用它。

后期发现匹配搜索脚本误保留 `backbone_register_border_tokens=1`，导致最高 AP 被污染。
最终纯方法必须显式设置为 0。此问题是全会话最重要的实验卫生教训之一。

## 8. LazyStrike 与 detection-aware dense consistency

### 8.1 LazyStrike 的离线 refine 实现

LazyStrike 不是在 detector 中永久增加 Aggregator 模块，而是离线优化 DINOv3 student
最后若干 block，使 CLS 更稳定地聚合前景 patch。针对 COCO 多目标场景，训练目标加入：

- box-balanced foreground aggregation；
- object coverage；
- horizontal-flip CLS consistency；
- frozen-teacher distillation；
- 避免把全部未标注区域当背景的保守监督。

主工具：

```text
tools/train_dinov3_lazystrike_refine.py
```

### 8.2 最佳 LazyStrike 结果

最佳 `v2-conservative e2`：

```text
COCO random 5k images
resolution=640
epochs=2
batch_size=2
FP32
train_blocks=10-11 + final norm
lr=5e-6
weight_decay=0.05
max_boxes_per_image=12
lazy_topk=1
target_weight=0.1
align/cover/consistency/distill=0.25/0.15/0.25/1.0
```

| Backbone | AP | AP50 | AP75 | APS | APM | APL | Last-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |
| LazyStrike v2 e2 | **34.36** | **51.71** | 36.37 | 16.77 | 36.48 | **52.97** | **33.71** |

它相对 base 提高 `0.75 AP`、`1.13 AP75`、`0.98 APL`，但 APS 低 0.10；seed43
仍为正收益但幅度缩小。

### 8.3 Dense consistency

为了直接优化 Projector 实际使用的 patch features，另实现 detection-aware dense
consistency：

1. 提取原图 student、翻转 student 和 frozen teacher 的多层 patch features；
2. 把翻转 feature grid 映射回原坐标；
3. 对 block 8/11 做 teacher preservation；
4. 用 patch-box overlap 构造 object-aware consistency；
5. box 内平均后再按 box/image 平均，避免大框主导；
6. 只给低权重 global consistency，避免把未标注物体当背景。

最佳 `dense_only e1`：

| Backbone | AP | AP50 | AP75 | APS | APM | APL | Last-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 33.61 | 51.12 | 35.24 | 16.87 | 36.02 | 51.99 | 33.16 |
| Dense-only | 34.03 | 50.94 | **36.75** | **17.36** | **36.85** | 51.95 | 33.56 |

LazyStrike 更偏总体 AP/大目标，dense consistency 更偏 AP75/小中目标定位。

### 8.4 组合为何没有叠加

| 组合 | AP | 主要现象 |
|---|---:|---|
| Lazy + dense simultaneous | 33.72 | final block 目标冲突 |
| Lazy -> dense | 34.01 | 擦除部分 Lazy 收益 |
| Dense -> Lazy | 33.94 | 转向大目标、APS 下降 |
| Uni + dense | 34.11 | 仅轻微超过 dense peak |
| Uni -> Lazy | 34.04 | 低于 standalone Lazy |
| Uni + Lazy | 33.65 | 无正向迁移 |

原因是三条路线并非正交：它们都会改变后部 ViT block 的统计，分别强调 register
absorption、CLS 前景聚合和 patch 等变一致性。简单叠加 loss 或顺序 refine 会破坏
backbone-projector-decoder 已形成的协同。

### 8.5 ADE20K linear segmentation

实现：

```text
tools/linear_seg_ade20k.py
```

20e frozen-backbone linear probe：

| Backbone | mIoU |
|---|---:|
| Original DINOv3-S | 40.89 |
| UniRefiner author COCO | 40.77 |
| UniRefiner balanced_zero3e | 40.75 |
| UniRefiner randn3e | 40.27 |
| ADE20K-image UniRefiner | 40.76 |
| LazyStrike v2 | 40.83 |

因此这些 refinement 主要是检测专门化，不足以声称通用 dense representation 提升。

## 9. 从相关工作形成 SDSR 的过程

本会话阅读或结合源码讨论了：

- DEIMv2-DINOv3 与 STA；
- UPLiFT local attenders；
- ViT-Up faithful feature upsampling；
- RaysUp geometry-aware ray representation；
- Spatial Frequency Modulation；
- CROWn anti-aliased downsampling/phase-calibrated fusion。

最终没有把所有方法机械拼接。提炼出的设计原则是：

1. P3 使用局部、内容相关、尽可能 faithful 的语义上采样；
2. 图像局部细节只作为重组条件，避免直接注入低级噪声；
3. P4 尽量保留原生 stride16 语义；
4. P5 采用抗混叠、相位感知且轻量的下采样；
5. 不同 ViT 深度对不同尺度应有不同贡献；
6. 必须保留足够的跨层/跨通道融合能力；
7. 创新点应是尺度职责解耦和语义重组，而不是论文模块的简单合集。

## 10. SDSR-v1：最初的完整替换方案

### 10.1 目标和接口

SDSR 保持 detector-facing contract：

```text
input: 4 x DINOv3 hidden maps, native stride16
output: ordered subset of P3/P4/P5, each 256 channels
unchanged: decoder, positional encoding, masks, CDN, Group DETR
```

### 10.2 v1 四个核心组件

**Layer-channel routing**

每个 hidden layer 独立 LayerNorm，经共享 1x1 projection；P3/P4 分别使用 channel-wise
softmax router：

```text
S_s(c) = sum_l softmax(a_s(l,c)) * Proj(LN(H_l))(c)
```

**P3 semantic local reassembly**

P3 source 为四个 stride8 child positions 预测 3x3 sampling distribution，经 `unfold` 和
`pixel_shuffle` 重组。权重从 `align_corners=False` 双线性相位先验初始化。

**Directional detail stem**

轻量 stride8 CNN 提供水平、垂直、局部和 dilated image cues，只参与 weight prediction，
不把 raw CNN feature 拼入 detector feature。

**P5 anti-aliased phase downsample**

融合固定 binomial low-pass decimation 和每通道四个 2x2 sampling phases；高频响应预测
mixing gate，初始化偏向 low-pass。

P4 只用 routed semantic mixture 和小型 depthwise residual refine，作为语义锚点。

### 10.3 参数量和第一次消融

Projector-only 参数：

| Projector | Parameters |
|---|---:|
| MultiScaleProjector | 10,629,888 |
| SDSR-v1 detail32 | 556,752 |

v1 仅为原 Projector 约 5.2%。但统一 Medium 首轮结果为：

| 版本 | 组件 | AP |
|---|---|---:|
| MSP baseline | 原始 MultiScaleProjector | **34.0308** |
| S0 | 轻量 routed pyramid | 30.8808 |
| S1 | + semantic P3 reassembly | 30.6227 |
| S2 | + directional guide | **31.3523** |
| S3 | + phase downsample | 30.7520 |

这次失败非常关键：SDSR 的尺度解耦思想本身没有被否定，但“同时删除 C2f、多层独立
处理和高容量跨通道融合”过于激进。S2 比其他 v1 变体好，说明 image-guided local cue
有信息，但不足以弥补语义融合容量损失。

## 11. SDSR-v2-v9：寻找掉点根因

这一阶段大量版本只跑 3-6 epoch，用于筛选梯度、收敛速度和早期 AP，不应作为最终 AP
写入论文表。它们的价值是定位 v1 的失败原因。

### 11.1 v2：恢复 layer identity

v2 引入 layer-preserving `LightweightScaleFusion`，避免先把四层压成单一 routed source。

| 配置 | 状态 | AP |
|---|---|---:|
| v2 current baseline | 完整运行 | 33.7428 |
| v2 core r64 | 完整运行 | 31.9491 |
| v2 directional r64 | 完整运行 | **32.4241** |
| v2 directional r96 | 完整运行 | 31.9215 |

结论：恢复 layer identity 明显修复 v1，但轻量 fusion 仍不够；增加 rank 不等于更好。

### 11.2 v3-v6：空间条件和联合重组

- v3：target-scale layer fusion、rank downsample、spatial-confidence detail refinement；
- v4：每层先做 local reassembly，再在目标尺度融合；
- v5：保留 layer channels 的 joint spatial reassembly；
- v6：加入 shallow-token phase detail path，并比较 anti-alias on/off。

早期 screen：

| 版本 | 配置 | 轮数 | Screen AP |
|---|---|---:|---:|
| v3 | core | 6e | 14.6249 |
| v3 | spatial | 3e | 9.2530 |
| v4 | guided | 3e | 9.6490 |
| v4 | semantic | 4e | 11.6524 |
| v5 | guided | 6e | 15.0876 |
| v5 | semantic | 3e | 9.9474 |
| v6 | anti-alias | 4e | 10.9306 |
| v6 | no anti-alias | 4e | **11.2714** |

结论：空间自适应更复杂后，优化变难；过早强化低通会压制有用 detail。局部重组需要建立
在可靠的语义主干上，不能替代主干。

### 11.3 v7-v9：scale-first 与 CSP mixing

- v7：先构造尺度，再做 full-channel P3 phase up/downsample，可选 bidirectional scale
  calibration；
- v8：引入 CSP semantic mixing 和 inverted depthwise bottleneck；
- v9：P4 anchor preserving、P3 layer-phase residual、P5 semantic residual。

早期 screen：

| 版本 | 配置 | 轮数 | Screen AP |
|---|---|---:|---:|
| v7 | b0 | 6e | **16.7887** |
| v8 | CSP r64 | 6e | 15.3026 |
| v8 | CSP r96 | 6e | 15.6357 |
| v9 | anchor | 6e | 15.0663 |

v7 的恢复说明“先对齐尺度、再融合”比“先压缩层、再重组”更可靠，但仍未达到 MSP。

## 12. SDSR-v10-v23：恢复高容量融合并达到 36 AP

### 12.1 v10-v13：ExactScaleFusion 与 P5 basis

v10 引入 `ExactScaleFusion`：每个 hidden state 独立对齐目标尺度，拼接后使用 C2f 融合。
这是 SDSR 从“彻底替换 C2f”转向“保留已验证融合骨架、只创新重采样和层贡献”的关键转折。

- v10：P3/P4 使用 ExactScaleFusion，P5 alias-aware；
- v11：P5 使用 `TwoBasisDownsample`；
- v12：扩展为 FourBasis；
- v13：高通初始化的 two-basis。

v10/v12/v13 主要是短 screen。v11 首次稳定达到强基线区间：

| v11 运行 | AP |
|---|---:|
| seed42, lr_drop=100, 23 rows | 34.1341 |
| seed43, lr_drop=100, e28 | 35.6191 |
| seed42, lr_drop=22, e28 | **35.6432** |

结论：C2f 和独立尺度对齐恢复了多层语义融合；P5 two-basis 比更复杂 basis 更稳定。

### 12.2 v14-v22：围绕 P5 和轻量残差搜索

- v14：layer-adaptive P5，一条 dense sampler + 三条 two-basis；
- v15：zero-init bottom-up cross-scale；
- v16：zero-init P4-to-P5 pooling residual；
- v17：加快 P4-to-P5 gate 梯度；
- v18：learnable binomial low-pass bases；
- v19：额外低分辨率 P5 fusion block；
- v20：per-layer low-rank semantic adapters；
- v21：low-rank spatial-semantic residual；
- v22：expandable grouped detail sampling。

v14/v15/v16 的 6e screen 约为 `19.42/19.39/19.23`，v18-v22 多为短 screen。
这批实验说明：

1. 简单跨尺度残差没有自动收益；
2. P5 需要更有针对性的深层语义细节，而不是不断堆通用 adapter；
3. zero-init 保证初始稳定，但也可能在短训练预算内学不起来；
4. 下一步应扩展“最深层 grouped detail”，而非全路径扩容。

### 12.3 v23：成熟 SDSR 骨架

v23 结构：

```text
P3: ExactScaleFusion(scale=2) + C2f
P4: ExactScaleFusion(scale=1) + C2f
P5: TwoBasisDownsample on all layers
    + expanded grouped 3x3 detail only on deepest feature
    + C2f fusion
optional: BidirectionalScaleCalibration
```

关键思想是非对称容量分配：P3/P4 完整保留高容量融合；P5 的普通路径轻量，仅给最深层
增加跨通道 grouped detail。

| v23 运行 | AP | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| seed42, 24e, lr_drop20 | 35.6114 | - | - | - | - | - |
| seed42, low-LR finish | **36.0590** | 53.2316 | 37.9790 | 17.7589 | 38.6705 | **55.9784** |
| seed43, e28 | 35.7784 | 52.630 | 37.840 | 18.065 | 38.806 | 54.535 |

v23 首次达到预定 `36 AP` 目标，并比无 scale-routing MSP low-LR reference 的
`35.898` 高 0.161 AP；但仍比另行调优的 routed MSP `36.142` 低 0.083 AP。两种 MSP
对照不能混用。

## 13. SDSR-v24-v36：细节路径和执行效率

### 13.1 v24-v33 的结构探索

- v24：最早层 grouped detail；
- v25：dense early-layer detail；
- v26：endpoint-adaptive grouped detail；
- v27：更宽最深层 groups，8 groups / 48 channels；
- v28：更窄最深层 groups，32 groups / 12 channels；
- v29：deep semantic basis residual；
- v30：保守 image-detail residual at P3；
- v31：grouped-detail + semantic residual P5；
- v32：deepest-layer phase-calibrated P5；
- v33：residual-parameterized off-diagonal grouped detail。

有代表性的完整结果：

| 版本 | 协议 | AP |
|---|---|---:|
| v29 | e28 | 35.5906 |
| v31 | e28 | **35.8457** |

这些版本没有稳定超过 v23 low-LR finish。结论是 v23 最深层 grouped detail 已经接近
合理容量，继续给早层或跨层增加 detail 往往增加噪声或优化难度。

### 13.2 v34-v36 的打包与等价转换

- v34：packed separable two-basis P5；
- v35：single fused-kernel two-basis；
- v36：打包重复 low-pass、shallow depthwise 和 normalization，同时保留 v23 完整
  deepest grouped 3x3 detail。

seed43 native training：

| Model | AP | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| v23 | 35.778 | 52.630 | 37.840 | 18.065 | 38.806 | 54.535 |
| v34 | 35.635 | 52.710 | 37.830 | 17.443 | 38.331 | 52.776 |
| v36 | 35.501 | 52.297 | 37.815 | 17.150 | 38.390 | 54.325 |

native v36 因浮点累加顺序改变，不保证训练轨迹等价。但将 v23 checkpoint 转换为 v36：

- strict load 成功；
- 最大 projector 输出差 `2.86102294921875e-06`；
- 独立评测复现 `36.059 AP`；
- converter 删除 optimizer/scheduler state，因此只用于评测、推理或重新加载权重，不用于
  直接 optimizer resume。

### 13.3 v36 效率结果

640x640、RTX 5090、统一 eager benchmark：

| Model | Parameters | GFLOPs | FP32 mean | AMP mean |
|---|---:|---:|---:|---:|
| MSP | 40.574M | 90.564 | 23.415 ms | 29.330 ms |
| v23 | 35.757M | 88.644 | 24.354 ms | 29.682 ms |
| v34 | 35.686M | 88.618 | 23.450 ms | 30.098 ms |
| v36 | **35.757M** | **88.646** | **22.229 ms** | 29.871 ms |

相对 MSP，v36 总参数少 11.9%，GFLOPs 少 2.1%，FP32 mean 快 5.1%；AMP mean 慢
1.8%，median 快 0.9%。Projector 单模块 batch1 仍比 MSP 慢，因此必须同时报告模块和
端到端指标。

推荐工作流：训练 v23，部署/评测时转换 v36；不要把 native v36 的较低 AP 当作转换失败。

## 14. 另一会话改动对 SDSR 迭代的影响

### 14.1 实际发生的影响

在 SDSR 长期迭代期间，另一会话在共享工作树中实现或完善了：

- iterative box refinement；
- scale routing；
- MultiScaleProjector P5 轻量化；
- 后续分割/INSID3 文件。

这确实影响了可用代码环境和部分实验 recipe。特别是 v23 之后的强基线普遍与 iterative
box refinement、scale routing 一起训练，因此不能把完整 AP 增益全部归因于 Projector。

### 14.2 没有发生的事情

SDSR 并不是简单复制另一会话的 P5 轻量化：

- SDSR-v1 在那些后续策略之前已经提出尺度解耦、P3 local reassembly 和 P5 phase-aware
  downsample；
- v10-v23 的 TwoBasis、deep grouped detail 和 ExactScaleFusion 是 SDSR 自身迭代；
- iterative refinement/scale routing 属于 decoder/attention recipe，不是 SDSR 内部；
- v40 的 P4 depth prior 属于 SDSR 层选择机制。

### 14.3 如何保证后期可信度

1. 使用 seed 和 detector initialization seed 双固定；
2. 每一阶段建立 source snapshot；
3. 相邻结构在同一 snapshot/runner 协议比较；
4. 文档中分别报告 strict projector AP 和加入 Dense/Group/CDN 后的 integrated AP；
5. 不把 INSID3 等另一会话文件算作本会话贡献，也不在清理时误删。

## 15. SDSR-v37-v46：在 v23 上继续寻找结构增益

统一主要协议为 COCO Medium、24e、`lr_drop=20`、total batch16。完整结果：

| 版本 | Seed | AP | AP50 | AP75 | 结论 |
|---|---:|---:|---:|---:|---|
| v37 R1 routed C2f | 42 | 35.4644 | 52.7464 | 37.4281 | 路由残差可用但不优 |
| v37 R2 routed C2f | 42 | 35.5850 | 52.5864 | 38.0183 | v37 最好 |
| v37 R3 routed C2f | 42 | 35.1996 | 52.0696 | 37.0467 | 路由过强 |
| v38 P3 local residual | 42 | 35.4078 | 52.5533 | 37.8640 | P3 残差无明显收益 |
| v38 P4 dynamic | 42 | 35.4638 | 52.5369 | 37.6466 | 动态路由有限 |
| v38 P4 dynamic | 43 | 35.4999 | 52.5917 | 37.3605 | 跨 seed 未突破 |
| v39 P4 static | 43 | 35.3803 | 52.3689 | 37.5331 | 固定路由不足 |
| v40 P4 fixed prior | 43 | 35.7905 | 52.8548 | 38.1473 | 强先验有效 |
| v40 P4 learnable | 42 | 35.8411 | 52.8678 | 38.3695 | 强候选 |
| v40 P4 learnable | 43 | **35.9690** | **53.1171** | **38.3228** | strict 最佳 |
| v40 bounded dynamic | 43 | 35.6071 | 52.4729 | 37.7550 | 动态自由度无收益 |
| v41 P3 uniform | 43 | 35.6444 | 52.7959 | 38.1525 | 不超 v40 |
| v41 P3 shallow | 43 | 35.5694 | 52.3204 | 38.0218 | 浅层偏置不稳 |
| v42 P5 uniform | 43 | 35.8212 | 52.7448 | 38.2189 | 接近 v40 |
| v42 P5 deep | 43 | 35.5558 | 52.7786 | 37.8092 | 过度深层偏置 |
| v43 spatial 0.10 | 43 | 35.5544 | 52.8152 | 37.8849 | 空间残差无收益 |
| v43 spatial 0.20 | 43 | 35.8568 | 52.8093 | 38.2264 | APL 较好但不超 v40 |
| v44 C3k2 | 43 | 35.4063 | 52.5067 | 37.8094 | 替换 C2f 后退化 |
| v44 RepC3 | 43 | 34.5526 | 51.5115 | 36.5184 | 明显退化 |
| v45 centered spatial | 43 | 35.7549 | 52.7350 | 38.3068 | 更稳但不超 v40 |
| v46 annealed centered | 43 | 35.6206 | 52.4764 | 37.9500 | 退火无额外收益 |

### 15.1 v37-v46 的具体逻辑

- v37：验证“P3/P4 新方法 + 保留 C2f”的折中，采用 routed C2f 三种强度；
- v38：分别向 P3 加 local residual、向 P4 加 image-conditioned dynamic routing；
- v39：把 P4 动态路由简化为静态层权重；
- v40：从已训练模型层贡献中提取强深度先验，测试 fixed、learnable、bounded dynamic；
- v41：把深度先验移到 P3，验证高分辨率是否偏浅层；
- v42：把深度先验移到 P5，验证低分辨率是否应更偏深层；
- v43：在尺度路由中加入空间条件残差；
- v44：用 C3k2/RepC3 替代 C2f，测试速度和表达；
- v45：中心化空间残差，避免改变全局 gate 均值；
- v46：对中心化残差做训练期 annealing。

结果支持：

1. P4 全局可学习深度先验比逐图动态路由可靠；
2. P3/P5 同时复杂化没有必要；
3. C2f 在当前 small model 和 24e budget 下仍优于 C3k2/RepC3；
4. 复杂路由增加自由度，但没有自动转化为 AP；
5. v40 learnable 是最适合继续整合 matching/data recipe 的结构。

## 16. 当前 SDSR-v40 的完整结构

`ScaleDecoupledReassemblyProjectorV40` 继承 v23：

```text
4 DINOv3 hidden states
  -> P3 ExactScaleFusion(scale=2) + C2f
  -> P4 ExactScaleFusion(scale=1) + C2f
       with learnable depth prior
  -> P5 TwoBasisDownsample per layer
       + deepest expanded grouped detail
       + C2f
  -> optional cross-scale calibration (current: none)
```

P4 gate 初始化：

```text
(0.2485995, 0.4129705, 0.7943345, 2.5440950)
```

内部保存为 logits，经 softmax 后乘 4，使平均 gate 为 1。最终使用 learnable 版，允许
detector loss 在强深层先验附近继续调整。

当前选择：

```text
projector_type=sdsr_v40_p4_learnable
projector_scale=P3 P4 P5
sdsr_rank_channels=64
sdsr_detail_channels=32
sdsr_use_local_reassembly=False
sdsr_use_directional_guide=False
sdsr_use_phase_downsample=True
sdsr_cross_scale_mode=none
sdsr_cross_scale_rank=32
projector_p5_mode=full
```

这解释了 v1 与 v40 的关系：

- v1 是概念最纯的完整替换，轻但掉点；
- v40 保留尺度解耦思想，但承认 C2f 和多层独立尺度对齐的重要性；
- v40 的创新集中在非对称尺度处理、P5 two-basis/detail 和 P4 depth prior，不再以
  “完全删除原模块内部一切组件”为目标。

## 17. v40 与 Dense O2O、Group、CDN 的联合搜索

### 17.1 Dense O2O

Dense O2O 不是新的 matcher。它通过 Mosaic/MixUp/CopyBlend 增加单图 GT 密度，再进入
普通 Group Hungarian matching 和 CDN。

v40 seed43：

| 配置 | AP | AP50 | AP75 | Peak memory |
|---|---:|---:|---:|---:|
| strict, no Dense | 35.9690 | 53.1171 | 38.3228 | 13,432 MiB |
| Mosaic/MixUp/CopyBlend=.5/.5/.5 | 36.1239 | 52.7063 | 38.6949 | 22,633 MiB |
| no-MixUp=.5/0/.5 | **36.6764** | **53.8563** | **39.3384** | 15,063 MiB |
| CopyBlend-only=0/0/.5 | 36.4538 | 53.3681 | 39.1693 | 13,344 MiB |

选定调度：

```text
mode=enhanced
start_epoch=2
image_stop_epoch=12
copyblend_stop_epoch=21
mosaic_prob=0.5
mixup_prob=0.0
copyblend_prob=0.5
copyblend_num_objects=1
copyblend_area_threshold=100
copyblend_expand_ratios=0.1 0.25
```

删除 MixUp 同时提高 AP 并降低约 7.6 GiB 峰值显存，是明确的正向结论。

### 17.2 Group

固定 Dense no-MixUp 与 CDN50/loss0.5：

| Group | AP | AP50 | AP75 | Peak memory | Time |
|---:|---:|---:|---:|---:|---:|
| 6 | **36.6764** | 53.8563 | 39.3384 | 15,063 MiB | 1:59:48 |
| 4 | 36.5252 | 53.6416 | 38.9015 | 11,954 MiB | 1:53:10 |
| 2 | 36.1016 | 53.3916 | 38.3982 | 10,248 MiB | 1:51:18 |

Group6 是精度默认，Group4 是显存备选。

### 17.3 CDN

固定 Dense no-MixUp 与 Group6：

| CDN | AP | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| disabled | 36.1761 | 53.3948 | 38.7153 | 18.2973 | 39.2829 | 54.5929 |
| DN25, loss0.5 | **36.9718** | **53.8691** | **39.7993** | 18.2784 | 40.4331 | 54.7916 |
| DN50, loss0.3 | 36.7024 | 53.8330 | 39.2941 | 17.9703 | **40.4426** | 54.0024 |
| DN50, loss0.5 | 36.6764 | 53.8563 | 39.3384 | 17.9813 | 40.0817 | 53.0949 |

这里的表来自历史 Reg1 搜索。DN25 比 DN50 更适合 v40 integrated recipe，但仍应在
Reg0 下做邻近复测后再宣称“严格全局最优”。

### 17.4 EMA 和 total batch

EMA `decay=0.993, tau=100`：

| Branch | AP |
|---|---:|
| EMA | 36.6606 |
| same-run regular | 36.6445 |
| historical no-EMA | **36.9718** |

EMA 不采用。total batch32 的试验降至 `33.8250 AP`，说明增大 total batch 不是免费加速，
也会改变 optimization noise、学习率和有效更新次数。

## 18. Register border 污染和干净结果

历史 matching 搜索意外使用：

```text
backbone_register_border_tokens=1
```

它在 token grid 外围添加一圈 register/noise patch，参与 DINOv3 self-attention，进入
Projector 前裁掉。因为最终路线没有采用 inference-time register，这个设置必须关闭。

仅将 1 改为 0 的严格复跑：

| Metric | Reg1 historical | Reg0 clean | Delta |
|---|---:|---:|---:|
| AP | 36.9718 | **36.4706** | -0.5012 |
| AP50 | 53.8691 | 53.6406 | -0.2285 |
| AP75 | 39.7993 | 38.9451 | -0.8542 |
| APS | 18.2784 | **19.0819** | +0.8034 |
| APM | 40.4331 | 39.7308 | -0.7023 |
| APL | 54.7916 | 53.6951 | -1.0965 |

可信表述：

- `36.4706 AP` 是当前纯 SDSR-v40 integrated medium baseline；
- `36.9718 AP` 是包含 inference-time register border 的 ablation；
- register 主要提高 AP75/APM/APL，却降低 APS；
- Reg1 搜出的 Group/CDN 组合在 Reg0 下是强候选，不等同于已重新穷举的最优组合。

## 19. 当前有效 Full COCO 运行

运行名：

```text
coco_full_s43_detinit1043_sdsr_v40_dense_nomix_g6_cdn25_l05_reg0_lrd20_e24
```

日志和输出：

```text
coco_full_s43_detinit1043_sdsr_v40_dense_nomix_g6_cdn25_l05_reg0_lrd20_e24_gpu1.log
output/coco_full_s43_detinit1043_sdsr_v40_dense_nomix_g6_cdn25_l05_reg0_lrd20_e24/
run_sdsr_v40_best_full_gpu1.sh
```

本文最终核验时，日志已进入 `Epoch [2/24]`，约 `330/7392` iteration；第 1 轮训练与
验证记录已经写入输出目录，日志中未发现 `Traceback`、OOM 或进程被杀信息。PID 文件
中的后台 launcher PID 可能在 shell 退出后失效，因此判断运行状态应以日志持续增长、
GPU process 和输出目录的 `log.txt` 为准，不能只依赖旧 PID 文件。

完整关键参数：

```text
dataset=COCO full, train/val=118287/5000
epochs=24
batch_size=8
grad_accum_steps=2
world_size=1
total_batch_size=16
resolution=640
multi_scale=True
expanded_scales=True
lr=1e-4
lr_encoder=1.5e-4
lr_drop=20
weight_decay=1e-4
lr_vit_layer_decay=0.8
lr_component_decay=0.7
seed=43
detector_init_seed=1043

projector=sdsr_v40_p4_learnable
projector_scale=P3 P4 P5
out_feature_indexes=2 5 8 11
dec_layers=4
num_queries=300
num_select=300
group_detr=6
dec_n_points=2
bbox_refine_mode=shared
iterative_bbox_refinement=True
scale_routing=True
scale_routing_mode=legacy

use_cdn=True
dn_number=25
dn_label_noise_scale=0.5
dn_box_noise_scale=0.6
dn_loss_coef=0.5

use_dense_o2o=True
dense_o2o_mode=enhanced
dense_start=2
dense_image_stop=12
copyblend_stop=21
mosaic/mixup/copyblend=0.5/0/0.5
copyblend_num_objects=1

backbone_register_border_tokens=0
use_ema=False
online_refine_mode=none
feature_adapter=none
```

此前 Reg1 full 完成约 3 epoch 后已停止，不能与当前 Reg0 run 拼接或续训。

## 20. 关键代码与产物索引

### 20.1 Decoder 和训练策略

```text
src/rfdetr/models/dn_components.py
src/rfdetr/models/lwdetr.py
src/rfdetr/models/transformer.py
src/rfdetr/datasets/dense_o2o.py
src/rfdetr/engine.py
src/rfdetr/config.py
src/rfdetr/main.py
run_coco_subset.py
```

### 20.2 Backbone refinement 和诊断

```text
tools/train_dinov3_lazystrike_refine.py
tools/refinement/unirefiner_objective.py
tools/analyze_dinov3_tokens.py
tools/linear_seg_ade20k.py
```

### 20.3 SDSR 实现

```text
src/rfdetr/models/backbone/semantic_reassembly_projector.py
src/rfdetr/models/backbone/semantic_reassembly_projector_v2.py
...
src/rfdetr/models/backbone/semantic_reassembly_projector_v46.py
src/rfdetr/models/backbone/backbone.py
tests/test_semantic_reassembly_projector.py
tools/convert_sdsr_v23_to_v36.py
```

### 20.4 关键 checkpoint/结果

```text
# v23 -> v36 converted 36.059 AP
output/coco_medium_v36_converted_from_v23_ap36p059/checkpoint_best_regular.pth

# v40 strict seed43
experiment_snapshots/sdsr_v40_strong_prior_medium_seed43

# v40 integrated historical matching search
experiment_snapshots/sdsr_v46_denseo2o_medium_seed43

# Reg0 clean medium
output/coco_medium_s43_detinit1043_sdsr_v40_dense_nomix_g6_cdn25_l05_reg0_lrd20_e24/

# Current Reg0 full
output/coco_full_s43_detinit1043_sdsr_v40_dense_nomix_g6_cdn25_l05_reg0_lrd20_e24/
```

## 21. 已验证、已否定和仍待验证的结论

### 21.1 已验证

1. CDN 与 Group DETR 可以正确兼容，且 CDN 在早期 Full COCO 有约 +0.4 AP。
2. Group6 比 Group13 更适合当前训练预算。
3. DINOv3 多层 feature fusion 非常重要；v1 的纯轻量替换明显失败。
4. P3/P4 的 ExactScaleFusion + C2f 是恢复 SDSR 性能的关键。
5. P5 two-basis + deepest grouped detail 是有效的轻量非对称结构。
6. v23 可达到 36.059 AP；转换为 v36 可以保持前向和 AP。
7. v40 的 P4 learnable depth prior 优于相邻动态/静态路由。
8. C3k2/RepC3 在当前协议下不能替代 C2f。
9. Dense no-MixUp 比 full Dense 更准且省显存。
10. EMA 和 total batch32 不适合当前 medium recipe。
11. register border 对 AP 有明显影响，不能遗留为隐式默认值。

### 21.2 已否定或不应重复

1. 不再做“删除全部 C2f，只靠极轻路由和重采样”作为主线。
2. 不把 2-6e screen AP 当最终结果。
3. 不认为参数更少必然更快；GPU kernel 和 memory layout 需要实测。
4. 不把 token heatmap、abnormal ratio 或 CLS coverage 当成 AP surrogate。
5. 不假设 UniRefiner、LazyStrike、dense consistency 的收益可直接相加。
6. 不把 COCO-refined backbone 的 detection 增益外推为分割收益。
7. 不在 detector 训练完成后直接 post-hoc refine backbone 后无适配替换回去。
8. 不默认开启 EMA、register border 或大 total batch。
9. 不把另一会话的 iterative refinement/scale routing 收益全部算给 SDSR。
10. 不把 Reg1 的 36.9718 写成纯 SDSR 最终结果。

### 21.3 仍待验证

1. 当前 Reg0 Full COCO 的最终 AP 和尺度分解；
2. Reg0 seed42 medium，评估 `36.4706` 的方差；
3. Reg0 下邻近 CDN25/CDN50/no-CDN 与 Group4/6 是否保持排序；
4. SDSR-v40 的标准 TensorRT/ONNX latency，而非只看 eager PyTorch；
5. 在更大 DINOv3 backbone 上 UniRefiner 是否比 small 更有价值；
6. 未来接入实例/语义分割时，SDSR 的 P3 边界和 refinement 是否产生新收益。

## 22. 当前工作树和并行会话注意事项

写入本文时工作树是 dirty 状态。不要 reset、checkout 或删除不属于当前任务的文件。

以下文件主要来自另一会话的分割/INSID3 工作，不属于本会话 SDSR 贡献：

```text
src/rfdetr/models/insid3.py
src/rfdetr/models/insid3_coco.py
tests/test_insid3.py
tests/test_insid3_coco.py
tools/evaluate_insid3_coco_instances.py
SEGMENTATION_ARCHITECTURES_AND_INSID3_COCO.md
```

当前 v40-v46、runner 注册、测试、脚本和 snapshots 也有未提交内容。提交时应按主题选择性
stage，不要用破坏性命令清理整个工作树。

## 23. 给下一会话的推荐顺序

### P0：完成当前 Full

1. 确认 PID 和日志仍活跃，不要启动同名第二进程；
2. 每 epoch 检查训练和验证是否完整；
3. Full 结束后报告 best epoch、AP/AP50/AP75/APS/APM/APL、显存和总训练时间；
4. 只使用 Reg0 run，不与已停止的 Reg1 checkpoint 混合。

### P1：补足论文可信度

1. Reg0 seed42 medium；
2. Reg0 下最小邻域复测：no-CDN、DN25、DN50，Group4/6；
3. 汇总两 seed 均值/标准差；
4. 明确拆分 `Projector-only`、`+ decoder recipe`、`+ Dense/CDN` 的增益。

### P2：整理代码

1. 提交 SDSR-v40-v46 和 Projector tests；
2. 单独提交 matching runner/scripts；
3. 保留 snapshots 和 `args.json`，日志可压缩但不要只留截图；
4. 不触碰另一会话的 INSID3 改动。

### P3：下一研究方向

1. 若继续 Projector，优先研究 P3 faithful boundary reconstruction，而不是继续堆 P4
   dynamic routing；
2. 若继续效率，围绕 v23->v36 等价打包和导出 kernel，而不是牺牲 C2f 容量；
3. 若继续 backbone，优先验证更大 DINOv3 或 detection-aware dense objective；
4. 若接分割，重新建立任务内 baseline，不从 detection AP 直接推断。

## 24. 最终可引用的阶段结论

SDSR 的完整故事不是“完全替换 MultiScaleProjector 后一次成功”，而是：

> 最初的 SDSR-v1 通过 layer-channel routing、P3 local reassembly 和 P5 anti-aliased
> phase downsampling，把 Projector 参数降到原来的约 5.2%，但因删除了 C2f 和高容量
> 多层融合而损失约 2.7-3.4 AP。随后 v2-v23 的实验逐步证明 layer identity、目标尺度
> 对齐和 C2f 对 DINOv3 检测至关重要，并形成 P3/P4 ExactScaleFusion、P5 two-basis
> 与 deepest grouped detail 的成熟骨架。v23 达到 36.059 AP，v36 可等价转换并减少
> 11.9% 总参数。后续 v37-v46 进一步确认 P4 可学习深度先验比复杂动态路由和替换 C2f
> 更有效，最终选定 SDSR-v40。关闭遗留 register border 后，SDSR-v40 与 Dense
> no-MixUp、Group6、CDN25 的可信 Medium 结果为 36.4706 AP；当前正以完全相同的
> Reg0 原则进行 Full COCO 验证。

这也是本会话与其他 handoff 最需要区分的贡献：不仅记录最终 v40，还保留了 SDSR 从
极轻完整替换失败，到以尺度职责解耦为核心、恢复必要融合容量并达到可用精度/效率平衡的
完整演化过程。
