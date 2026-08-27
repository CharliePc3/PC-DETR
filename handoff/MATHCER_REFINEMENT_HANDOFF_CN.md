# RF-DETR-DINOv3 会话工作与贡献交接文档

> 文档日期：2026-08-25  
> 项目目录：`/data/cpc/root/project/RF-DETR-DINOv3`  
> 本会话主要工作区间：CDN、Group、SA-Matching、Dense O2O Decoder 训练优化  
> 本会话核心提交：`38e05ce`、`ce7c923`、`260f259`、`7cbdaaf`  
> 当前仓库分支：`exp/scale-decoupled-projector`

## 1. 文档用途

本文档把本会话完成的研究、代码实现、实验结论和工程经验交接给后续会话，重点回答：

1. RF-DETR-DINOv3、RF-DETR/LW-DETR 和 DEIMv2 的结构关系是什么；
2. Decoder 侧已经实现并验证了哪些训练优化；
3. CDN、Group、SA-Matching、Dense O2O 分别作用在哪里；
4. 哪些配置已被验证，哪些只完成单 seed，哪些方向已经否定；
5. 关键代码、日志、结果和 Git 提交在哪里；
6. 后续最值得继续的实验是什么；
7. 当前仓库状态有哪些风险，接手时应避免什么。

本文是本会话 Decoder 优化阶段的历史快照。当前仓库后来已经进入 projector、backbone
refinement、segmentation 等后续主线，Medium AP 也已经超过本文时期的 33--35 AP。
因此本文中的增益不能直接与后续新架构结果相加；任何旧策略迁移到当前最强模型时，都必须
重新建立 contemporaneous baseline。

## 2. 十分钟结论

### 2.1 已完成的主要贡献

1. 在 RF-DETR-DINOv3 中完整接入了 DINO 风格 CDN：正/负去噪 Query、隔离 Attention
   Mask、逐层 DN Loss、Group-aware Query 布局和推理剥离。
2. 让 CDN 与 RF-DETR 原有 Group DETR 策略正确兼容，并通过 Medium 和 Full COCO
   证明 CDN 有效。
3. 将 Group 数量从 RF-DETR 默认的 13 系统消融到 8/7/6/4，确定 Group 6 是当时最优
   精度/成本点。
4. 实现 Budgeted SA-Matching，并扩展到 Group 1/2/4/6；完成双 seed、预算和 Group
   替代实验，确认当前 SA 设计不值得继续投入。
5. 实现 DEIM/DEIMv2 风格 Dense O2O 数据管线，包括 Mosaic、MixUp、增强版
   CopyBlend 和分阶段调度。
6. 完成 D1/D2、CopyBlend 数量/概率/上下文/停止轮次及组合消融，确认 Dense O2O
   有效，`CopyBlend N=1` 是本阶段单 seed 最强候选。
7. 建立可续跑、独立输出目录、PID/锁/GPU 等待、结果自动汇总的实验队列。
8. 定位 Dense O2O 显存接近翻倍的根因：高密度 GT 放大 CDN `pad_size`，再被
   Group 6 复制，而不是 GT tensor 本身占用大量显存。

### 2.2 最终决策

| 方向 | 结论 |
|---|---|
| Group DETR | 保留 Group 6；Group 13 不再是主线 |
| CDN | 保留，参数为 DN 50 / label 0.5 / box 0.6 / loss 0.5 |
| Budgeted SA | 当前实现否定，暂停预算和窗口搜索 |
| Dense O2O D2 | 双 seed 有效，保留 |
| CopyBlend N=1 | 当时最高 AP 候选，但本会话结束时只有 seed 42 |
| N=1 + tight context | 无叠加收益，否定 |
| 过早关闭 CopyBlend | 性能下降，否定 |

### 2.3 本阶段推荐配置

```text
decoder_layers=4
num_queries=300
num_select=300
group_detr=6

use_cdn=true
dn_number=50
dn_label_noise_scale=0.5
dn_box_noise_scale=0.6
dn_loss_coef=0.5
dn_negative=true

use_dense_o2o=true
dense_o2o_mode=enhanced
dense_o2o_start_epoch=2
dense_o2o_image_stop_epoch=12
dense_o2o_copyblend_stop_epoch=21
dense_o2o_mosaic_prob=0.5
dense_o2o_mixup_prob=0.5
dense_o2o_copyblend_prob=0.5
dense_o2o_copyblend_num_objects=1
dense_o2o_copyblend_expand_ratios=(0.1, 0.25)
```

严格来说，双 seed 验证完成的是 D2 base（`N=3`）；`N=1` 在本会话结束时仍是单 seed
promotion candidate，不应写成已完成双 seed 的最终配置。

## 3. 项目与架构认知

## 3.1 模型继承关系

```text
LW-DETR
  -> RF-DETR：以 DINOv2 ViT 为 backbone，保留 Group DETR / DETR decoder
      -> RF-DETR-DINOv3：将预训练 ViT 替换为 DINOv3，并继续改造检测接口
```

RF-DETR-DINOv3 的核心仍是：

```text
image
  -> DINOv3 ViT hidden states
  -> multiscale projector（生成 P3/P4/P5）
  -> deformable transformer encoder/decoder
  -> classification + box heads
```

RF-DETR 的训练期 Group 策略把 Query 分成多个独立组，各组进行严格 O2O Hungarian
matching；推理只保留第一组 Query。因此 Group 增加训练正匹配和收敛速度，但也增加训练
计算与显存，不改变推理 Query 数。

## 3.2 RF-DETR 与 DEIMv2 的多尺度思路

本会话早期对两个 DINOv3 检测器的 backbone-to-decoder 接口进行了对比：

- RF-DETR 路线从 ViT 多个隐藏层提取特征，再由 `MultiScaleProjector` 做通道映射和
  上/下采样，生成多尺度特征。
- DEIMv2-DINOv3 路线在多个 ViT 层之间构造/Resize 多尺度特征，并额外使用极轻量 CNN
  捕获细粒度局部信息，再与 ViT 特征融合。
- 两者都在解决 ViT 默认单一 patch stride（通常 1/16）与检测 Decoder 多尺度输入之间的
  不匹配，但侧重点不同：RF-DETR 更贴近原有轻量 projector，DEIMv2 更主动补充局部细节。

这部分讨论后来促成了独立的 projector/refinement 主线，但 CDN/Dense O2O 不属于
MultiScaleProjector 改进。它们是 Decoder 训练和数据监督优化。

## 3.3 四种训练机制必须区分

| 机制 | 主要位置 | 是否直接改变 Hungarian matching | 核心作用 |
|---|---|---:|---|
| Group DETR | Decoder + Matcher | 是 | 多组独立 O2O，增加每个 GT 的训练正匹配 |
| CDN | Decoder 输入与 DN Loss | 否 | 已知对应关系的带噪 GT Query，稳定分类/回归 |
| Budgeted SA | Matcher / aux outputs | 是 | 按尺度给 GT 分配额外辅助正匹配 |
| Dense O2O | batch 数据增强 | 否 | 增加一张图中的 GT 密度，再执行普通严格 O2O |

特别注意：CDN Query 不参加 Hungarian matching；Dense O2O 也不是新的 matcher，它改变
图像和 targets，新增 GT 随后才进入现有 Group Hungarian matching 和 CDN 构造。

## 4. CDN：原理、实现与贡献

## 4.1 原理澄清

本会话结合 DN-DETR、DINO 和 Stable-DINO 讨论并明确：

- DN-DETR 构造带噪 GT Query，模型学习重建正确类别与框；它提供稳定、直接的优化路径。
- DINO 的 CDN 同时包含正样本和负样本：正样本使用较小噪声并重建 GT，负样本使用更大
  噪声并被识别为 no-object，从而形成对比式去噪训练。
- DINO Mixed Query Selection 使用 encoder top-K 的坐标初始化 anchor/reference boxes，
  content query 仍使用可学习嵌入。它与 Stable-DINO 都关注优化路径质量，但并不是同一种
  方法：Mixed Query Selection 是 Query 初始化，Stable-DINO 是匹配/分类监督稳定性。

本项目直接接入 CDN，而没有先实现一个临时 DN 版本。二者接口形态相同，CDN 是更完整的
训练方案，直接实现可以减少重复工程。

## 4.2 代码实现

关键文件：

- `src/rfdetr/models/dn_components.py`
- `src/rfdetr/models/lwdetr.py`
- `src/rfdetr/models/transformer.py`
- `src/rfdetr/config.py`
- `src/rfdetr/main.py`
- `run_coco_subset.py`

已实现能力：

1. 根据 batch targets 构造 positive/negative CDN Query；
2. 标签噪声和框噪声；
3. DN groups 之间隔离、DN 与普通 matching Query 隔离的 self-attention mask；
4. 与 Group DETR 的 Query 排布兼容；
5. Decoder 输出后把 DN Query 与正常 Query 分离；
6. 最终层和 auxiliary decoder layers 的分类、L1、GIoU DN losses；
7. 独立 DN loss coefficient 与负样本系数；
8. 推理时完全移除 CDN，不增加推理开销；
9. CLI/config/logging/checkpoint resume 支持。

## 4.3 CDN 参数消融

早期 12-epoch Medium 消融：

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

在后续 total batch 16、Group 6 协议中，`dn_number=100`、`dn_loss_coef=0.3/0.7`
也没有超过 `DN 50 + loss 0.5`。

## 4.4 Full COCO 验证

| 配置 | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| Group 13，无 CDN | 51.10 | 69.98 | 55.03 | 29.65 | 55.83 | 70.91 |
| Group 13 + CDN | 51.50 | 70.18 | 55.65 | 29.76 | 56.24 | 71.03 |
| Group 6 + CDN | **51.55** | 70.17 | **55.85** | **30.18** | 56.12 | **71.08** |

结论：CDN 在 Full COCO 上相对同 Group13 无 CDN 提升约 0.40 AP，属于已验证贡献。

## 5. Group DETR：兼容与消融

## 5.1 工程贡献

CDN 最初只在 Group 1/Medium 环境 smoke。随后完成：

- CDN Query 在每个 Group 内的正确布局；
- normal/DN Query reshape、拼接和输出拆分；
- 每组 loss 归一化；
- 训练使用多 Group、推理只使用第一组；
- unique output directory，避免多个消融覆盖 checkpoint/log；
- 分布式训练、后台运行和 checkpoint resume 的实际验证。

## 5.2 Group 数量消融

18 epochs、total batch 16、同一 CDN 配置：

| Group | AP |
|---:|---:|
| 13 | 32.70 |
| 8 | 32.49 |
| 7 | 32.55 |
| 6 | **32.99** |
| 4 | 32.29 |

24-epoch contemporaneous Group6 strict baseline：

- seed 42：33.948 AP
- seed 43：33.983 AP
- 两 seed 均值：33.965 AP

结论：Group 6 是本阶段准确率主线。Group 2/4 可以作为效率候选，但不能替代主配置。

## 6. Budgeted SA-Matching：实现与否定结论

## 6.1 实现

关键文件：

- `src/rfdetr/models/matcher.py`
- `src/rfdetr/models/lwdetr.py`
- `run_coco_subset.py`
- `tests/test_budgeted_sa_matcher.py`
- `tools/smoke_budgeted_sa.py`

实现内容：

1. 按 GT pixel area 分 small/medium/large；
2. 按总预算为 auxiliary decoder outputs 分配额外匹配；
3. Group > 1 时保持 inference group 严格 O2O，把额外匹配放在辅助 Group；
4. Group 1 时允许 auxiliary decoder outputs 使用未匹配 Query 增加 SA positives，同时
   final decoder output 保持严格 O2O；
5. 记录 requested/matched 和各尺度匹配数量；
6. 支持 start/stop epoch 调度和预算消融。

## 6.2 S1/S2/S3 与预算实验

最终确认中：

- strict 两 seed 平均：33.97 AP
- SA 6/7/9、S3 两 seed平均：33.87 AP
- SA 平均降低约 0.10 AP 和 0.40 AP50
- 6/7/8、6/8/10、7/8/9 均低于 strict seed42 baseline

## 6.3 SA 替代 Group

| 配置 | AP | 相对 Group6 strict |
|---|---:|---:|
| Group6 strict | **33.948** | 0.000 |
| Group4 + SA 4/7/9 | 33.565 | -0.382 |
| Group2 + SA 2/7/9 | 33.509 | -0.439 |
| Group1 + SA 1/7/9 | 32.247 | -1.701 |

匹配统计显示 requested matches 几乎 100% 被满足，因此失败不是 Query 容量不足。当前
预算对 small 的预算等于 Group 数，没有增加小目标匹配，却给 medium/large 注入大量额外
正样本。Group 越小，这种监督重分配越强，G1 受损最明显。

结论：当前 auxiliary-only SA 与 Group 的功能高度重叠，但没有形成稳定互补。暂停该路线，
除非提出新的尺度分配或 loss weighting 假设。

## 7. Dense O2O：实现、调度与实验

## 7.1 实现定位

关键文件：

- `src/rfdetr/datasets/dense_o2o.py`
- `src/rfdetr/engine.py`
- `src/rfdetr/main.py`
- `src/rfdetr/config.py`
- `run_coco_subset.py`
- `tools/smoke_dense_o2o.py`

支持两个模式：

- `image`：Mosaic + MixUp，对应基础 Dense O2O；
- `enhanced`：Mosaic + MixUp + CopyBlend。

CopyBlend 可控制：

- `copyblend_num_objects`：每张目标图最多复制几个目标；
- `copyblend_prob`：满足调度窗口时触发 CopyBlend 的概率；
- `copyblend_expand_ratios`：复制目标时保留多少周围背景；
- `copyblend_area_threshold`：进入复制池的最小目标面积；
- `copyblend_stop_epoch`：CopyBlend 停止轮次。

## 7.2 D2 调度

```text
epochs 0-1:   原始图像
epochs 2-11:  Mosaic + MixUp + CopyBlend
epochs 12-20: CopyBlend only
epochs 21-23: 原始图像，恢复严格数据分布
```

这一调度本身已经是一种 curriculum：先增大目标密度，再降低增强强度，最后回到推理时的
原始分布。`CopyBlend stop=12` 结果变差，证明较长的轻量 CopyBlend tail 有价值。

## 7.3 D2 双 seed 结果

| 配置 | 两 seed AP 均值 |
|---|---:|
| Group6 + CDN strict | 33.965 |
| D2 base | **34.314** |

D2 base 平均提升：

- AP：+0.349
- AP75：+0.432
- APm：+0.779
- APl：+0.303
- AP50：-0.168
- APs：-0.180

Dense O2O 的稳定收益更偏向精确定位和中大目标，而不是 AP50。

## 7.4 CopyBlend 消融

seed 42：

| 配置 | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| D2 base，N=3 | 34.290 | 51.031 | 36.290 | 16.250 | **37.506** | 54.409 |
| N=1 | **34.897** | **51.760** | 37.378 | 17.541 | 37.089 | 53.582 |
| p=0.25 | 34.589 | 51.251 | 36.963 | **18.499** | 37.334 | 54.153 |
| context 0.00--0.10 | 34.705 | 51.080 | **37.470** | 17.934 | 37.032 | **54.751** |
| stop=12 | 34.092 | 50.905 | 36.358 | 17.024 | 36.912 | 52.220 |
| N=1 + context 0.00--0.10 | 34.395 | 51.064 | 36.911 | 17.470 | 36.734 | 52.656 |

解释：

- N=1 比 N=3 更好，说明默认每次复制 3 个目标过强；频繁但温和的增强更合适。
- p=0.25 对 APs 最好，适合作为小目标候选，而非总体 AP 最优。
- tight context 单独改善 AP75/APl，说明减少背景污染有效。
- N=1 与 tight context 不叠加，两个降强度操作同时使用后监督过弱或多样性不足。
- stop=12 退化，说明后半程 CopyBlend 不应过早关闭。

## 8. Dense O2O 显存问题

同一 `batch=8、accum=2、group=6、CDN`：

| 配置 | 峰值显存 |
|---|---:|
| strict，无 Dense O2O | 12,643 MiB |
| D2 base | 24,840 MiB |
| D2 N=1 | 25,912 MiB |
| N=1 + tight context | 21,615 MiB |

### 8.1 根因

Dense O2O 前期把 local batch 的平均目标数从约 107 提高到约 412--419。当前 CDN：

```python
max_gt = max(len(target["labels"]) for target in targets)
group_size = max_gt * 2  # positive + negative
dn_groups = max(dn_number // group_size, 1)
pad_size = group_size * dn_groups
```

所以 `dn_number=50` 不是 hard cap。当单图 `max_gt > 25` 时，至少保留一个 DN group，
`pad_size=2*max_gt`，然后 Group 6 再把 CDN Query 复制六份。四层 Decoder 需要保存这些
Query 的 attention 和反向激活，峰值显存因此接近翻倍。

### 8.2 推荐修复

增加真正的 CDN GT/Query 上限，例如：

```text
dense_o2o_cdn_max_gt=25
```

当 GT 超过上限时，只为 CDN 分层或随机采样最多 25 个 GT；所有 GT 仍参与普通 Group
Hungarian matching。这样不削弱 Dense O2O 的检测监督，只限制去噪辅助路径的极端 Query
数量。不能只调低 `dn_number`，因为当前 `max(..., 1)` 仍可能产生 `2*max_gt` padding。

## 9. 实验与工程基础设施贡献

本会话不仅完成模型实现，也建立了较稳定的实验运行方式：

1. 每个实验使用唯一 `output_dir`，避免 checkpoint/log 相互覆盖；
2. 脚本在输出目录存在完整 24 epoch 记录时自动跳过；
3. 存在 checkpoint 时自动 resume；
4. 使用 PID file、lock file 和 GPU memory threshold，减少重复启动；
5. 队列失败立即停止，不静默污染后续实验；
6. 使用 `jq` 从 `log.txt` 自动抽取最佳 epoch、AP/AP50/AP75/APs/APm/APl；
7. 所有关键比较统一 total batch size 16；
8. 明确区分 batch size、gradient accumulation 和 world size，避免不同 effective batch
   的结果被误当成纯方法消融；
9. 完成分布式后台训练中断、残留显存进程定位和 checkpoint 续跑验证。

## 10. 关键代码与文档索引

### 10.1 代码

| 文件 | 作用 |
|---|---|
| `src/rfdetr/models/dn_components.py` | CDN Query、mask、post-process、DN losses |
| `src/rfdetr/models/lwdetr.py` | CDN 接入模型与 Criterion、SA/Dense 配置传递 |
| `src/rfdetr/models/transformer.py` | Group-aware normal/DN Query 拼接与 Decoder 输入 |
| `src/rfdetr/models/matcher.py` | Group O2O 和 Budgeted SA matching |
| `src/rfdetr/datasets/dense_o2o.py` | Mosaic、MixUp、CopyBlend 和调度 |
| `src/rfdetr/engine.py` | epoch/step 传入 Dense O2O、统计日志 |
| `run_coco_subset.py` | 实验 CLI、校验、输出命名与入口 |
| `tests/test_budgeted_sa_matcher.py` | Group1/Group2 SA 行为测试 |
| `tools/smoke_budgeted_sa.py` | SA smoke test |
| `tools/smoke_dense_o2o.py` | Dense O2O smoke test |

### 10.2 实验脚本

| 文件 | 作用 |
|---|---|
| `run_budgeted_sa_medium_queue.sh` | SA S1/S2/S3 初筛 |
| `run_sa_confirmation_budget_queue_gpu0.sh` | strict/SA 双 seed 与预算确认 |
| `run_sa_group_replacement_queue_gpu1.sh` | G1/G2/G4 strict 与 SA 替代实验 |
| `run_dense_o2o_independent_queue_gpu3.sh` | Dense O2O D1/D2 独立实验 |
| `run_dense_o2o_d2_ablation_queue_gpu0.sh` | D2 N/p/stop/context 消融 |
| `run_dense_o2o_d2_n1_ctx00010_gpu3.sh` | N=1 + tight context 组合实验 |

### 10.3 结果文档

| 文件 | 内容 |
|---|---|
| `EXPERIMENT_SUMMARY_DECODER_OPTIMIZATION.md` | Decoder 优化总表与最终决策 |
| `EXPERIMENT_LOG_BUDGETED_SA.md` | SA 窗口、预算、双 seed、替代 Group 结果 |
| `EXPERIMENT_PLAN_D2_SA_REPLACEMENT.md` | D2 与 SA 替代实验计划及结果 |
| `EXPERIMENT_LOG_DENSE_O2O_STAGE2.md` | Dense O2O Stage 2 设计 |

### 10.4 代表性日志和输出

```text
# strict Group6 + CDN, seed 42/43
coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_strict_current_20260719_seed42_e24_nohup.log
coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_strict_current_20260719_seed43_e24_nohup.log

# D2 N=1, seed 42
coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_d2_cb1_p05_stop21_ctx01025_seed42_e24_nohup.log
output/coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_d2_cb1_p05_stop21_ctx01025_seed42_e24/log.txt

# N=1 + tight context
coco_medium_tb16_bs8_g6_p3p4p5_ms_exp_cdn_d2_cb1_p05_stop21_ctx00010_seed42_e24_nohup.log

# Full COCO Group6 + CDN
full_coco_dinov3_e24_p3p4p5_ms_exp_group6_cdn_nohup_v2.log

# SA queues
sa_confirmation_budget_gpu0_queue_nohup.log
sa_group_replacement_gpu1_queue_nohup.log
```

## 11. Git 提交索引

| Commit | 贡献 |
|---|---|
| `38e05ce add CDN denoising training path` | 初版 CDN 完整训练路径 |
| `ce7c923 Add group-aware CDN denoising baseline` | CDN 与 Group DETR 正确兼容 |
| `a53162a Save backbone and dense detection experiments` | Dense O2O、Budgeted SA、日志与脚本主体 |
| `260f259 Evaluate Dense O2O and SA group replacement` | Group1 SA、D2 消融脚本、测试和结果 |
| `7cbdaaf Summarize decoder optimization experiments` | Decoder 优化统一总结 |

这些提交已经被合并进当前 `exp/scale-decoupled-projector` 分支历史。若需要复现本阶段
代码，优先使用 Git 提交而不是从当前脏工作区手动拷贝文件。

## 12. 测试与验证状态

本会话完成过：

- CDN smoke forward/backward；
- Group + CDN 多组训练 smoke；
- Dense O2O smoke；
- SA matcher 行为测试；
- 相关 Python `py_compile`；
- 实验脚本 `bash -n`；
- 多次完整 Medium 24-epoch；
- Group13/Group6 + CDN Full COCO；
- 后台训练中断与 resume。

当时 `rfdetr-dinov3` 环境未安装 `pytest`，所以
`tests/test_budgeted_sa_matcher.py` 的两个测试函数使用同一 Python 环境直接调用并通过。
后续环境如已安装 pytest，应重新执行：

```bash
PYTHONPATH=src /home/cpc/.conda/envs/rfdetr-dinov3/bin/python \
  -m pytest -q tests/test_budgeted_sa_matcher.py
```

## 13. 未完成任务与推荐顺序

## P0：接手前先保护当前工作区

当前分支为 `exp/scale-decoupled-projector`，工作区包含大量后续 projector、segmentation、
benchmark 和实验文档改动，其中有已修改文件、删除文件和未跟踪文件。它们不是本交接文档
创建时产生的，也不能被 reset/checkout/clean。

开始任何修改前必须：

```bash
git status --short --branch
git log --oneline --decorate -12
```

只暂存自己修改的文件，禁止 `git add .`。

## P1：确认 N=1 的稳定性

在当前架构上先建立 contemporaneous strict/D2 baseline，再补第二 seed。不要直接把旧
seed42 的 34.897 AP 与当前 SDSR/projector 结果比较。

如果只是复现历史阶段，补：

```text
D2 enhanced
CopyBlend N=1
p=0.5
context=0.1--0.25
stop=21
seed=43
```

## P2：解决 Dense O2O × CDN 显存放大

给 CDN 增加 hard GT/query cap，并做三项验证：

1. strict batch 不触发 cap 时输出完全一致；
2. dense batch 触发 cap 后峰值显存明显下降；
3. cap 只影响 CDN targets，不删除 Hungarian matching 的任何 GT。

建议比较 `max_gt=20/25/32`，先看显存和 DN loss 稳定性，再跑完整 AP。

## P3：课程式路径退火

Dense O2O 自身已有数据课程。下一步更有研究价值的是 Decoder 优化路径退火：

| 实验 | Epoch 0--20 | Epoch 21--23 |
|---|---|---|
| T0 | G6 + CDN | G6 + CDN |
| T1 | G6 + CDN | G6，无 CDN |
| T2 | G6 + CDN | active Group1 + CDN |
| T3 | G6 + CDN | active Group1，无 CDN |

这里的 Group1 应理解为后期只优化推理实际使用的第一组 Query，而不是在训练中动态改变
Parameter tensor 形状。实现时应通过 active-group mask/loss routing 完成，避免破坏 optimizer
state 和 checkpoint 兼容性。

重点记录：best AP、final AP、last-3/last-5 mean、AP75、收敛震荡、时间和显存。

## P4：不要继续的方向

- 不继续穷举 SA 预算或 start/stop window；
- 不重复 DN 100/200 或 loss 0.25/0.7；
- 不重复 N=1 + context 0.00--0.10；
- 不把 CopyBlend stop 提前到 12；
- 不在不同 total batch size、不同代码版本之间宣称方法增益；
- 不把 Dense O2O 描述成 matcher 改进；
- 不把 CDN 描述成 Hungarian matching 分支。

## 14. 对本会话贡献的总体评价

本会话的核心贡献不是单独增加一个 loss，而是把 RF-DETR-DINOv3 的 Decoder 训练策略从
一个难以拆分的 Group baseline，整理成了四个边界清楚、可以独立消融的模块：

```text
Group O2O matching
+ CDN known-assignment denoising
+ Dense target curriculum
+ optional scale-aware auxiliary matching
```

通过实现、双 seed、Full COCO 和反例实验，最终得到：

- Group 6 比 Group 13 更经济且不损失精度；
- CDN 是可保留的稳定收益；
- Dense O2O D2 是可保留的定位收益；
- 当前 SA-Matching 没有证明价值；
- 多条优化路径并非越多越好，监督强度和训练/推理路径差异需要后期退火；
- Dense targets 与 CDN/Group 存在重要的非线性显存交互。

这使后续研究可以从“继续堆模块”转向更明确的问题：限制辅助 Query 成本、控制优化路径
数量、在训练后期恢复严格推理路径，并在当前最强 projector 上重新验证这些结论。
