# RF-DETR-DINOv3 分割方向与首轮实验

时间：2026-08-24（北京时间）

## 1. 结论先行

建议采用“三条分工明确的路线”，而不是在常规掩码头和无 decoder 方法之间二选一：

1. **工程主线：RF-DETR 查询条件掩码头。** 它可以直接继承已经训练好的类别、定位和实例查询。medium 实验表明，先冻结检测器训练 P3 掩码头、再在冻结状态引入零门控 P4/P5 上下文，是当前保检测的安全默认；联合解冻能继续大幅提高 mask AP，但必须作为有明确 bbox AP 预算的可选档。
2. **研究主线：EoMT 风格的 encoder-only segmentation queries。** “去掉 decoder”本身已经不是足够的新意；更有研究价值的是让 encoder-only 分割查询与 RF-DETR 检测查询共享、初始化或蒸馏，并研究这种双路径是否能用更低开销改善边界或任意概念分割。
3. **能力扩展：INSID3 风格的免训练任意概念分割。** 它适合作为 one/few-shot 模式和 DINOv3 语义诊断工具，不适合作为闭集 COCO 实例分割主干的直接替代。

现阶段不建议删掉 RF-DETR decoder 后直接押注 EoMT。首轮可执行原型证明这条路线确实能学习实例掩码、显存也很低，但在相同的 DINOv3-S 和极小训练集条件下，收敛、泛化以及小目标分割均明显弱于复用预训练检测查询的常规方案。

工程默认配置应改为 **P3 mask-head-only warm-up → 冻结 detector 的零门控 P4/P5 上下文训练**。medium 上 1e-6/1e-7 联合一轮只比冻结上下文多 0.10 mask AP，却少 0.63 bbox AP；5e-6/5e-7 则是明显偏向分割的激进档。P4/P5 上下文仍不能解决小目标，5%/25% 硬边界采点也均为负结果，必须默认关闭。研究创新应建立在可靠主线之上，而不是用“非传统结构”替代尚未建立的基线。

## 2. 对现有项目的理解

### 2.1 已有检测工作

代码、输出目录、训练日志和既有 Codex 会话共同显示，这个项目不是简单的 RF-DETR backbone 替换，而是已经围绕 DINOv3 特征、projector、decoder、查询和训练机制做过系统搜索：

本轮不仅查看了当前工作树，还回溯了 `output/` 下各阶段 `log.txt` / `results.json`、项目内多份阶段总结，以及父级 Codex 会话 `/home/cpc/.codex/sessions/2026/05/24/rollout-2026-05-24T17-39-42-019e595a-c817-75c1-a139-5127ef69921f.jsonl`。下面的判断因此以实际实现和历史对照为准，不把目录名或单个最佳 checkpoint 当成完整结论。

- P3/P4/P5 多尺度是检测性能的重要来源；P3 是主要增益项。日志中 P3/P4/P5 达到 51.60 AP，而仅 P4 为 48.72 AP。
- 已验证 iterative bbox refinement、scale routing、Group-DETR、CDN、DenseO2O，以及 SDSR 系列 projector。
- medium 子集上的 iterative-refinement + scale-routing 检查点达到 36.142 AP；SDSR-v36 为 36.059 AP。
- 检测训练还揭示了一个重要规律：refinement 更像参数预条件，直接保留其约束或降低学习率并不能稳定保留收益。
- 冻结 DINOv3 的 ADE 语义探针中，官方 small backbone 的 mIoU 为 0.408883，已有 backbone refinement 大多为 0.4075–0.4083，说明当前 refinement 并未直接增强冻结语义分割能力。

这些结果决定了分割方案应优先复用已学好的检测查询和 P3，而不是重新从 DINOv3-S 随机学习一整套实例查询。

### 2.2 项目中原本已有的分割能力

项目已经包含 `SegmentationHead`、mask Hungarian cost、点采样 BCE/Dice loss、`pred_masks` 和 COCO `segm` evaluator。现有头的核心是：

- 使用 P3 空间特征；
- 每个 decoder 层对空间特征做轻量 depthwise block；
- 将 decoder query 经过 MLP；
- 通过空间特征和查询向量点积得到实例掩码。

因此，“常规分割头”不是从零开发，而是需要解决训练策略、检测配置对齐、多尺度细节和现有增强兼容性。

当前限制：

- segmentation + CDN 训练目前显式报错；
- DenseO2O 的图像级变换没有同步维护 masks；
- 首轮常规实验为隔离变量关闭了 CDN 和 DenseO2O；
- 早期 Group-DETR=1 实验与原检测检查点的 Group-DETR=6 不完全对齐，后续已补充 Group=6 对照。

## 3. 两篇论文回答的是不同问题

### 3.1 Your ViT is Secretly an Image Segmentation Model / EoMT

EoMT 并不是“完全没有分割头”或“没有查询”。它去掉的是独立 pixel adapter 和 Transformer decoder，但仍然包含：

- 学习式 segmentation queries；
- 将 queries 插入 ViT 最后若干原生 block，与 patch tokens 联合 self-attention；
- 线性分类头；
- 三层 mask MLP；
- query embedding 与上采样 patch feature 点积得到 mask；
- 中间层 mask supervision，以及论文中很关键的 masked attention / annealing。

论文消融表明 masked attention 不是装饰：去掉它约损失 3 PQ（53.2 对 56.0）。模型规模也很关键：ViT-S 相对 ViT-Adapter + Mask2Former 低 5.8 PQ，而 ViT-L 差距缩至 1.1，ViT-g 约 0.7。因此在当前 DINOv3-S 上直接判断 encoder-only 路线的最终上限并不公平，但小模型恰好也是本项目需要面对的效率约束。

### 3.2 INSID3

INSID3 是 training-free in-context segmentation，不是闭集、全监督的 COCO instance segmentation：

- 冻结 DINOv3-L；
- 用噪声或常量图像特征的 SVD 估计并去除位置偏置；
- 聚类目标特征；
- 做前向/反向 patch matching；
- 用 cluster seed 和聚合得到目标区域。

论文在八个数据集上的平均 mIoU 为 55.1%，但使用的是 304M 参数的 DINOv3-L，输入任务也提供 reference image/mask。它最适合做任意概念模式、交互式分割或验证 DINOv3 潜在语义，不应与 COCO 闭集实例 AP 直接横向比较。

## 4. 创新性判断

### 4.1 单纯“去 decoder”不够新

EoMT 已经系统证明了把查询插入 ViT 尾部的路线，官方实现也已扩展到 DINOv3。若只是将相同结构移植到 RF-DETR-DINOv3，创新性主要是工程复现，而不是新的方法贡献。

### 4.2 更值得做的创新点

建议把论文问题定义为：**检测 decoder 中已经形成的实例查询知识，能否帮助 DINOv3 内部的 encoder-only 分割查询以更低开销获得高质量掩码？** 可检验的实现包括：

1. **Query bridge：** 检测查询经可学习投影初始化 segmentation queries，或由检测 top-k query 动态生成。
2. **双路径蒸馏：** 常规 RF-DETR mask head 作 teacher，encoder-only mask branch 作 student；推理时可按速度需求裁掉 teacher。
3. **共享 mask embedding：** decoder query 和 encoder query 使用同一个 mask embedding/pixel embedding 空间，直接约束跨路径一致性。
4. **尺度分工：** encoder-only 路径负责全局/大目标语义，P3 查询掩码头负责小目标和边界，再由置信门控融合。
5. **INSID3 初始化：** 对未见类别或提示对象，用 DINOv3 in-context cluster 形成无参数 query seed，再进入轻量 refinement。

这几个问题都能构成明确消融，且真正利用了当前项目已积累的检测查询、多尺度和训练策略，而不是为了“非传统”而非传统。

## 5. 本轮实现

### 5.1 常规主线

- runner 增加 segmentation head、mask loss/downsample/point sampling 等显式配置；
- 支持完整 detector checkpoint（可选择包含 encoder）精确 warm start；
- 增加 `segmentation_head_only` 阶段，只保留约 0.933M mask-head 参数训练；
- 冻结阶段将 detector 保持 eval，避免 BN/dropout 状态漂移；
- 增加可选 P4/P5 语义上下文支路：以 P3 为高分辨率锚点，P4/P5 经轻量投影后上采样，并由逐通道零初始化门控融合；门控为零时与原 P3-only 头严格等价；
- 增加默认关闭的 target-boundary point sampler：可用指定比例的一像素形态学边界点替换原 PointRend 不确定性点，空/常量 mask 自动回退均匀采样；
- 增加检查点分割可视化脚本。

### 5.2 INSID3 诊断

实现了 DINOv3 特征抽取、位置基去偏、聚类、reference mask、双向匹配和聚类聚合，并加入 CPU 单元测试和 COCO one-shot probe。

### 5.3 EoMT 风格原型

实现了一个与 RF-DETR 隔离的、可训练的 encoder-only 对照：

- queries 插入 DINOv3 最后 N 个原生 block；
- 兼容 DINOv3 RoPE（查询作为不旋转的 prefix，末尾 H×W patch 继续使用原 RoPE）；
- class linear + 3-layer mask MLP + F4 patch map dot product；
- 中间层辅助损失；
- 可选的 query-to-patch masked attention：每个 query 只读取其中间预测掩码覆盖的 patch；
- 按 ViT 尾部 block 分层退火，早期 block 先解除约束，推理时全部恢复为未掩码注意力；
- 可冻结 query 插入前的图像前缀；
- COCO Hungarian matching、BCE/Dice/class loss、标准 segm AP；
- 额外提供独立于分类分数的 oracle mask IoU/Recall 和 mask-selected class accuracy。
- 增加检测查询初始化实验：将预训练 RF-DETR query 经固定半正交映射从 256 维提升到 384 维，在保持归一化 Gram 几何和初始化方差的前提下初始化 encoder-only queries。

实现同时保留未掩码控制组。掩码注意力路径已通过真实 DINOv3-S 前后向验证；它复现了论文的核心训练机制，但训练配方、数据规模和 backbone 规模仍不是官方 EoMT 完整复现，因此结果只用于项目内路线筛选。

## 6. 首轮实验

除单独标注的 quick/medium 扩展实验外，下面的 COCO 常规训练结果来自 128 张训练图、64 张验证图的功能性 overfit 子集；它们适合比较训练稳定性，不代表完整 COCO 泛化性能。

### 6.1 常规 RF-DETR 掩码头

| 训练策略 | epoch | bbox AP | mask AP | mask AP50 | 结论 |
|---|---:|---:|---:|---:|---|
| detector + mask 全量联合，LR 1e-4 | 1 | 33.48 | 1.39 | 4.82 | 检测立即灾难性遗忘 |
| detector + mask 全量联合，LR 1e-4 | 5 | 31.85 | 11.64 | 19.89 | mask 最快，但检测损失约 18.5 AP |
| 仅 mask head | 5 | 50.33 | 4.45 | 12.36 | 检测完全保持 |
| 仅 mask head | 20 | 50.33 | 7.81 | 20.58 | 收敛慢但稳定 |
| 从原 detector 直接联合，LR 1e-5 | 10 | 45.33 | 5.92 | 15.50 | 降 LR 仍不足以避免遗忘 |
| mask-only 20 后联合，LR 1e-5 / encoder 1e-6 | 4 | **50.24** | **10.78** | **25.53** | 该 overfit 阶段最佳 Pareto 点 |
| 同上 | 5 | 50.77 | 9.76 | 25.80 | bbox 仍保持，mask 有小幅波动 |
| Group6、仅 mask head | 20 | 50.33 | 7.06 | 18.03 | 严格同构加载，冻结阶段不优于 Group1 |
| Group6 mask-only 20 后联合 | 5 | **51.78** | **10.69** | **24.79** | 与 Group1 mask AP 持平，bbox Pareto 更强 |
| Group6、P3+P4/P5 零门控上下文、仅 mask head | 17 | 50.33 | **9.07** | 20.01 | 冻结检测器时比同构 P3-only 最佳高 2.01 AP |
| P3+P4/P5 mask-only 后联合 | 4 | 50.60 | **10.76** | 23.21 | mask 略高但 bbox 低于 P3-only 最佳，不是新 Pareto 点 |

最重要的结果不是某个绝对 AP，而是训练顺序：随机 mask head 的梯度会破坏检测器；先让 mask head 进入可用尺度，再小学习率解冻，能把 mask AP 从约 7 推到约 10.7，同时保持检测。Group6 冻结阶段没有自动带来收益，但联合阶段最终达到 bbox/mask AP 51.78/10.69；因此主线应保留与最佳检测器一致的 Group6，而不是为分割回退到 Group1。

面积分解显示当前 mask head 仍偏向大目标。渐进式第 4 轮的 APs/APm/APl 为 2.17/6.69/22.33，下一步应优先改 P3 小目标和边界，而不是继续盲目联合微调。

P3+P4/P5 上下文的最佳冻结点 APs/APm/APl 为 1.53/5.44/16.52；相同 epoch 的 P3-only 为 1.56/3.89/12.19。它主要增强中大目标语义，并没有稳定提升小目标。训练后 P4/P5 门控的平均绝对值分别为 0.0224/0.0274，证明两个支路确实被使用。渐进解冻后最佳 mask AP 从 P3-only 的 10.69 微升到 10.76，但 bbox AP 从 51.78 降到 50.60，因此没有构成新的严格 Pareto 点。这进一步说明后续“小目标/边界”实验仍应围绕 P3 分辨率和采样，而不能只堆低分辨率语义。

在更大的 quick 子集（1,000 train / 500 val）上，Group6 P3-only mask-head-only 第 9 轮达到 bbox/mask AP 39.42/8.79，mask AP50/AP75 为 18.69/6.55，APs/APm/APl 为 1.63/7.59/17.44。十轮中 bbox AP 始终严格保持 39.42，说明冻结 warm-up 的稳定性不只是 128/64 过拟合集的偶然现象。

从该冻结点以 detector/encoder LR 1e-5/1e-6 联合解冻，三轮 bbox/mask AP 依次为 38.25/11.67、37.64/13.66、37.06/14.30。分割收益很大，但检测损失也单调累积；第 3 轮 APs/APm/APl 已达到 2.55/12.41/30.43。它说明渐进解冻的方向成立，却也说明不能只按 mask AP 选模型。

将 LR 减半为 5e-6/5e-7 后，三轮 bbox/mask AP 为 **39.45/10.66、38.50/12.08、37.94/13.03**，整体形成更好的 Pareto。epoch 1 几乎无检测损失，epoch 2 严格支配高 LR epoch 1（bbox +0.25、mask +0.41）。已分别保留 `checkpoint_epoch1_pareto.pth` 和 `checkpoint_epoch2_balanced.pth`，不让后续“只按 mask AP”保存逻辑覆盖它们。

同图可视化表明，平衡模型能把鸟群场景中冻结头错误的 sheep 分类纠正为 bird，并提高拥挤车辆的实例置信；但掩码内部仍有孔洞，细杆和物体边缘仍有明显粗糙/外溢。定量和定性证据都支持同一判断：渐进解冻主要改善 query—类别和实例对齐，下一阶段需要专门的边界/高分辨率机制。

为了区分“多训练三轮”和“多尺度上下文”，又从同一个 quick epoch-9 检查点冻结 detector，严格比较 P3 继续训练与新增零门控 P4/P5。第 3 轮 P3 为 8.86 AP，P3+P4/P5 为 9.52（+0.65）；后者 AP50/AP75 分别高 0.99/1.16，APl 高 2.08，但 APs/APm 低 0.38/0.07。它把 overfit 子集的 +2 AP 修正为更可信的 +0.65 AP，并再次证明收益集中于大目标/高 IoU 语义，不是小目标或边界解法。

边界采点也做了同检查点、同三轮的直接对照：P3 control 为 8.86 AP，25% boundary points 为 7.12，5% 为 7.34；两个比例的 APs/APm/APl 都全面落后。5% 没有消除破坏，说明问题不只是比例过大，而是把困难窄边带混入同一个 BCE/Dice 采样集合会改变成熟头的优化尺度。该选项因此保持默认 0，仅作为负对照基础设施；下一版应尝试不挤占原采样预算的低权重独立 boundary loss，并在 mask warm-up 后延迟启用。当前实现直接在 640×640 GT 上计算形态学边界，训练峰值显存约从 2.6–3.7 GiB 增到 6.8 GiB，正式版本还应预计算或下采样边界图。

进一步扩大到 medium 子集（5,000 train / 2,000 val）后，Group6 P3-only mask-head-only 第 10 轮达到 bbox/mask AP **35.91/11.39**，mask AP50/AP75 为 **24.82/9.81**，APs/APm/APl 为 **1.68/9.45/24.61**。第 1 轮 mask AP 为 5.19，而十轮 bbox AP 始终逐位保持 35.91；这比 1k/500 quick 集更有力地确认了 warm-up 的稳定性，也说明该头在十轮时仍未明显饱和。其代价仍集中在小目标：APl 已达 24.61，APs 只有 1.68，后续优化必须优先提高 P3 有效分辨率和细结构监督。

medium 的后续对照给出了更可信的工作点：

| 从 P3 warm-up 最佳点继续 | epoch | bbox AP | mask AP | AP50/AP75 | APs/APm/APl |
|---|---:|---:|---:|---:|---:|
| P3-only，detector 冻结 | 3 | **35.91** | 11.87 | 25.63/9.72 | 1.91/9.90/25.15 |
| P3+P4/P5 上下文，detector 冻结 | 3 | **35.91** | **13.34** | 27.66/11.36 | 1.97/10.57/28.64 |
| 全量联合，LR 1e-6 / encoder 1e-7 | 1 | 35.28 | 13.44 | 27.93/11.49 | 2.04/11.11/29.44 |
| 全量联合，LR 5e-6 / encoder 5e-7 | 1 | 34.75 | 16.09 | 31.79/14.30 | 2.48/13.95/35.28 |
| 同上 | 2 | 33.94 | 17.63 | 33.55/16.33 | 2.85/15.91/38.13 |
| 同上 | 3 | 33.53 | **18.98** | 35.65/17.88 | 3.64/17.46/40.40 |

严格同轮数比较时，P4/P5 上下文相对 P3-only 的净 mask AP 为 **+1.47**，AP50/AP75 为 +2.03/+1.65；APs/APm/APl 净增益为 +0.06/+0.67/+3.49，且 bbox AP 完全不变。最终 P4/P5 门控平均绝对值达到 0.0443/0.0489。它证明支路有效，同时也把收益边界定位得很清楚：大目标很强，小目标几乎没有净增益。

上下文支路只增加 132,608 参数（41.507M→41.640M，约 +0.32%）。RTX 5090、batch=1、640 输入、eager、无后处理、30 次 warm-up + 200 次重复的只读基准中，TF32 中位延迟 24.14→24.86 ms，FP16 中位延迟 30.47→31.17 ms，约增加 2%–3%；峰值显存按精度增加约 31–44 MiB。因此 +1.47 mask AP 的工程开销很小。分割头 `einsum` 尚不受项目旧 FLOP tracer 支持，本轮显式跳过 FLOP 统计，没有用不完整数字估算算力。

当前 PyTorch/CUDA eager 栈在这个小 batch 上的 FP16 绝对延迟反而高于 TF32，因此不将跨精度的绝对数字解读为理论吞吐；本轮只使用同一精度下 P3 与 context 的相对差值做工程判断。

medium 也修正了 quick 集对联合解冻的乐观估计。1e-6/1e-7 联合一轮相对冻结上下文只多 0.10 mask AP，却少 0.63 bbox AP；5e-6/5e-7 第一轮虽然增加 4.70 mask AP，但已经损失 1.16 bbox AP，第三轮损失扩大到 2.38。因而冻结上下文是当前安全默认；若任务愿意交换检测精度，分别保留 5e-6 第 1/2 轮检查点作为激进 Pareto 点，而不是只保留 mask AP 最高的末轮。

三组 medium 同图可视化进一步显示：冻结上下文主要改善掩码覆盖，不改变类别/框分支；1e-6 联合会重新分配 query 置信度，但边界几何改善有限。拥挤人群仍有实例粘连，细杆、小物体和内部孔洞仍明显，和面积 AP 分解一致。

### 6.2 INSID3 风格 one-shot probe

DINOv3-S、448 输入、5 类×6 个 COCO episodes；这是项目内诊断，不是论文官方 fold protocol：

| 去偏秩 | mean IoU | std |
|---:|---:|---:|
| 0 | 0.5091 | 0.3214 |
| 64 | 0.5384 | 0.2870 |
| 128 | 0.5386 | 0.2920 |
| 192 | **0.5509** | 0.2927 |

位置去偏带来约 +4.2 IoU 点，验证了 DINOv3 的 latent mask 和位置偏置假设；但标准差接近 0.29，类别/episode 失败率很高。因此它适合作为辅助模式，不适合直接替换主监督头。

### 6.3 Encoder-only EoMT 风格控制组

配置：DINOv3-S、320×320、100 queries、F4 masks、只训练尾部 block 和 heads；128/64 overfit 子集，score threshold 0.001。

| 尾部 query blocks / 训练机制 | 可训练参数 | 最佳 epoch | val mask AP | AP50 | APs/APm/APl | val oracle IoU | mask-selected class acc. |
|---|---:|---:|---:|---:|---|---:|---:|
| 2 / 始终未掩码 | 4.11M | 48 | **5.98** | 11.15 | 0.05/0.17/12.25 | **0.302** | 34.1% |
| 4 / 始终未掩码 | 7.66M | 47 | 5.15 | 7.60 | 0.00/0.15/14.27 | 0.305 | 42.1% |
| 2 / masked attention 退火 | 4.11M | 50 | 5.58 | 10.82 | 0.00/0.96/13.54 | 0.300 | 45.1% |
| 2 / 退火 50 + 未掩码 10 | 4.11M | 57 | 5.70 | **11.87** | 0.00/0.62/11.76 | 0.291 | **45.6%** |
| 2 / RF-DETR 检测查询初始化 | 4.11M | 37 | 5.76 | 10.35 | 0.00/0.02/12.25 | 0.292 | 39.5% |

2-block 检查点的训练集 AP/验证集 AP 为 14.15/5.98；4-block 为 20.45/5.15，说明更深尾部主要增加拟合与大目标能力，并没有改善泛化。第 10 到第 48 轮，2-block 的验证 oracle IoU 只从 0.269 增到 0.302，但 AP 从近零升至 5.98，表明后半段收益很大一部分来自类别/查询置信度收敛，而不是掩码形状质变。

masked attention 的真实 DINOv3-S 前后向、逐层概率退火和推理期全未掩码路径均已验证。它把 mask-selected 分类正确率从 34.1% 提高到约 45%，但 oracle mask IoU 没有提高，最终 AP 也没有超过始终未掩码控制组。这个小实验不复现论文的大模型收益，却提示该机制更可能在查询—类别对齐、而非 F4 掩码几何上起作用。当前 masked 训练还比控制组多一个中间辅助输出，因此下一轮正式复现需要同时对齐 loss 权重和训练长度。

静态检测查询初始化也没有提高最终上限：最佳 5.76 AP，略低于随机查询的 5.98。但它明显加速中期收敛：epoch 20 为 1.82 对 0.39 AP，epoch 37 为 5.76 对 4.57 AP。其验证 oracle IoU 为 0.292，低于随机查询的 0.302，而 mask-selected 分类正确率从 34.1% 提高到 39.5%；这说明迁移主要改善查询/类别对齐，没有改善 F4 掩码几何。这是一个有价值但边界清楚的正信号——检测 query 的槽位几何可迁移，但固定随机方向的 256→384 映射不是最终 bridge。下一步应比较可学习投影、动态 top-k detection queries 和 mask-head teacher distillation，并把“收敛速度”和“最终 AP”分开报告。

640×640、100 queries、2 blocks 的一次完整前反向峰值显存约 302 MiB（不含进程基础占用），说明这条路线确实很轻，具备升级 DINOv3-B/L 的空间。

## 7. 推荐实验顺序

### Phase A：先建立可信的常规完整 COCO 基线

1. 使用与最佳检测检查点一致的 Group-DETR 配置；只关闭尚未支持 masks 的 CDN/DenseO2O。
2. mask-head-only warm-up，以验证集 mask AP/损失而不是固定 epoch 决定切换点。
3. detector 保持冻结，从 warm-up 最佳点引入零门控 P4/P5 上下文；这是当前 medium 上最好的安全工作点。
4. 若需要更高 mask AP，再做带 bbox AP 预算的联合解冻。medium 上 1e-6/1e-7 一轮已经损失 0.63 bbox AP，5e-6/5e-7 属于激进档；正式实验应尝试只解冻 query/mask 相关层、检测 teacher consistency，或从更低 LR 起步。
5. 在 1k/5k 子集后再上 full COCO，避免把 128/64 子集结论当泛化结论。

### Phase B：提升小目标与边界

1. 保留已经验证的 P3+P4/P5 语义上下文，但下一步把变量转向 P3 有效分辨率：轻量 stride-8 detail path、局部高分辨率重采样或只对小目标裁块细化。
2. 对小/中目标提高原不确定性 point sampling 密度；边界监督使用低权重、延迟启用的独立 loss，不要像本轮负对照那样直接替换 5%–25% 原采样点。
3. 研究 mask feature 与已有 scale routing 的共享，而不是另建重型 pixel decoder。
4. 补齐 CDN masks 和 DenseO2O mask 变换后，再恢复检测最佳训练配方。

### Phase C：做真正有论文价值的 hybrid

1. 在已经实现的 masked attention/annealing 上复现公开训练配置，避免拿 128/64 小子集结果判断完整方法上限。
2. 至少使用 DINOv3-B；论文已表明 ViT-S 对 encoder-only 不利。
3. 做四组核心消融：随机 encoder queries、检测 query 初始化、teacher distillation、共享 mask space。
4. 对比参数、峰值显存、吞吐、bbox AP、mask AP 和任意概念 mIoU，明确创新的收益边界。

## 8. 复现入口与产物

主要代码：

- `src/rfdetr/models/eomt.py`
- `src/rfdetr/models/insid3.py`
- `tools/train_eomt_coco_subset.py`
- `tools/probe_insid3_coco.py`
- `tools/smoke_eomt.py`
- `tools/visualize_segmentation_checkpoint.py`
- `tools/benchmark_detector.py`（已支持分割 checkpoint 与 `--skip-flops`）
- `run_coco_subset.py`

关键输出：

- 常规 mask-only：`output/segmentation_warmstart/full_detector_overfit_maskonly_e5`
- 渐进解冻：`output/segmentation_warmstart/maskonlye20_then_jointlr1e5_e5`
- Group6 mask-only：`output/segmentation_warmstart/group6_maskonly_overfit_e20`
- Group6 渐进解冻：`output/segmentation_warmstart/group6_maskonlye20_then_jointlr1e5_e5`
- Group6 多尺度上下文冻结训练：`output/segmentation_warmstart/group6_maskonly_multicontext_overfit_e20`
- Group6 多尺度上下文渐进解冻：`output/segmentation_warmstart/group6_multicontexte20_then_jointlr1e5_e5`
- Group6 quick mask-only：`output/segmentation_quick/group6_maskonly_e10_bs4`
- Group6 quick 渐进解冻：`output/segmentation_quick/group6_maskonlye10_then_jointlr1e5_e3_bs4`
- Group6 quick 更低 LR 渐进解冻：`output/segmentation_quick/group6_maskonlye10_then_jointlr5e6_e3_bs4`
- Group6 quick P3 继续训练控制：`output/segmentation_quick/group6_p3e9_then_maskonly_e3_control_bs4`
- Group6 quick P3→P4/P5 上下文：`output/segmentation_quick/group6_p3e9_then_context_maskonly_e3_bs4`
- Group6 quick 25%/5% boundary 负对照：`output/segmentation_quick/group6_p3e9_then_boundary025_maskonly_e3_bs4`、`output/segmentation_quick/group6_p3e9_then_boundary005_maskonly_e3_bs4`
- Group6 medium mask-only：`output/segmentation_medium/group6_maskonly_e10_bs4`
- Group6 medium P3 继续训练控制：`output/segmentation_medium/group6_p3e10_then_maskonly_e3_control_bs4`
- Group6 medium P3→P4/P5 上下文：`output/segmentation_medium/group6_p3e10_then_context_maskonly_e3_bs4`
- Group6 medium 1e-6/1e-7 联合：`output/segmentation_medium/group6_maskonlye10_then_jointlr1e6_e1_bs4`
- Group6 medium 5e-6/5e-7 联合：`output/segmentation_medium/group6_maskonlye10_then_jointlr5e6_e3_bs4`（另存第 1/2 轮 checkpoint）
- 可视化：`output/segmentation_visuals/maskonly_e20`
- quick 冻结/平衡模型同图可视化：`output/segmentation_visuals/quick_maskonly_best`、`output/segmentation_visuals/quick_lr5e6_epoch2_balanced`
- medium 同图可视化：`output/segmentation_visuals/medium_p3_maskonly_best`、`output/segmentation_visuals/medium_context_best`、`output/segmentation_visuals/medium_jointlr1e6`
- medium 推理基准：`output/segmentation_benchmarks/medium_p3_maskonly_best.json`、`output/segmentation_benchmarks/medium_context_best.json`
- INSID3 probe：`output/insid3_coco_probe/dinov3_small_coco30_r448`
- EoMT 2-block：`output/eomt_coco_subset/dinov3_small_tail2_q100_r320_e50`
- EoMT 4-block：`output/eomt_coco_subset/dinov3_small_tail4_q100_r320_e50`
- EoMT 2-block 掩码退火：`output/eomt_coco_subset/dinov3_small_tail2_maskanneal_q100_r320_e50`
- EoMT 退火后未掩码收尾：`output/eomt_coco_subset/dinov3_small_tail2_maskanneal50_then_unmasked10`
- EoMT 检测查询初始化：`output/eomt_coco_subset/dinov3_small_tail2_detqueryinit_q100_r320_e50`

推荐主线的关键参数可复现为两段冻结训练：第一段在原检测配置后附加 `--segmentation-head --segmentation-head-only --mask-feature-levels 1 --lr 1e-4 --lr-encoder 1e-5`；第二段从第一段的 `checkpoint_best_regular.pth` 重启，保持 `--segmentation-head-only --lr 1e-4 --lr-encoder 1e-5`，改为 `--mask-feature-levels 3`。若要复现联合 Pareto，再从第一段 checkpoint 去掉 `--segmentation-head-only`，分别使用 1e-6/1e-7 或 5e-6/5e-7 的 detector/encoder LR。所有阶段都要保留 `--detector-pretrain-include-encoder`，否则不是当前报告中的精确 warm start。运行器会把完整参数写入 checkpoint 的 `args`，上述输出目录中的检查点可直接审计其余检测配置。

参考材料：

- 本地 EoMT 论文：`/data/cpc/root/storage/paper/分割/Kerssies 等 - 2025 - Your ViT is Secretly an Image Segmentation Model.pdf`
- 本地 INSID3 论文：`/data/cpc/root/storage/paper/分割/Cuttano 等 - 2026 - INSID3 Training-Free In-Context Segmentation with DINOv3.pdf`
- EoMT 官方实现：https://github.com/tue-mps/eomt
- INSID3 官方实现：https://github.com/visinf/INSID3

## 9. 结果解释边界

- overfit 子集很小，类别长尾严重，单轮 AP 波动明显；首轮数字用于路线筛选而非论文主表。
- 常规方案继承了一个已经学习过 COCO 检测的检查点；encoder-only 控制组只继承 DINOv3，自身的 80 类分类/query heads 随机初始化，两者不是同等预训练条件。这恰好说明“共享检测查询”是最值得研究的 hybrid 变量，但不能据此断言完整 EoMT 一定较差。
- masked attention/annealing 已实现，但当前小实验比论文多一个中间辅助输出，且尚未对齐官方完整训练配方；不能把这组结果直接等同于论文消融表中的 3 PQ 差异。
- 实验开始时 GPU 0/2/3 均有其他进程，长基线只使用空闲的 GPU 1；后续 GPU 3 释放出足够余量后，仅在其上并行运行约 5 GiB 的冻结头对照。GPU 2 始终接近满显存，GPU 0 始终有活跃负载，本轮没有中止或干扰这些进程。
- 明确限定正式 `tests/` 目录运行全量回归，209 项测试全部通过；相关 runner、模型与工具均通过 `py_compile`，定向 diff 无空白错误。唯一测试警告是隔离环境无法联网检查 Albumentations 版本，与实现无关。从仓库根目录无限定收集会误入 `experiment_snapshots/` 中的重复源码，产生 pytest `ImportPathMismatchError`，因此不是有效的回归命令，也不是本次改动回归。
