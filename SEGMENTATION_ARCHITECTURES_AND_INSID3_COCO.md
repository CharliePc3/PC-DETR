# 分割结构、训练路径与 INSID3 COCO 实例化

时间：2026-08-28（北京时间）

## 1. RF-DETR 原生 SegmentationHead

### 1.1 模型结构

上游 RF-DETR 的实例分割并没有独立 Transformer mask decoder。检测器先生成多尺度 projector 特征和每层 detection decoder queries，然后复用：

1. 取 projector 的第一层高分辨率特征（在当前项目配置中是 P3）；
2. 将它插值到输入尺寸的 1/4；
3. 对每个 detection decoder 层依次施加一个简化 ConvNeXt block：depthwise 3×3、LayerNorm、pointwise linear、GELU 和 residual；
4. 空间特征经 1×1 projection，decoder query 经 residual MLP 和 linear projection；
5. 使用 `einsum("bchw,bnc->bnhw")` 点积并加一个标量 bias，得到每个 detection query 的 mask logits。

空间分支的 blocks 是串行的：第 `i` 层 query 使用经过前 `i+1` 个空间 block 的特征，而不是每层从同一 P3 独立开始。类别、框和 mask 因而绑定在同一个 query slot 上。

### 1.2 训练路径

训练时 `sparse_forward` 不立即物化完整的 query×mask logits，而是返回空间 embedding、query embedding 和 bias：

1. Hungarian matcher 在随机归一化坐标上采样预测/GT mask；
2. assignment cost 包含 class、L1 bbox、GIoU、mask sigmoid-CE 和 mask Dice；
3. assignment 完成后只选 matched queries；
4. PointRend 式采点先随机 oversample 3 倍，再选择 75% 最不确定点和 25% 随机点；
5. matched masks 使用 sigmoid BCE + Dice loss；
6. decoder auxiliary outputs 和 two-stage encoder output 也获得相应 mask loss。

推理时分类概率按 query×class 全局 top-k，取相应 query 的 box 和 mask；mask 双线性插值到原图后以 logit 0 为阈值，最后转换为 COCO RLE。

### 1.3 当前 RF-DETR-DINOv3 继承版本

当前版本保留上述核心结构、matching、loss 和 postprocess，只增加了：

- `mask_feature_levels=1` 时严格使用原 P3 路径；
- `mask_feature_levels=3` 时，以 P3 为锚点，P4/P5 经 1×1 Conv、GroupNorm、GELU 后上采样，并通过逐通道零初始化 gate 相加；gate 为零时初始输出与 P3-only 完全一致；
- 可选 target-boundary point replacement；实验为负，默认比例为 0；
- `segmentation_head_only`：冻结 detector 参数并保持其 eval 状态，只训练 mask head。

已有 medium baseline 并非从官方 backbone 直接训练分割，而是：官方 DINOv3-S → medium 子集检测训练 24 轮 → 加入随机 RF-DETR mask head → mask-only/联合训练。因此它应标记为 **detector-pretrained transfer baseline**，不能与从 DINOv3 权重开始的 encoder-only 方法直接做公平架构比较。

## 2. 当前 EoMT 风格 encoder-only 原型

### 2.1 Token 与模块结构

输入经过 DINOv3 `prepare_tokens_with_masks`，形成：

```text
[CLS, storage/register tokens, H×W patch tokens]
```

先正常运行 query 插入点之前的 ViT blocks，然后把 `Q` 个可学习 segmentation queries 插在 special tokens 与 patch tokens 之间：

```text
[CLS, storage tokens, segmentation queries, patch tokens]
```

这些 token 共同通过最后 `N` 个原生 DINOv3 blocks。RoPE 仍只应用到末尾 H×W patch tokens，新增 queries 是不旋转的 prefix。

每个参与 query 的 block 后进行预测：

- segmentation query → linear class head，输出 80 类 + no-object；
- segmentation query → 3 层 mask MLP；
- patch tokens → H×W map → 插值到 F4 → 1×1 pixel projection；
- query mask embedding 与 pixel embedding 点积得到 Q 个 masks。

可选 masked attention 使用上一中间预测限制 query→patch attention；mask 会 detach，并按 block 深度逐步退火，推理时恢复普通 self-attention。query 插入前的图像 tokenizer/blocks 可冻结，尾部 blocks、norm、queries 和 heads 可训练。

### 2.2 数据与监督路径

当前训练工具读取 COCO instance annotations：

1. 原图直接拉伸到 320×320；
2. 每个非 crowd 实例 GT mask 最近邻缩放到 80×80（F4）；
3. 可做同步水平翻转；
4. 输入按 ImageNet mean/std 归一化；
5. Hungarian cost 只有 class + mask BCE + mask Dice，不使用 bbox；
6. matched queries 计算 class/BCE/Dice，其他 queries 监督为 no-object；
7. 尾部中间预测使用 auxiliary loss；
8. 推理将 masks 插值回原图，结合 query class score 输出标准 COCO segm AP。

它与 RF-DETR baseline 的关键差异不是单纯“少一个 decoder”：EoMT queries 在 ViT block 内直接与 patch self-attention，且没有检测框分支；当前小实验也没有获得 detector-pretrained queries/classifier 的同等信息。

## 3. INSID3 → COCO 实例分割的两个实现

公共实现位于 `src/rfdetr/models/insid3_coco.py`，统一入口为 `tools/evaluate_insid3_coco_instances.py`。两种模式都只从 COCO train 读取 support annotations；COCO val masks 仅由 COCOeval 使用。

### 3.1 纯 full-image K-shot

```text
每类别 K 个 train 图像+类别 union mask
        ↓ DINOv3/INSID3
K 个目标图概念 mask + cluster score map
        ↓ support voting
类别 foreground/confidence map
        ↓ 8 邻域 connected components
多个 instance masks + component confidence
        ↓ COCO RLE
standard COCO segm AP
```

示例：

```bash
PYTHONPATH=src python tools/evaluate_insid3_coco_instances.py \
  --mode full-image \
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST/medium \
  --output-dir output/insid3_coco_instances/full_image_5shot \
  --categories person car dog cat bicycle \
  --shots 5 --image-size 448 --svd-components 128 \
  --device cuda
```

K-shot 聚合使用二值 mask vote 决定前景，用相对 merge threshold 经温度缩放后的 cluster score 排序实例。该方案仍可能合并相邻同类实例；连通域只是把概念 mask 转为 COCO 实例格式的第一版明确基线。

### 3.2 RF-DETR detector-guided

```text
RF-DETR target image
  └─ category + bbox + detector score
              ↓ expanded ROI
同类别 K 个 train instance crops + masks
              ↓ DINOv3/INSID3 on crop
support vote + cluster confidence
              ↓ select center/best component
paste mask into original image
              ↓ score = detector score × mask quality
COCO RLE / standard segm AP
```

示例：

```bash
PYTHONPATH=src python tools/evaluate_insid3_coco_instances.py \
  --mode detector-guided \
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST/medium \
  --output-dir output/insid3_coco_instances/detector_guided_5shot \
  --categories person car dog cat bicycle \
  --shots 5 --image-size 448 --svd-components 128 \
  --detector-checkpoint output/coco_medium_seed43_iterref_scaleroute_p3p4p5_full_lrd20_exactsteps_e24/checkpoint_best_regular.pth \
  --detector-threshold 0.25 --max-detections 100 \
  --device cuda
```

这个完整系统不是 training-free，因为 detector 已监督训练；准确说法是 **detector-guided、training-free mask generation**。它不使用 RF-DETR SegmentationHead，bbox/class/实例拆分来自 detector，mask 来自 DINOv3 in-context correspondence。

## 4. 初始验证边界

- 新增公共逻辑覆盖 support voting、cluster score、4/8 邻域实例拆分、ROI 选择/粘贴和 COCO RLE；相关 INSID3 测试共 10 项通过。
- CPU、224 输入、单类单图真实 COCO smoke 中，full-image 路径产生 2 个实例并完整进入 COCOeval；detector-guided 路径从真实 checkpoint 重建 detector，产生 1 个 ROI mask 并完整进入 COCOeval。
- 单图 smoke 的 AP 为 0，只证明端到端协议可执行，不是质量结论。正式比较需要至少 quick/medium 多图、相同类别集合和 1/5-shot 消融。
