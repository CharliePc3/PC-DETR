# ICLR 2027 Paper Experiment Contract

状态：冻结。本文定义 PC-DETR 的 ICLR 2027 论文实验线。包含本文的提交即 frozen
commit；worker 必须记录并使用完整的 40 位 SHA，不能用浮动分支名代替。

## 1. 冻结范围

论文主方法固定为 `SDSR-v40 P4 learnable`，公平基线固定为原始 P3/P4/P5
`MultiScaleProjector`。除 projector 外，二者必须使用同一 recipe、seed、数据、初始化、
训练轮数和评测协议。

冻结依据是 `handoff/ADAPTER_REFINEMENT_HANDOFF_CN.md` 第 25.7 节和第 26 节。唯一
机器可读配置是 `paper/ICLR27_RECIPE.json`；唯一公共启动入口是
`paper/run_iclr27_experiment.py`。launcher 不提供科学超参数覆盖入口。

本实验线不包含 SDSR-v41-v46 的后续候选、任何 SDSR-v47、新结构搜索、INSID3、实例或
语义分割、UniRefiner/LazyStrike、online refinement、feature adapter、register border、
photometric 邻近点或旧 Full 续训。

## 2. 不变量

以下参数在 analysis 和 system track 中完全相同：

- DINOv3-S intermediate blocks `2/5/8/11`，输出 `P3/P4/P5`；
- SDSR 使用 `sdsr_v40_p4_learnable`，rank/detail=`64/32`，cross-scale=`none`，关闭
  local reassembly 和 directional guide，启用 phase downsample；
- decoder 4 层，query/select=`300/300`，Group DETR=6，iterative box refinement，
  shared bbox head，legacy scale routing；
- decoder sampling points 为 P3/P4/P5=`2/3/1`；`dec_n_points=2` 仅作为旧接口 fallback；
- CDN 开启，DN25，label/box noise=`0.5/0.6`，loss coef=`0.5`，总 query budget=300；
- Dense O2O enhanced：从 epoch 2 开始，Mosaic/MixUp/CopyBlend=`0.5/0/0.5`，
  CopyBlend objects=1、area threshold=100、expand ratio=`0.1/0.25`；
- resolution=640，随机 multi-scale 和 expanded scales 全程开启，
  `multi_scale_stop_epoch=-1`；
- register border=0，augmentation preset=`default`，online refinement=`none`，
  feature adapter=`none`，segmentation head 关闭；
- total batch=16（batch 8、gradient accumulation 2、world size 1）；
- detector/encoder LR=`2.5e-4/1.25e-4`，weight decay=`1e-4`，ViT/component layer
  decay=`0.8/0.7`；
- 主论文运行开启 EMA (`decay=0.993, tau=100`) 并同时报告 raw/EMA。仅效率专项允许
  `--no-use-ema`，且必须单独标成 efficiency run，不能把它与主表 EMA 指标混写。

禁止根据 smoke AP、单 seed 波动或 worker 空闲资源改动以上参数。

## 3. 两条批准的实验轨道

`analysis` 用于论文消融与 paired seed：COCO Medium、24e、`lr_drop=20`、Dense image
stop=12、CopyBlend stop=21。

`system` 用于最终 Full COCO：30e、`lr_drop=25`、Dense image stop=15、CopyBlend
stop=26。该 1.25 倍阶段映射来自 handoff 26.3，不是新的搜索结果。启动 Full 必须显式传入
`--confirm-full-coco`。

允许变化的实验轴只有：

1. `--system sdsr_v40|msp`；
2. `--track analysis|system`；
3. paper 预注册的 seed 与对应 detector initialization seed（默认约定为 `seed+1000`）；
4. worker 数、GPU 设备和输出路径等非科学运行参数；
5. 仅效率专项可关闭 EMA。

任何其他变化必须使用新的 recipe ID，不能写入本 freeze 的 `RESULTS.json`。

## 4. Worker 前置检查

每个 worker 必须使用独立 clean worktree，并执行：

```bash
git rev-parse HEAD
git status --porcelain --untracked-files=normal
```

第一条必须等于负责人发布的 frozen commit SHA；第二条必须为空。launcher 会再次拒绝 dirty
worktree。必须记录 COCO 数据版本、Medium 子集构造方式、DINOv3 外部仓库提交和预训练
权重校验和；这些外部输入尚不能由本仓库 SHA 单独恢复。

不得复用已有输出目录，不得从其他 recipe 的 checkpoint resume。SDSR/MSP paired run
必须分开输出目录，并使用相同 seed/detector-init-seed。不可通过复制旧日志构造结果。

## 5. Exact command templates

Analysis（示例 seed43 SDSR；MSP 只将 `--system` 改为 `msp`）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/cpc/.conda/envs/rfdetr-dinov3/bin/python \
  paper/run_iclr27_experiment.py \
  --track analysis \
  --system sdsr_v40 \
  --data-root /data/cpc/root/dataset/COCO_RFDETR_TEST \
  --output-dir output/iclr27_analysis_sdsr_v40_seed43 \
  --seed 43 \
  --detector-init-seed 1043 \
  --num-workers 8
```

System（Full COCO；创建命令不代表授权立即启动）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/cpc/.conda/envs/rfdetr-dinov3/bin/python \
  paper/run_iclr27_experiment.py \
  --track system \
  --system sdsr_v40 \
  --data-root /data/cpc/root/dataset/COCO \
  --output-dir output/iclr27_system_sdsr_v40_seed43 \
  --seed 43 \
  --detector-init-seed 1043 \
  --num-workers 8 \
  --confirm-full-coco
```

在真正运行前可追加 `--dry-run` 只打印完全展开的底层命令。`--dry-run` 不训练。

## 6. 结果合同

每个输出目录必须保留 launcher 生成的 `RUN_MANIFEST.json`、训练生成的 `log.txt` 和
checkpoint。训练结束后运行：

```bash
/home/cpc/.conda/envs/rfdetr-dinov3/bin/python paper/analyze_results.py \
  --output-dir output/iclr27_analysis_sdsr_v40_seed43 \
  --train-log iclr27_analysis_sdsr_v40_seed43.log
```

工具生成大写 `RESULTS.json`，其规范为 `paper/RESULTS.schema.json`。所有 COCO AP 均按
0-100 points 保存，不能混入 0-1 比例。必须报告 raw；启用 EMA 的完整运行也必须报告
EMA。统一记录 best/final/last-5、参数量、训练时间、可获得的峰值显存、artifact 路径、
命令、环境和 frozen SHA。latency/GFLOPs 没有按统一协议测量时写 `null`，不能由参数量
推断。

## 7. 论文表述边界

- 当前 Medium H recipe 是 seed43 选择结果，尚不是多 seed 全局最优证明；
- 已有同配置 SDSR 单次波动约 0.13 AP，SDSR 对 MSP 的 +0.08 AP 不能写成统计稳定胜出；
- 可以表述 SDSR trainable parameters 比 MSP 少约 11.9%，但现有 eager 训练数据不支持
  “更快”或“更省显存”；
- bounded CDN 的主要价值是约束训练 query/显存，不应把它单独宣称为已证明 AP 增益；
- `multi_scale_stop_epoch` 修复必须保留，但冻结 recipe 使用 `-1`，即全程随机多尺度；
- register border 必须为 0；历史 Reg1 结果只能作为独立 ablation；
- smoke AP 只检查可运行性，绝不用于选择或更改 recipe。

若任一不变量被违反，该 run 必须标记为 contract violation，不能进入论文主表或 paired
统计。
