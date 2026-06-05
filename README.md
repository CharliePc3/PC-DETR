# RF-DETR-DINOv3

Experimental RF-DETR variant that replaces the original RF-DETR DINOv2 backbone with Meta DINOv3.

This repository keeps the RF-DETR/LW-DETR detection architecture, dataset pipeline, losses, transformer decoder, and export utilities, while routing the visual backbone through DINOv3.

## Current Integration

- Backbone: DINOv3 ViT via Meta's official DINOv3 implementation.
- Weight loading: local DINOv3 checkpoints through `pretrained_encoder` or `DINOV3_WEIGHTS_DIR`.
- Source loading: DINOv3 is imported as a Python package when available; otherwise set `DINOV3_REPO_DIR` to a local Meta DINOv3 checkout.
- Primary local smoke test entrypoint: `train_dinov3.py`.

## Local Smoke Test

```bash
cd /data/cpc/root/project/RF-DETR-DINOv3
export DINOV3_REPO_DIR=/data/cpc/root/project/DINOv3
export DINOV3_WEIGHTS_DIR=/data/cpc/root/storage/dinov3权重
python train_dinov3.py
```

For background execution:

```bash
nohup python train_dinov3.py > smoke_test.log 2>&1 &
tail -f smoke_test.log
```

## Notes

- This is an experiment branch, not a drop-in replacement for official RF-DETR checkpoints.
- RF-DETR full-model pretrained weights are not expected to match this DINOv3 backbone unless they were trained from this codebase.
- DINOv3 official weights are loaded into the backbone only. The detection head and projector are initialized by this project unless a compatible full RF-DETR-DINOv3 checkpoint is provided.

## Attribution

This project is derived from RF-DETR, which is built on LW-DETR and Deformable DETR. The DINOv3 backbone follows Meta's official DINOv3 implementation and checkpoint format.
