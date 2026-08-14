# SiameseDiff pretrained weights

This directory is the only location for externally downloaded initialization
and official model weights. Large binaries are intentionally excluded from Git.

Expected files:

```text
pretrained_weights/
├── control_sd15.ckpt          # SD-v1.5 with initialized ControlNet
└── merged_pytorch_model.pth   # optional official Polyps release
```

Do not store experiment checkpoints here. Locally trained diffusion, NAM, and
downstream checkpoints belong under `../checkpoints/`; see `../README.md`.
