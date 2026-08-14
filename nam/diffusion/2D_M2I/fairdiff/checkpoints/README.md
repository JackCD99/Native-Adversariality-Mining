# FairDiff checkpoint layout

Training scripts create the following stable paths automatically:

```text
checkpoints/
├── diffusion/synapse/
│   ├── best.ckpt
│   ├── last.ckpt
│   └── runs/<experiment-timestamp>/
├── nam/synapse/
│   ├── nam_latest.pt
│   └── <experiment-timestamp>/checkpoints/
└── downstream/synapse/
    ├── nnunet/best.pt
    ├── swinunet/best.pt
    └── samed/best.pt
```

`best.ckpt` is selected by the lowest validation FID measured every 500
iterations, matching the paper protocol. `last.ckpt` is the latest resumable
state. Large generated artifacts and licensed model weights should not be
committed to Git.
