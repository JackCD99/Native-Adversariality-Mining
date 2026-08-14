# Real-data baseline checkpoints

This directory stores downstream models trained only on each dataset's real
training split. These checkpoints define the baseline used to measure the
generalization gain from synthetic training.

```text
real_checkpoint/<dataset>/<model>/
├── best.pt                 best validation checkpoint
├── latest.pt               most recent checkpoint
├── epoch_0020.pt           20-epoch archival checkpoints
├── epoch_0040.pt
└── <experiment>-real-<timestamp>/
    ├── config.json
    └── matching run-local checkpoints
```

Synthetic training must initialize from the corresponding `best.pt`. Test-set
evaluation must not be used for checkpoint selection.
