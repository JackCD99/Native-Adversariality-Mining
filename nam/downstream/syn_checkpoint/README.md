# Synthetic-training checkpoints

This directory stores downstream models continued from a real-data baseline
with matched real and synthetic streams.

```text
syn_checkpoint/<dataset>/<generator>/<model>/
├── best.pt                 best validation checkpoint used for final testing
├── latest.pt               most recent checkpoint
├── epoch_0020.pt           20-epoch archival checkpoints
├── epoch_0040.pt
└── <experiment>-syn-<timestamp>/
    ├── config.json
    └── matching run-local checkpoints
```

The generator level prevents checkpoints from different synthesis methods from
overwriting one another. Reported generalization results are obtained by
loading `best.pt` and evaluating it once on the held-out test split.
