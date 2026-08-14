# DiffBoost experiment checkpoints

- `diffusion/<dataset>/{best.ckpt,last.ckpt}`: fine-tuned ControlNet/SD state.
- `nam/<dataset>/nam_latest.pt`: adversariality miner and provenance.
- `downstream/<dataset>/<model>/{best.pt,latest.pt}`: downstream states.

Checkpoint binaries are ignored by Git; training creates the directories.
