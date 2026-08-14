# SegDiff experiment checkpoints

- `diffusion/<dataset>/{best,last}/`: Diffusers UNet, scheduler, training state.
- `nam/<dataset>/nam_latest.pt`: NAM miner and provenance.
- `downstream/<dataset>/<model>/{best.pt,latest.pt}`: real/synthetic model states.

Large checkpoint files are ignored by Git. Empty directories are created by the
training entry points from `configs/segdiff_2d.yaml`.
