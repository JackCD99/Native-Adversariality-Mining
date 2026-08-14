# SegDiff integration

This package provides the conditional epsilon model, paired preprocessing, deterministic DDIM,
and NAM bridge for [segmentation-guided-diffusion](https://github.com/mazurowski-lab/segmentation-guided-diffusion).

The `official_diffusers` backend builds the upstream `UNet2DModel` layout. The `legacy_unet`
backend loads compatible checkpoints through `semi_diffseg.denoise_model.denoise_unet.UNetModel`.
Configure `diffusion.backend`, `diffusion.project_dir`, and `diffusion.checkpoint` explicitly.

```bash
python -m nam.diffusion.2D_M2I.segdiff.pre_training
python -m nam.diffusion.2D_M2I.segdiff.NAM_training
python -m nam.diffusion.2D_M2I.segdiff.sampling --method base
python -m nam.diffusion.2D_M2I.segdiff.sampling --method nam
```

Pre-training uses epsilon MSE and direct mask conditioning. `condition_dropout_probability` and
`cfg_scale` expose conditional/unconditional alternatives. NAM freezes SegDiff and the downstream
anchor, then optimizes the latent ResUNet through a ten-step differentiable rollout. Sampling uses
deterministic DDIM-50. Training and sampling write TensorBoard events, structured metrics, image-mask
panels, error maps, and noise distributions.

Checkpoints are stored under `checkpoints/diffusion/<dataset>/` and `checkpoints/nam/<dataset>/`.
`best` is selected by validation noise MSE; `last` is resumable. External weights belong under
`pretrained_weights/<dataset>/` and are excluded from Git.
