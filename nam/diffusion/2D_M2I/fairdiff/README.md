# FairDiff integration

- Source: [wenyi-li/FairDiff](https://github.com/wenyi-li/FairDiff)
- Pinned commit: `3a0a67ad1f1a3be719b6d529178eeb217a2868a0`
- Reference: Li et al., *FairDiff: Fair Segmentation with Point-Image Diffusion*, MICCAI 2024.

```bash
git clone https://github.com/wenyi-li/FairDiff third_party/FairDiff
git -C third_party/FairDiff checkout 3a0a67ad1f1a3be719b6d529178eeb217a2868a0
```

The adapter retains MaskImageGen ControlLDM, DDIM, the upstream YAML model, prompts, and optimizer
policy. Categorical labels are converted with a fixed palette; `mask_encoding: scalar` remains
available for legacy grayscale conditions. Fixed-mask synthesis does not invoke PointMaskGen.

```bash
python -m nam.diffusion.2D_M2I.fairdiff.pre_training
python -m nam.diffusion.2D_M2I.fairdiff.NAM_training
python -m nam.diffusion.2D_M2I.fairdiff.sampling --method base
python -m nam.diffusion.2D_M2I.fairdiff.sampling --method nam
```

The default setup uses 256x256 pairs, validation-FID checkpoint selection, deterministic DDIM-50,
and a latent ResUNet miner. Run directories include checkpoints, TensorBoard events, JSONL metrics,
PNG previews, segmentation error maps, and noise/score histograms. See `pretrained_weights/README.md`
and `checkpoints/README.md` for artifact placement.
