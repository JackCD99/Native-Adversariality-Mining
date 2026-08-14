# DiffBoost integration

- Source: [NUBagciLab/DiffBoost](https://github.com/NUBagciLab/DiffBoost)
- Pinned commit: `32da5619c9ff03b9f33d521b83254f7a60236e15`

```bash
git clone --recurse-submodules https://github.com/NUBagciLab/DiffBoost third_party/DiffBoost
git -C third_party/DiffBoost checkout 32da5619c9ff03b9f33d521b83254f7a60236e15
```

The backend retains SD-v1.5 latent diffusion, ControlNet guidance, CLIP text conditioning, and DDIM.
Use `condition_mode: mask` for mask-to-image synthesis or `condition_mode: edge` for the upstream
edge-guided setting. When no edge is supplied, the adapter deterministically derives a boundary map.

The RadImageNet initialization is not redistributed. Place a licensed checkpoint at
`pretrained_weights/radimagenet/model.ckpt`. Target-dataset fine-tuning writes validation-selected
`best.ckpt` and resumable `last.ckpt` under `checkpoints/diffusion/<dataset>/`.

```bash
python -m nam.diffusion.2D_M2I.diffboost.pre_training
python -m nam.diffusion.2D_M2I.diffboost.NAM_training
python -m nam.diffusion.2D_M2I.diffboost.sampling --method base
python -m nam.diffusion.2D_M2I.diffboost.sampling --method nam
```

The configured pipeline exposes the augmentation range, edge/mask control, CFG, DDIM steps, and text
weights. Training and sampling persist TensorBoard events, JSONL metrics, generated previews,
segmentation comparisons, and score/noise distributions.
