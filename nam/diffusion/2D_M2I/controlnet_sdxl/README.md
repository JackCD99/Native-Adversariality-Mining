# ControlNet-SDXL integration

This package provides the PASCAL VOC+SBD natural-image branch. A semantic label map is converted to
the standard VOC palette and conditions ControlNet attached to SDXL. The SDXL VAE, text encoders,
and U-Net remain frozen during ControlNet optimization.

NAM consumes the initial 4x64x64 denoising score for 512x512 synthesis. Its latent ResUNet predicts
diagonal Gaussian offsets and is optimized through a ten-step differentiable rollout using a frozen
segmentation anchor.

```bash
python scripts/train_diffusion_2d.py --config configs/controlnet_sdxl_voc.yaml
python scripts/train_downstream_2d.py --config configs/controlnet_sdxl_voc.yaml --phase real
python scripts/train_nam_2d.py --config configs/controlnet_sdxl_voc.yaml
python scripts/generate_2d.py --config configs/controlnet_sdxl_voc.yaml --method nam
python scripts/train_downstream_2d.py --config configs/controlnet_sdxl_voc.yaml --phase synthetic
python scripts/evaluate_2d.py --config configs/controlnet_sdxl_voc.yaml
```

The default synthesis budget is 8,422. Sampling uses deterministic DDIM-50 and records previews,
condition masks, noise shifts, TensorBoard events, JSONL metrics, and per-sample provenance.
