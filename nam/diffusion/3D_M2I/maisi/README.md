# MAISI integration

This adapter uses the DDPM workflow from the [MONAI MAISI tutorial](https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi)
at commit `81dcf0f63e2a3a064e882ef0f26d5889b7bedf53`.

```bash
git clone https://github.com/Project-MONAI/tutorials third_party/monai-tutorials
git -C third_party/monai-tutorials checkout 81dcf0f63e2a3a064e882ef0f26d5889b7bedf53
pip install -e ".[volumetric-diffusion]"
```

Store the downloaded autoencoder and DDPM U-Net in `pretrained_weights/maisi/`. MAISI code is
Apache-2.0; the released weights use NVIDIA OneWay Noncommercial License v1.

```bash
python nam/diffusion/3D_M2I/maisi/pre_training.py
python nam/diffusion/3D_M2I/maisi/NAM_training.py
python nam/diffusion/3D_M2I/maisi/sampling.py
```

ControlNet training uses the L1 epsilon objective, AdamW, and polynomial decay. NAM uses a volumetric
ResUNet on five-dimensional score tensors. TensorBoard includes center-slice images, labels,
predictions, difference maps, score/noise histograms, optimizer curves, and 2.5D validation FID.
