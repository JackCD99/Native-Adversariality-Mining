# 3D mask-to-image implementation

This directory contains the two volumetric Table-I generators:

| Method | Official backbone | Latent noise | NAM miner |
| --- | --- | --- | --- |
| VolDiT | VQ-GAN + DiT3D + TGCA | `8x24x24x12` | 25.190M-parameter 3D ResUNet |
| MAISI | AutoencoderKlMaisi + DiffusionModelUNetMaisi + ControlNetMaisi | `4x48x48x24` | 25.172M-parameter 3D ResUNet |

Both pipelines align volumes to `192x192x96`, validate generator checkpoints using axial/coronal/sagittal 2.5D FID every 500 iterations, and synthesize with deterministic DDIM-50. Each method contains `model.py`, `pre_training.py`, `NAM_training.py`, `sampling.py`, method-specific `utils/`, public-weight instructions, and generated-checkpoint documentation.

The miner is not slice-wise: `nam.models.ResUNet3DMiner` accepts only `BxCxDxHxW`, uses `Conv3d` throughout, has four residual resolution blocks, self-attention at the bottleneck, and zero-initialized mean/variance heads. Latent diffusion models downsample only the first two encoder transitions as specified in the supplement.

Use the method-level `README.md` and YAML file in `configs/` for the complete run order. To switch the default LA setup to ImageCAS, replace the dataset/anchor paths and prompt, and set the sampling budget from 70 to 700.
