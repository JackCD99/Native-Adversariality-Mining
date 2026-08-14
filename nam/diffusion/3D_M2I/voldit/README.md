# VolDiT integration

The adapter targets [Cardio-AI/VolDiT](https://github.com/Cardio-AI/voldit) at commit
`76c7063a1d51884dfeb7cd51c63d4191b5358839`.

```bash
git clone https://github.com/Cardio-AI/voldit third_party/voldit
git -C third_party/voldit checkout 76c7063a1d51884dfeb7cd51c63d4191b5358839
hf download AICM-HD/voldit --local-dir pretrained_weights/voldit
```

Configure the VQ-GAN and unconditional DiT checkpoints in `configs/voldit_3d.yaml`. Task-specific
TGCA weights are selected by validation 2.5D FID.

```bash
python nam/diffusion/3D_M2I/voldit/pre_training.py
python nam/diffusion/3D_M2I/voldit/NAM_training.py
python nam/diffusion/3D_M2I/voldit/sampling.py
```

NAM uses a volumetric ResUNet that consumes BxCxDxHxW scores. Sampling uses deterministic DDIM-50.
Every run records TensorBoard scalars, center-slice previews, target/prediction/error panels, score and
noise distributions, JSONL metrics, checkpoints, and sample provenance.
