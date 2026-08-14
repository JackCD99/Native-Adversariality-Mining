# SiameseDiff integration

The adapter uses [Siamese-Diffusion](https://github.com/Qiukunpeng/Siamese-Diffusion) as the
generator backend. Clone the upstream source into the configurable project directory:

```bash
git clone https://github.com/Qiukunpeng/Siamese-Diffusion.git third_party/Siamese-Diffusion
```

## Pretrained weights

Store external weights in `pretrained_weights/`; model binaries are excluded from Git. Build the
SD-v1.5 ControlNet initialization with the upstream command:

```bash
python third_party/Siamese-Diffusion/tool_add_control.py
```

Save the result as `pretrained_weights/control_sd15.ckpt`. The released polyp checkpoint can also
be downloaded from [SylarQ/Siamese-Diffusion](https://huggingface.co/SylarQ/Siamese-Diffusion):

```bash
hf download SylarQ/Siamese-Diffusion merged_pytorch_model.pth \
  --local-dir nam/diffusion/2D_M2I/siamesediff/pretrained_weights
```

## Run order

All entry points have usable parser defaults after dataset and checkpoint paths are configured:

```bash
python -m nam.diffusion.2D_M2I.siamesediff.pre_training
python scripts/train_downstream_2d.py --phase real
python -m nam.diffusion.2D_M2I.siamesediff.NAM_training
python -m nam.diffusion.2D_M2I.siamesediff.sampling --method base
python -m nam.diffusion.2D_M2I.siamesediff.sampling --method nam
python scripts/train_downstream_2d.py --phase synthetic
```

Training uses 256x256 images, 4x32x32 latent noise, AdamW, a ten-step differentiable DDIM path,
and a frozen generator/anchor pair. Full synthesis uses deterministic DDIM-50. TensorBoard events,
JSONL metrics, Base/NAM comparison panels, noise histograms, and PNG diagnostics are written to the
run directory.

## Checkpoints

```text
checkpoints/
|-- diffusion/<dataset>/{last.ckpt,best_fid.ckpt}
|-- nam/<dataset>/{nam_latest.pt,runs/}
`-- downstream/<dataset>/<model>/{best.pt,latest.pt}
```

`best_fid.ckpt` is selected on validation FID. `nam_latest.pt` contains the miner, optimizer, RNG
state, generator metadata, and exact generator/anchor checkpoint paths.
