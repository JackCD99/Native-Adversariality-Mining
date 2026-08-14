# JoDiffusion integration

The adapter targets [00why00/JoDiffusion](https://github.com/00why00/JoDiffusion) at commit
`9fef37099c982e0fa512e84456e4d717d797b593`.

```bash
git clone https://github.com/00why00/JoDiffusion third_party/JoDiffusion
git -C third_party/JoDiffusion checkout 9fef37099c982e0fa512e84456e4d717d797b593
```

Download the `0why0/JoDiffusion` pipeline and the task-specific label VAE into
`pretrained_weights/jodiffusion/`. The label VAE input channels must equal
`ceil(log2(diffusion.num_classes))`; output channels must equal `diffusion.num_classes`.

```bash
python "nam/diffusion/2D_M&I/jodiffusion/pre_training.py"
python "nam/diffusion/2D_M&I/jodiffusion/NAM_training.py"
python "nam/diffusion/2D_M&I/jodiffusion/sampling.py"
```

Joint U-ViT epsilon training, paired image/label synthesis, and deterministic DDIM are retained. The
8-channel miner operates on the shared image-mask score. Runs include TensorBoard events, JSONL
metrics, paired previews, score/noise histograms, and validation-FID checkpoints.
