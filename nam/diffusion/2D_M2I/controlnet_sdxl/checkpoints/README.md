# ControlNet-SDXL checkpoints

The experiment configuration uses the following stable layout:

```text
checkpoints/controlnet_sdxl/diffusion/   trained semantic ControlNet
checkpoints/controlnet_sdxl/nam/         NAM miner checkpoints
```

`pre_training.py` writes a Diffusers-compatible ControlNet directory and its
optimizer state. `NAM_training.py` writes step checkpoints and
`nam_latest.pt`. Real-only baselines are written to
`nam/downstream/real_checkpoint/pascal_voc_sbd/<model>/`; synthetic-trained
models are written to
`nam/downstream/syn_checkpoint/pascal_voc_sbd/controlnet_sdxl/<model>/`.
