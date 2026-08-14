# Checkpoint layout

Training writes only generated artifacts beneath this directory:

```text
checkpoints/
|-- diffusion/<dataset>/       LoRA attention weights and training state
`-- nam/<dataset>/             NAM latest and periodic checkpoints
```

Checkpoint binaries are excluded from Git. Stable paths are configured in
`configs/sd15_lora_pneumoniamnist.yaml` and `configs/sd15_lora_isic.yaml`.
