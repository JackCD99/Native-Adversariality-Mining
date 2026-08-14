# SDXL initialization

The ControlNet is initialized from
[`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
Accept the model license and either keep the Hugging Face identifier in
`configs/controlnet_sdxl_voc.yaml` or download a local Diffusers snapshot:

```bash
hf download stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir pretrained_weights/controlnet_sdxl/stable-diffusion-xl-base-1.0
```

The task-specific semantic ControlNet is trained from the SDXL U-Net weights;
no unrelated pretrained ControlNet is required.

References:

- ControlNet: https://github.com/lllyasviel/ControlNet
- SDXL: https://github.com/Stability-AI/generative-models
- Diffusers SDXL ControlNet API: https://github.com/huggingface/diffusers
