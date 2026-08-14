# Pretrained weights

The default configuration loads
[`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
through Diffusers. Authenticate with `huggingface-cli login` when required by
the upstream model license. The base model is cached outside the repository.

Run `pre_training.py` to create the dataset-specific LoRA attention weights.
Do not commit downloaded backbones, LoRA checkpoints, or NAM checkpoints.
