# Stable Diffusion v1.5 with LoRA

This package implements the class-conditional synthesis branch used for
PneumoniaMNIST-224 and ISIC. It follows the public Diffusers text-to-image LoRA
recipe while exposing the same score, NAM, deterministic DDIM, and fixed-budget
interfaces as the segmentation generators.

The pretrained backbone is
[`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
The adaptation is restricted to low-rank attention processors; the VAE, text
encoder, and base U-Net weights remain frozen. Downloaded backbone weights stay
in the Hugging Face cache. Dataset-specific LoRA and NAM weights are written to
`checkpoints/diffusion/<dataset>/` and `checkpoints/nam/<dataset>/`.

```bash
python scripts/train_diffusion_2d.py --config configs/sd15_lora_pneumoniamnist.yaml
python scripts/train_downstream_2d.py --config configs/sd15_lora_pneumoniamnist.yaml --phase real
python scripts/train_nam_2d.py --config configs/sd15_lora_pneumoniamnist.yaml
python scripts/generate_2d.py --config configs/sd15_lora_pneumoniamnist.yaml --method nam
python scripts/train_downstream_2d.py --config configs/sd15_lora_pneumoniamnist.yaml --phase synthetic
python scripts/evaluate_classification.py --config configs/sd15_lora_pneumoniamnist.yaml
```
