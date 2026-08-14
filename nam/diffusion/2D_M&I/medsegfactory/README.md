# MedSegFactory integration

The adapter targets [jwmao1/MedSegFactory](https://github.com/jwmao1/MedSegFactory) at commit
`b227c6b5f0ff6b02d6046a1cdf57fc47cb74ae96`.

```bash
git clone https://github.com/jwmao1/MedSegFactory third_party/MedSegFactory
git -C third_party/MedSegFactory checkout b227c6b5f0ff6b02d6046a1cdf57fc47cb74ae96
```

Place Stable Diffusion v1.5 under `pretrained_weights/medsegfactory/stable-diffusion-v1-5`.
The published JCA U-Net checkpoint (`JohnWeck/StableDiffusion/checkpoint-300.pth`) belongs under
`pretrained_weights/medsegfactory/`.

```bash
python "nam/diffusion/2D_M&I/medsegfactory/pre_training.py"
python "nam/diffusion/2D_M&I/medsegfactory/NAM_training.py"
python "nam/diffusion/2D_M&I/medsegfactory/sampling.py"
```

The JCA U-Net, paired epsilon objective, independent image/mask noise, and CFG remain configurable.
Two independent 8-to-4 miners consume the joint initial score. Runs persist TensorBoard events,
JSONL metrics, Base/NAM panels, error maps, noise distributions, and validation-FID checkpoints.
