# Pretrained SegDiff weights

Place external or migrated weights under one dataset directory:

```text
pretrained_weights/
├── polyps/Denoise_Unet-1001_model.pth
├── acdc/Denoise_Unet-1001_model.pth
└── synapse/Denoise_Unet-176_model.pth
```

Weights are not redistributed. See `../README.md` for the official source,
license, legacy source bridge, verified server-run selection, and load command.

The upstream Duke Breast MRI and CT Organ checkpoints are available from the
[official Google Drive folder](https://drive.google.com/drive/folders/1OaOGBLfpUFe_tmpvZGEe2Mv2gow32Y8u).
For an official checkpoint, create `<dataset>/unet/`, rename the weight file to
`diffusion_pytorch_model.safetensors`, and rename its architecture file to
`config.json`, exactly as specified by the upstream README.
