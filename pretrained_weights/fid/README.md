# FID feature encoders

The evaluation protocol uses modality-specific feature spaces rather than a
single natural-image encoder for every dataset.

| Modality | Encoder | Weight source |
|---|---|---|
| RGB | ImageNet Inception-v3 | Torchvision `Inception_V3_Weights.IMAGENET1K_V1` |
| Chest X-ray | DenseNet-121 (`densenet121-res224-all`) | [TorchXRayVision](https://github.com/mlmed/torchxrayvision) |
| MRI / CT / CTA | RadImageNet ResNet-50 | [RadImageNet model repository](https://github.com/BMEII-AI/RadImageNet) |

Place the RadImageNet ResNet-50 state dictionary at:

```text
pretrained_weights/fid/radimagenet_resnet50.pt
```

The loader accepts raw state dictionaries and common `state_dict`, `model`,
`model_state_dict`, or `net` wrappers. Classification-head tensors are ignored;
FID uses the pooled penultimate representation. Do not commit downloaded model
weights. Their use and redistribution remain subject to the upstream license.

TorchXRayVision downloads its registered DenseNet-121 weights into the standard
PyTorch cache on first use. Install the optional dependency with
`pip install torchxrayvision`; the repository does not redistribute the model.

For three-dimensional experiments, the same frozen RadImageNet encoder is
applied to axial, coronal, and sagittal slices, and the three FID values are
averaged to obtain the reported 2.5D FID.
