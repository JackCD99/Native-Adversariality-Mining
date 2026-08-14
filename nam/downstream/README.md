# Downstream model contract

The downstream package separates architecture definitions, real-data baseline training, and
real-plus-synthetic continuation. Each model directory contains:

- `model.py`: the network and model-specific configuration contract;
- `train_real.py`: the official-style real-data optimization policy;
- `train_syn.py`: synthetic continuation initialized from the validation-selected real checkpoint.

The shared trainer in `nam/downstream/training.py` owns device setup, automatic mixed precision,
optimization, validation, TensorBoard logging, preview generation, checkpoint serialization, and
resume semantics. Architecture packages provide only behavior that is specific to that model.

## Checkpoint namespaces

Real-only runs write to:

```text
nam/downstream/real_checkpoint/<dataset>/<model>/
```

Synthetic continuation writes to:

```text
nam/downstream/syn_checkpoint/<dataset>/<generator>/<model>/
```

Each run maintains `best.pt` and `latest.pt`, and writes `epoch_XXXX.pt` every 20 epochs by default.
`best.pt` is selected exclusively on the validation split. NAM uses the real-data `best.pt` as a
frozen adversariality anchor; final evaluation can independently select the real or synthetic
checkpoint namespace.

## Entry points

```bash
python scripts/train_downstream_2d.py --config configs/table1_2d.yaml --phase real
python scripts/train_downstream_2d.py --config configs/table1_2d.yaml --phase synthetic
python scripts/train_downstream_3d.py --config configs/table1_3d.yaml --phase real
python scripts/train_downstream_3d.py --config configs/table1_3d.yaml --phase synthetic
```

Use `--dry-run --print-config` to validate the selected dataset, architecture, checkpoint paths, and
resolved training policy before allocating a GPU run.

## Upstream implementations

| Package | Primary source |
|---|---|
| nnU-Net | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| Swin-Unet | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) |
| SwinUNETR | [MONAI research contributions](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) |
| SAMed | [hitachinsk/SAMed](https://github.com/hitachinsk/SAMed) |
| DeepLabV3 | [Torchvision segmentation models](https://pytorch.org/vision/stable/models/deeplabv3.html) |
| Mask2Former | [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) |
| ResNet-50 | [Torchvision classification models](https://pytorch.org/vision/stable/models/resnet.html) |
| ViT-S/16 | [timm](https://github.com/huggingface/pytorch-image-models) |

Consult the upstream license and citation requirements before distributing derived checkpoints.
