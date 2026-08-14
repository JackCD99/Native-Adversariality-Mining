# Pipeline dependency tree

## Stable interfaces

```text
dataset factory
└── NAMBatch(image, target, condition, sample_id, metadata)
    ├── generator pre-training → generator checkpoint
    ├── real downstream training → real checkpoint
    └── NAM training
        ├── frozen generator
        ├── frozen nnU-Net anchor
        └── miner checkpoint
            └── Base/NAM sampling → generated tensor pairs
                └── synthetic downstream continuation → synthetic checkpoint
                    ├── DSC/ASD evaluation
                    ├── fixed-budget adversariality CSV/JSON
                    └── FID and visualization outputs
```

The public dispatch functions are `run_diffusion_pretraining`,
`run_nam_training`, `run_sampling`, and `run_downstream_training`. The dispatch
registry routes each generator to its own `pre_training.py`, `NAM_training.py`,
and `sampling.py`; joint image-mask and 3D methods therefore retain their
specialized miner layouts.

## Table I branches

| Dataset | Dimension | Generators | Downstream models | Dataset factory |
|---|---:|---|---|---|
| ACDC | 2D | SegDiff, DiffBoost, JoDiffusion | nnU-Net, Swin-Unet, SAMed | `nam.data.acdc.dataset:build_dataset` |
| Synapse | 2D | FairDiff, DiffBoost, MedSegFactory | nnU-Net, Swin-Unet, SAMed | `nam.data.synapse.dataset:build_dataset` |
| Polyps | 2D | SegDiff, DiffBoost, SiameseDiff, MedSegFactory | nnU-Net, Swin-Unet, SAMed | `nam.data.polyps.dataset:build_dataset` |
| LA | 3D | VolDiT, MAISI | nnU-Net, SwinUNETR, SAMed | `nam.data.la.dataset:build_dataset` |
| ImageCAS | 3D | VolDiT, MAISI | nnU-Net, SwinUNETR, SAMed | `nam.data.imagecas.dataset:build_dataset` |

`configs/table1_matrix.yaml` records the full matrix and maps each generator to
its base configuration. The method configurations inherit shared downstream and
evaluation settings from `table1_2d.yaml` or `table1_3d.yaml`.

## Additional dataset roles

- MMWHS exposes paired MR/CT slice interfaces for cross-modality experiments.
- EndoScene, ColonDB, and ETIS are held-out polyp segmentation test sets and
  enter the tree at the final downstream-evaluation stage.
- PneumoniaMNIST and ISIC use the LoRA-tuned SD-v1.5 adapter and the ResNet-50
  or ViT-S/16 classification pipelines recorded in `configs/transfer_matrix.yaml`.

## Medical-classification transfer branch

```text
PneumoniaMNIST-224 or ISIC
`-- class-aware prompt + label
    `-- LoRA-tuned SD-v1.5
        |-- frozen ResNet-50 or ViT-S/16 anchor
        `-- latent NAM miner (4 x H/8 x W/8)
            `-- matched-budget NAM images and class labels
                `-- real + synthetic classifier continuation
                    `-- held-out accuracy / balanced accuracy / specificity
```

## Natural-image transfer branch

```text
PASCAL VOC 2012 + SBD (8422 / 1203 / 2406, 21 classes, 512 x 512)
└── semantic masks + `a photo of [VOC object]`
    └── ControlNet tuned on SDXL
        ├── frozen DeepLabV3-ResNet50 anchor
        └── latent NAM miner (4 x 64 x 64)
            └── 8422 NAM samples with DDIM-50
                ├── DeepLabV3-ResNet50 continuation → mIoU
                └── Mask2Former-ResNet50 continuation → mIoU
```

The full branch is configured by `configs/controlnet_sdxl_voc.yaml`; its matrix
record is `configs/transfer_matrix.yaml`.

These roles are deliberately separated so that an evaluation-only dataset is
never used for generator, miner, or checkpoint training.

## Checkpoint and sample locations

Each generator package contains `pretrained_weights/README.md` and
`checkpoints/README.md`. Configuration fields resolve all paths:

```text
pretrained_weights/<method>/        upstream initialization
checkpoints/<method>/diffusion/     trained generator
checkpoints/<method>/nam/           NAM miner
nam/downstream/real_checkpoint/<dataset>/<model>/
                                    real-only baselines and 20-epoch snapshots
nam/downstream/syn_checkpoint/<dataset>/<generator>/<model>/
                                    synthetic-trained checkpoints for testing
outputs/synthetic/<method>/         generated tensor pairs and metadata
outputs/evaluation/                 metrics, CSV files, JSON summaries, figures
```

Synthetic dataset roots must point to the exact `<experiment>-nam` directory
created by the method sampler. No source file contains a machine-specific data,
checkpoint, or output path.
