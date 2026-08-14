<h1 align="center">Native Adversariality Mining (NAM)</h1>

<p align="center">
  <b>Mining Native Adversariality in Diffusion Models for Medical Generalization</b>
</p>

<p align="center">
  Official PyTorch implementation of NAM and the experiments in the TPAMI extended manuscript.
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Diffusion-Based_Native_Adversarial_Synthesis_for_Enhanced_Medical_Segmentation_Generalization_CVPR_2026_paper.pdf">
    <img src="https://img.shields.io/badge/CVPR%202026-Highlight-ff4d4f" alt="CVPR 2026 Highlight">
  </a>
  <a href="#citation">
    <img src="https://img.shields.io/badge/TPAMI-Extended%20Version-0054a6" alt="TPAMI extended version">
  </a>
  <a href="pyproject.toml">
    <img src="https://img.shields.io/badge/Python-%E2%89%A53.10-3776ab?logo=python&logoColor=white" alt="Python >=3.10">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.1-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch >=2.1">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="Apache-2.0">
  </a>
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Diffusion-Based_Native_Adversarial_Synthesis_for_Enhanced_Medical_Segmentation_Generalization_CVPR_2026_paper.pdf">CVPR paper</a>
  ·
  <a href="https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhang_Diffusion-Based_Native_Adversarial_CVPR_2026_supplemental.pdf">CVPR supplement</a>
  ·
  <a href="docs/TABLE1_PROTOCOL.md">Table I protocol</a>
  ·
  <a href="docs/EVALUATION_VISUALIZATION.md">Evaluation & visualization</a>
</p>

---

## Overview

Diffusion models can generate realistic medical images, but realistic synthetic data are not necessarily the samples that most improve a downstream model. **Native Adversariality Mining (NAM)** focuses on synthetic samples that are difficult for a downstream model while remaining supported by the base diffusion model.

Instead of modifying the diffusion model or adding adversarial guidance along the denoising trajectory, NAM learns a lightweight miner that changes the initial noise distribution. The diffusion generator and downstream anchor remain frozen during NAM optimization. Once trained, the miner is used to select seeds that are more likely to produce informative, high-adversariality samples for synthetic-data augmentation.

This repository provides code for:

- 2D and 3D NAM training;
- mask-to-image (M2I), image-and-mask (M&I), and text-to-image synthesis;
- medical segmentation, medical classification, and natural-image segmentation;
- Base-vs-NAM fixed-budget synthesis;
- downstream training with real and synthetic data;
- adversariality, FID, DSC/ASD/mIoU/accuracy evaluation;
- cross-model adversariality analysis and publication visualizations;
- mitigation strategies for high-adversariality samples that may expose defective generator modes.

<p align="center">
  <img src="assets/nam_overview.png" width="900" alt="Overview of Native Adversariality Mining">
</p>

### Core idea

Standard diffusion augmentation samples initial noise randomly:

```text
random noise -> diffusion model -> synthetic sample
```

NAM learns which regions of the noise space are more likely to produce useful hard samples:

```text
random probe noise
        |
        v
initial denoising response
        |
        v
      NAM miner
        |
        v
reselected initial noise
        |
        v
frozen diffusion model
        |
        v
native-adversarial synthetic sample
```

The method is designed to increase synthetic adversariality without the large distribution drift often introduced by adversarial guidance.

<p align="center">
  <img src="assets/native_vs_artificial_adversariality.png" width="760" alt="Native versus artificial adversariality">
</p>

---

## What can be reproduced

The repository covers the experimental settings used in the extended study:

| Setting | Included |
|---|---|
| In-domain medical segmentation | ACDC, Synapse, Polyps |
| 3D medical segmentation | LA, ImageCAS |
| Cross-center evaluation | EndoScene/CVC-300, CVC-ColonDB, ETIS |
| Cross-modality evaluation | MMWHS MRI ↔ CT |
| Medical classification | PneumoniaMNIST-224, ISIC |
| Natural-image segmentation | PASCAL VOC 2012 + SBD |
| Diffusion paradigms | 2D M2I, 2D M&I, 3D M2I, text-to-image |
| Downstream architectures | CNN, Transformer, SAM-based, classification and natural-image models |
| Analysis | adversariality, FID, t-SNE, cross-model consistency |
| Mitigation | HAT, QSF, LSRS, ASG |

For the main Table I experiments, `configs/table1_matrix.yaml` records the dataset–generator–downstream matrix and `scripts/list_table1.py` prints the corresponding experiment cells.

---

## Repository structure

```text
Native-Adversariality-Mining/
├── assets/                     # figures used in this README
├── configs/                    # experiment configurations
├── docs/                       # reproduction and evaluation notes
├── nam/
│   ├── data/                   # dataset loaders, manifests, preparation guides
│   ├── diffusion/
│   │   ├── 2D_M2I/            # 2D mask-to-image diffusion adapters
│   │   ├── 2D_M&I/            # joint image-mask diffusion adapters
│   │   ├── 2D_T2I/            # text-to-image branch
│   │   └── 3D_M2I/            # volumetric mask-to-image diffusion adapters
│   ├── downstream/             # downstream architectures and training
│   ├── engine/                 # shared training/sampling entry logic
│   ├── evaluation/             # metrics and visualization utilities
│   ├── mitigation/             # defective-mode mitigation methods
│   └── utils/                  # configuration, logging, I/O and utilities
├── pretrained_weights/         # documentation for external evaluation weights
├── scripts/                    # executable experiment entry points
├── pyproject.toml
├── requirements.txt
└── README.md
```

Most users only need to edit a YAML configuration, prepare the corresponding dataset and model weights, and then use the scripts in `scripts/`.

---

## Installation

### Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer
- CUDA-capable GPU for diffusion/NAM training
- Linux is recommended for full experiments
- Windows can be used for configuration inspection and many Python components, but several upstream medical/diffusion repositories are primarily tested on Linux

Clone the repository:

```bash
git clone https://github.com/JackCD99/Native-Adversariality-Mining.git
cd Native-Adversariality-Mining
```

Create an environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install a PyTorch build compatible with your CUDA environment first, then install NAM:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Optional dependency groups are defined in `pyproject.toml`. Install only the components required for your experiment:

```bash
# medical imaging
pip install -e ".[medical]"

# diffusion pipelines
pip install -e ".[diffusion]"

# natural-image transfer
pip install -e ".[natural]"

# classification
pip install -e ".[classification]"

# 3D diffusion
pip install -e ".[volumetric-diffusion]"

# plotting / t-SNE
pip install -e ".[visualization]"

# X-ray-specific dependencies
pip install -e ".[xray]"

# VQA-based mitigation
pip install -e ".[mitigation-vqa]"
```

A convenient environment for most 2D medical experiments is:

```bash
pip install -e ".[medical,diffusion,visualization]"
```

---

## Configuration system

Experiments are configured with YAML files in `configs/`.

Common command-line options are available for the main scripts:

```text
--config PATH
--set KEY=VALUE [KEY=VALUE ...]
--dry-run
--print-config
--log-level {DEBUG,INFO,WARNING,ERROR}
```

Before launching a long GPU experiment, it is useful to validate the configuration:

```bash
python scripts/train_nam_2d.py \
  --config configs/table1_2d.yaml \
  --dry-run \
  --print-config
```

Configuration values can be overridden without editing the YAML:

```bash
python scripts/train_nam_2d.py \
  --config configs/table1_2d.yaml \
  --set runtime.seed=0 training.reward=lcbce
```

### Main configuration files

| Config | Typical use |
|---|---|
| `configs/table1_2d.yaml` | default 2D Table I reproduction example |
| `configs/table1_3d.yaml` | 3D Table I experiments |
| `configs/table1_matrix.yaml` | full Table I experiment matrix |
| `configs/segdiff_2d.yaml` | SegDiff |
| `configs/diffboost_2d.yaml` | DiffBoost |
| `configs/fairdiff_2d.yaml` | FairDiff |
| `configs/jodiffusion_2d.yaml` | JoDiffusion |
| `configs/medsegfactory_2d.yaml` | MedSegFactory |
| `configs/maisi_3d.yaml` | MAISI |
| `configs/controlnet_sdxl_voc.yaml` | PASCAL VOC + SBD transfer |
| `configs/sd15_lora_pneumoniamnist.yaml` | PneumoniaMNIST classification |
| `configs/sd15_lora_isic.yaml` | ISIC classification |
| `configs/mitigation.yaml` | mitigation experiments |

The default `table1_2d.yaml` configuration corresponds to a **Polyps + SiameseDiff + nnU-Net** experiment. It is a useful starting point for checking the full pipeline.

---

## Recommended first reproduction

For a first run, we recommend reproducing the Polyps/SiameseDiff/nnU-Net setting before moving to the full experiment matrix.

### 1. Inspect the experiment

```bash
python scripts/list_table1.py
```

Then inspect the resolved default configuration:

```bash
python scripts/train_nam_2d.py \
  --config configs/table1_2d.yaml \
  --dry-run \
  --print-config
```

### 2. Prepare the Polyps dataset

Follow:

```text
nam/data/polyps/README.md
```

The dataset package contains the expected split manifests. Raw images and masks are not redistributed in this repository.

The default Table-I split contains:

- 1,128 training image-mask pairs
- 161 validation pairs
- 323 test pairs

Place processed files under the dataset package's ignored `data/` directory using the relative paths expected by the manifests.

### 3. Prepare SiameseDiff

Clone the upstream implementation:

```bash
git clone https://github.com/Qiukunpeng/Siamese-Diffusion.git third_party/Siamese-Diffusion
```

See:

```text
nam/diffusion/2D_M2I/siamesediff/README.md
```

for the expected pretrained weights and checkpoint locations.

The default Table-I configuration expects a SiameseDiff diffusion checkpoint at:

```text
nam/diffusion/2D_M2I/siamesediff/checkpoints/diffusion/polyps/best_fid.ckpt
```

If you reproduce the diffusion model yourself:

```bash
python scripts/train_diffusion_2d.py \
  --config configs/table1_2d.yaml
```

### 4. Train the real-data downstream baseline

NAM uses a downstream model trained on real data as its adversariality anchor.

```bash
python scripts/train_downstream_2d.py \
  --config configs/table1_2d.yaml \
  --phase real
```

For the default configuration, the validation-selected nnU-Net checkpoint is expected at:

```text
nam/downstream/real_checkpoint/polyps/nnunet/best.pt
```

### 5. Train NAM

```bash
python scripts/train_nam_2d.py \
  --config configs/table1_2d.yaml
```

The default Table-I protocol uses:

```text
optimizer          AdamW
learning rate      1e-4
weight decay       1e-2
max iterations     3000
KL weight beta     0.001
kappa_up           0.5
truncated steps    10
DDIM full steps    50
```

The diffusion model and downstream anchor are frozen while the NAM miner is optimized.

### 6. Generate matched Base and NAM synthetic sets

NAM:

```bash
python scripts/generate_2d.py \
  --config configs/table1_2d.yaml \
  --method nam
```

Base diffusion sampling:

```bash
python scripts/generate_2d.py \
  --config configs/table1_2d.yaml \
  --method base
```

For the default Polyps experiment, the fixed synthetic budget is 1,128 samples, matching the number of real training samples.

### 7. Continue downstream training with synthetic data

```bash
python scripts/train_downstream_2d.py \
  --config configs/table1_2d.yaml \
  --phase synthetic
```

The Table I protocol uses matched real and synthetic batches with a 1:1 real-to-synthetic ratio.

### 8. Evaluate

Evaluate the real baseline:

```bash
python scripts/evaluate_2d.py \
  --config configs/table1_2d.yaml \
  --checkpoint-phase real
```

Evaluate the synthetic-augmented model:

```bash
python scripts/evaluate_2d.py \
  --config configs/table1_2d.yaml \
  --checkpoint-phase syn
```

Evaluate adversariality:

```bash
python scripts/evaluate_adversariality.py \
  --config configs/table1_2d.yaml
```

Evaluate FID:

```bash
python scripts/evaluate_fid.py \
  --config configs/table1_2d.yaml
```

---

## Full Table I reproduction protocol

The main Table I experiments use five medical datasets, eight diffusion pipelines, and three downstream architectures.

For every experiment cell, the intended order is:

1. prepare the real-data split;
2. train or register the corresponding diffusion generator;
3. train the real-data downstream baseline;
4. freeze the generator and nnU-Net adversariality anchor;
5. optimize NAM for 3,000 iterations;
6. synthesize a fixed-budget dataset with deterministic DDIM-50;
7. continue downstream training with real and synthetic data;
8. select checkpoints on the validation split;
9. report final performance on the held-out test split.

The Table I continuation protocol uses:

- 1:1 real-to-synthetic batches;
- paired CutMix with probability 0.5;
- architecture-specific optimization schedules for nnU-Net, Swin-Unet/SwinUNETR, and SAMed.

The complete matrix and additional details are available in:

- [`configs/table1_matrix.yaml`](configs/table1_matrix.yaml)
- [`docs/TABLE1_PROTOCOL.md`](docs/TABLE1_PROTOCOL.md)

To list the configured Table I cells:

```bash
python scripts/list_table1.py
```

---

## Supported diffusion models

The repository integrates ten generator settings across several synthesis paradigms.

| Generator | Paradigm | Space | Typical datasets | Local guide | Upstream |
|---|---|---|---|---|---|
| SegDiff | mask → image | 2D pixel-space DPM | ACDC, Polyps | [`README`](nam/diffusion/2D_M2I/segdiff/README.md) | [segmentation-guided diffusion](https://github.com/mazurowski-lab/segmentation-guided-diffusion) |
| DiffBoost | mask → image | 2D latent diffusion | ACDC, Synapse, Polyps | [`README`](nam/diffusion/2D_M2I/diffboost/README.md) | [DiffBoost](https://github.com/NUBagciLab/DiffBoost) |
| FairDiff | mask → image | 2D latent diffusion | Synapse | [`README`](nam/diffusion/2D_M2I/fairdiff/README.md) | [FairDiff](https://github.com/wenyi-li/FairDiff) |
| SiameseDiff | mask → image | 2D Stable Diffusion | Polyps | [`README`](nam/diffusion/2D_M2I/siamesediff/README.md) | [Siamese-Diffusion](https://github.com/Qiukunpeng/Siamese-Diffusion) |
| JoDiffusion | image + mask | joint 2D latent diffusion | ACDC | [`README`](nam/diffusion/2D_M%26I/jodiffusion/README.md) | [JoDiffusion](https://github.com/00why00/JoDiffusion) |
| MedSegFactory | image + mask | dual-stream 2D latent diffusion | Synapse, Polyps | [`README`](nam/diffusion/2D_M%26I/medsegfactory/README.md) | [MedSegFactory](https://github.com/jwmao1/MedSegFactory) |
| VolDiT | mask → volume | 3D latent DiT | LA, ImageCAS | [`README`](nam/diffusion/3D_M2I/voldit/README.md) | [VolDiT](https://github.com/Cardio-AI/voldit) |
| MAISI | mask → volume | 3D latent diffusion | LA, ImageCAS | [`README`](nam/diffusion/3D_M2I/maisi/README.md) | [MONAI MAISI](https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi) |
| ControlNet-SDXL | semantic mask → image | 2D latent diffusion | PASCAL VOC + SBD | [`README`](nam/diffusion/2D_M2I/controlnet_sdxl/README.md) | [ControlNet](https://github.com/lllyasviel/ControlNet) |
| SD-v1.5 + LoRA | text → image | 2D latent diffusion | PneumoniaMNIST, ISIC | [`README`](nam/diffusion/2D_T2I/sd15_lora/README.md) | [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |

The upstream repositories are not copied into this project. Each local adapter README describes where to place the corresponding source code and pretrained weights.

---

## Downstream models

| Model | Task | Dimensionality | Reference |
|---|---|---:|---|
| nnU-Net | medical segmentation | 2D / 3D | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| Swin-Unet | medical segmentation | 2D | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) |
| SwinUNETR | volumetric segmentation | 3D | [MONAI SwinUNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) |
| SAMed | medical segmentation | 2D / slice-wise 3D evaluation | [hitachinsk/SAMed](https://github.com/hitachinsk/SAMed) |
| DeepLabV3-R50 | natural-image segmentation | 2D | [Torchvision](https://pytorch.org/vision/stable/models/deeplabv3.html) |
| Mask2Former-R50 | natural-image segmentation | 2D | [Mask2Former](https://github.com/facebookresearch/Mask2Former) |
| ResNet-50 | medical classification | 2D | [Torchvision](https://pytorch.org/vision/stable/models/resnet.html) |
| ViT-S/16 | medical classification | 2D | [timm](https://github.com/huggingface/pytorch-image-models) |

Real-only and synthetic-augmented checkpoints are stored separately:

```text
nam/downstream/
├── real_checkpoint/<dataset>/<model>/
│   ├── best.pt
│   ├── latest.pt
│   └── epoch_XXXX.pt
└── syn_checkpoint/<dataset>/<generator>/<model>/
    ├── best.pt
    ├── latest.pt
    └── epoch_XXXX.pt
```

`best.pt` is selected using the validation split. NAM uses the real-data checkpoint as the frozen adversariality anchor.

---

## Datasets

Data are not redistributed. Dataset packages under `nam/data/` provide preparation notes, split manifests, loaders, and prompt metadata where required.

| Scenario | Dataset | Task / modality | Train | Val | Test |
|---|---|---|---:|---:|---:|
| In-domain | ACDC | cardiac MRI segmentation | 4,010 | 572 | 1,146 |
| In-domain | Synapse | abdominal CT segmentation | 2,645 | 378 | 756 |
| In-domain / cross-center source | CVC-ClinicDB + Kvasir-SEG | polyp segmentation | 1,128 | 161 | 323 |
| 3D | LA | left-atrium MRI segmentation | 70 | 10 | 20 |
| 3D | ImageCAS | coronary CTA segmentation | 700 | 100 | 200 |
| Cross-center | EndoScene / CVC-300 | polyp evaluation | — | — | 60 |
| Cross-center | CVC-ColonDB | polyp evaluation | — | — | 380 |
| Cross-center | ETIS-LaribPolypDB | polyp evaluation | — | — | 196 |
| Cross-modality | MMWHS CT | whole-heart segmentation | 3,714 | 531 | 1,060 |
| Cross-modality | MMWHS MRI | whole-heart segmentation | 2,029 | 290 | 579 |
| Natural image | PASCAL VOC 2012 + SBD | semantic segmentation | 8,422 | 1,203 | 2,406 |
| Classification | PneumoniaMNIST-224 | chest X-ray classification | 4,099 | 585 | 1,171 |
| Classification | ISIC | dermoscopic classification | 1,925 | 275 | 550 |

Detailed split information is available in [`nam/data/SPLITS.md`](nam/data/SPLITS.md).

### Manifest format

Segmentation:

```text
sample_id relative/image/path relative/target/path [optional prompt]
```

Classification:

```text
sample_id relative/image/path class_id [optional prompt]
```

The manifests use dataset-relative paths. After preprocessing, place the corresponding files below the dataset package's ignored `data/` directory.

---

## 2D medical segmentation workflow

The general 2D workflow is:

```bash
# 1. train/register a diffusion model
python scripts/train_diffusion_2d.py --config <CONFIG>

# 2. train the real-data baseline
python scripts/train_downstream_2d.py --config <CONFIG> --phase real

# 3. train NAM
python scripts/train_nam_2d.py --config <CONFIG>

# 4. synthesize NAM and Base sets
python scripts/generate_2d.py --config <CONFIG> --method nam
python scripts/generate_2d.py --config <CONFIG> --method base

# 5. train downstream model with synthetic augmentation
python scripts/train_downstream_2d.py --config <CONFIG> --phase synthetic

# 6. evaluate
python scripts/evaluate_2d.py --config <CONFIG> --checkpoint-phase syn
python scripts/evaluate_adversariality.py --config <CONFIG>
python scripts/evaluate_fid.py --config <CONFIG>
```

---

## 3D workflow

For volumetric experiments, use the corresponding 3D entry points:

```bash
python scripts/train_diffusion_3d.py --config configs/table1_3d.yaml
python scripts/train_downstream_3d.py --config configs/table1_3d.yaml --phase real
python scripts/train_nam_3d.py --config configs/table1_3d.yaml
python scripts/generate_3d.py --config configs/table1_3d.yaml --method nam
python scripts/train_downstream_3d.py --config configs/table1_3d.yaml --phase synthetic
python scripts/evaluate_3d.py --config configs/table1_3d.yaml --checkpoint-phase syn
python scripts/evaluate_adversariality.py --config configs/table1_3d.yaml --spatial-dims 3
```

See the VolDiT and MAISI adapter READMEs for their external dependencies and checkpoint layout.

---

## Natural-image transfer

The PASCAL VOC + SBD branch uses ControlNet-SDXL with DeepLabV3-R50 and Mask2Former-R50.

Configuration:

```text
configs/controlnet_sdxl_voc.yaml
```

Typical workflow:

```bash
python scripts/train_diffusion_2d.py --config configs/controlnet_sdxl_voc.yaml
python scripts/train_downstream_2d.py --config configs/controlnet_sdxl_voc.yaml --phase real
python scripts/train_nam_2d.py --config configs/controlnet_sdxl_voc.yaml
python scripts/generate_2d.py --config configs/controlnet_sdxl_voc.yaml --method nam
python scripts/train_downstream_2d.py --config configs/controlnet_sdxl_voc.yaml --phase synthetic
python scripts/evaluate_2d.py --config configs/controlnet_sdxl_voc.yaml --checkpoint-phase syn
```

---

## Medical classification transfer

PneumoniaMNIST-224 and ISIC use a task-adapted SD-v1.5 + LoRA generator. ResNet-50 and ViT-S/16 are used as downstream classifiers.

Configurations:

```text
configs/sd15_lora_pneumoniamnist.yaml
configs/sd15_lora_isic.yaml
```

Example:

```bash
python scripts/train_diffusion_2d.py \
  --config configs/sd15_lora_pneumoniamnist.yaml

python scripts/train_downstream_2d.py \
  --config configs/sd15_lora_pneumoniamnist.yaml \
  --phase real

python scripts/train_nam_2d.py \
  --config configs/sd15_lora_pneumoniamnist.yaml

python scripts/generate_2d.py \
  --config configs/sd15_lora_pneumoniamnist.yaml \
  --method nam

python scripts/train_downstream_2d.py \
  --config configs/sd15_lora_pneumoniamnist.yaml \
  --phase synthetic

python scripts/evaluate_classification.py \
  --config configs/sd15_lora_pneumoniamnist.yaml
```

---

## Adversariality evaluation

`scripts/evaluate_adversariality.py` measures the difficulty of a stored synthetic dataset using a frozen downstream checkpoint.

Example:

```bash
python scripts/evaluate_adversariality.py \
  --config configs/table1_2d.yaml
```

Select another proxy:

```bash
python scripts/evaluate_adversariality.py \
  --config configs/table1_2d.yaml \
  --proxy lce
```

The exported CSV contains sample identifiers and several per-sample scores, including:

```text
id
path
lce
lcbce
ldice
adv_lce
adv_lcbce
adv_ldice
adv
```

Larger normalized adversariality scores correspond to harder samples under the selected downstream model.

The evaluation is fixed-budget: stored samples are read in deterministic dataset order and truncated to the requested budget without adding test-time perturbations.

See [`docs/EVALUATION_VISUALIZATION.md`](docs/EVALUATION_VISUALIZATION.md).

---

## Distribution alignment / FID

Use:

```bash
python scripts/evaluate_fid.py \
  --config configs/table1_2d.yaml
```

The feature extractor is configured in the experiment YAML. For example, the default Polyps configuration uses Inception-v3 features.

Keep the real/synthetic splits and encoder fixed when comparing Base and NAM.

---

## Visualization

Install:

```bash
pip install -e ".[visualization]"
```

Available publication-analysis modules include:

```bash
python -m nam.evaluation.Vis.tsne
python -m nam.evaluation.Vis.adversariality_distribution
python -m nam.evaluation.Vis.cross_model_consistency
```

### t-SNE

The t-SNE utility extracts downstream bottleneck features, projects all configured groups jointly, and stores the source features together with the plotted coordinates.

### Cross-model consistency

Cross-model analysis matches samples by sample ID and reports:

- Pearson correlation;
- Spearman rank correlation;
- mean absolute difference.

This is useful for testing whether samples mined with one downstream anchor remain difficult for another downstream model.

Figures are saved as PDF and PNG.

---

## NAM objectives and important hyperparameters

The miner reward is selected in the configuration. Available adversariality proxies include:

```text
lce
lcbce
ldice
lfocal
```

For the main Table I protocol, the NAM optimization settings are:

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-2 |
| Iterations | 3,000 |
| KL weight `beta` | 0.001 |
| Adversariality cap `kappa_up` | 0.5 |
| Truncated rollout | 10 steps |
| Final sampling | deterministic DDIM-50 |

The exact diffusion-specific configuration remains in the corresponding YAML/adapter package.

---

## Mitigation strategies

High-adversariality mining can also expose defective modes that already exist in a diffusion model. The repository includes four optional sampling-time mitigation strategies.

| Method | Purpose |
|---|---|
| HAT | truncates / controls excessive adversariality |
| QSF | filters samples using a quality/semantic score |
| LSRS | reranks candidate seeds using latent semantic signals |
| ASG | uses attention signals to suppress incompatible generations |

Configuration:

```text
configs/mitigation.yaml
```

Implementation:

```text
nam/mitigation/
```

Use mitigation only when required by a particular experiment. Some strategies add substantial sampling overhead or require model-specific hooks.

---

## Checkpoints and pretrained weights

Large model binaries are intentionally excluded from Git.

### Diffusion models

Each diffusion adapter contains a local README describing:

- the upstream source repository;
- required pretrained weights;
- expected filenames;
- checkpoint locations;
- task-specific setup.

For example:

```text
nam/diffusion/2D_M2I/siamesediff/README.md
```

### Downstream models

Real-data baseline:

```text
nam/downstream/real_checkpoint/<dataset>/<model>/best.pt
```

Synthetic-augmented model:

```text
nam/downstream/syn_checkpoint/<dataset>/<generator>/<model>/best.pt
```

### NAM miner

The miner checkpoint path is defined by the experiment configuration. The default Polyps/SiameseDiff example uses:

```text
outputs/polyps-siamesediff-nnunet/checkpoints/nam_latest.pt
```

Do not mix checkpoints produced from different generators, datasets, downstream anchors, or spatial dimensions.

---

## Outputs and experiment records

The code records experiment metadata and metrics alongside training artifacts. Depending on the entry point and configuration, outputs can include:

```text
checkpoints/
tensorboard/
metrics.jsonl
config.json
environment.json
visualizations/
samples.jsonl
```

Typical contents:

| File / directory | Description |
|---|---|
| `checkpoints/` | latest, best, and periodic model states |
| `tensorboard/` | scalar metrics, image previews, histograms |
| `metrics.jsonl` | machine-readable training/evaluation history |
| `config.json` | resolved experiment configuration |
| `environment.json` | Python/PyTorch/CUDA environment metadata |
| `visualizations/` | preview and diagnostic images |
| `samples.jsonl` | generated sample IDs, seeds, prompts/conditions, checkpoints and output paths |

TensorBoard:

```bash
tensorboard --logdir outputs --port 6006
```

---

## Paper-to-code map

| Experiment / analysis | Main files |
|---|---|
| Table I experiment matrix | `configs/table1_matrix.yaml`, `scripts/list_table1.py` |
| 2D Table I example | `configs/table1_2d.yaml` |
| 3D experiments | `configs/table1_3d.yaml` |
| NAM training | `scripts/train_nam_2d.py`, `scripts/train_nam_3d.py` |
| Base/NAM sampling | `scripts/generate_2d.py`, `scripts/generate_3d.py` |
| Downstream training | `scripts/train_downstream_2d.py`, `scripts/train_downstream_3d.py` |
| Segmentation evaluation | `scripts/evaluate_2d.py`, `scripts/evaluate_3d.py` |
| Classification evaluation | `scripts/evaluate_classification.py` |
| Adversariality analysis | `scripts/evaluate_adversariality.py` |
| FID | `scripts/evaluate_fid.py` |
| t-SNE | `nam/evaluation/Vis/tsne.py` |
| Adversariality plots | `nam/evaluation/Vis/adversariality_distribution.py` |
| Cross-model consistency | `nam/evaluation/Vis/cross_model_consistency.py` |
| Mitigation | `configs/mitigation.yaml`, `nam/mitigation/` |
| Table I protocol | `docs/TABLE1_PROTOCOL.md` |
| Evaluation guide | `docs/EVALUATION_VISUALIZATION.md` |
| Dataset split inventory | `nam/data/SPLITS.md` |

---

## Methods compared in the paper

The paper compares NAM with random diffusion sampling and representative methods from several families. Third-party comparison implementations are not redistributed here unless explicitly stated by their own local package.

### Heuristic targeting

- UGDM
- VGD

### Diversity-oriented augmentation

- AugPaint
- DiffAug
- CIG
- DA-Fusion

### Utility-proxy methods

- GAL
- UtilGen

### Adversarial guidance

- AdvDiffuser
- P2P
- Diff-PGD
- DiffAttack
- NatADiff

When reproducing these comparisons, use the original paper/repository and respect its license and preprocessing protocol.

---

## Reproducibility notes

For meaningful Base-vs-NAM comparisons:

1. use the same real-data split;
2. use the same diffusion checkpoint;
3. use the same downstream initialization;
4. keep the synthetic-data budget fixed;
5. keep the DDIM schedule and condition list fixed;
6. use the same downstream continuation schedule;
7. select checkpoints using validation data only;
8. report final metrics on the held-out test set.

For Table I, the synthetic budget matches the real training-set size.

For adversariality analysis, keep the downstream checkpoint fixed across the compared synthetic groups.

For cross-model consistency, match samples by stable sample ID rather than by CSV row position.

---

## Troubleshooting

### Configuration cannot be loaded

Run:

```bash
python scripts/train_nam_2d.py \
  --config <CONFIG> \
  --dry-run \
  --print-config
```

Check dataset factories, generator names, checkpoint paths and optional dependencies.

### Dataset file not found

Confirm that:

- the dataset-specific README has been followed;
- the downloaded data have been converted to the expected layout;
- manifest paths are relative to the package's `data/` directory;
- the split files have not been edited unintentionally.

### Diffusion checkpoint mismatch

Verify:

- upstream revision;
- diffusion architecture/config;
- latent/noise channels;
- condition channels;
- scheduler/prediction type;
- checkpoint filename.

### NAM shape mismatch

Check whether the experiment is 2D or 3D and whether the miner matches the generator's noise shape.

### CUDA out of memory

Reduce batch size first. For 3D experiments, patch/volume size and visualization frequency can also dominate memory.

Do not change the synthetic budget when comparing Base and NAM unless the experiment is explicitly studying budget scaling.

### `best.pt` is missing

Train the real-data baseline and complete validation before NAM training. The NAM anchor should be a validation-selected real-data checkpoint.

### FID numbers are inconsistent

Confirm that Base and NAM use:

- the same feature extractor;
- the same real split;
- the same synthetic budget;
- the same preprocessing/resolution.

---

## Publications

### TPAMI extended manuscript

**Mining Native Adversariality in Diffusion Models for Medical Generalization**  
Hongyu Zhang, Haipeng Chen, Zhimin Xu, Chengxin Yang, and Yingda Lyu.  
Submitted to *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2026.

### CVPR 2026 Highlight

**Diffusion-Based Native Adversarial Synthesis for Enhanced Medical Segmentation Generalization**  
Hongyu Zhang, Haipeng Chen, Zhimin Xu, Chengxin Yang, and Yingda Lyu.  
*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2026. **Highlight**.

---

## Citation

If this repository is useful for your research, please cite the relevant paper.

```bibtex
@article{zhang2026mining,
  title   = {Mining Native Adversariality in Diffusion Models for Medical Generalization},
  author  = {Zhang, Hongyu and Chen, Haipeng and Xu, Zhimin and Yang, Chengxin and Lyu, Yingda},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  note    = {Submitted}
}

@inproceedings{zhang2026diffusion,
  title     = {Diffusion-Based Native Adversarial Synthesis for Enhanced Medical Segmentation Generalization},
  author    = {Zhang, Hongyu and Chen, Haipeng and Xu, Zhimin and Yang, Chengxin and Lyu, Yingda},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {1461--1471},
  year      = {2026},
  note      = {Highlight}
}
```

Please also cite the generator, downstream architecture, and dataset used in your experiment.

---

## Acknowledgements

This project builds on open-source work in diffusion models, medical-image synthesis, segmentation, classification, and evaluation. The corresponding upstream repositories are linked from each adapter README.

We thank the authors of the datasets and open-source implementations used in this project.

---

## License

The NAM code in this repository is released under the [Apache License 2.0](LICENSE).

Datasets, pretrained model weights, third-party repositories, and external checkpoints remain subject to their original licenses and terms of use.
