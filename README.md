<div align="center">

# 🔥 Native Adversariality Mining

### *Mining Native Adversariality in Diffusion Models for Medical Generalization*

**Official PyTorch implementation of NAM · TPAMI extended experiments · CVPR 2026 Highlight preliminary work**

<br>

<a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Diffusion-Based_Native_Adversarial_Synthesis_for_Enhanced_Medical_Segmentation_Generalization_CVPR_2026_paper.pdf"><img src="https://img.shields.io/badge/CVPR%202026-Highlight-ff4d4f?style=for-the-badge" alt="CVPR 2026 Highlight"></a>
<a href="#publications"><img src="https://img.shields.io/badge/TPAMI-Extended%20Manuscript-0054a6?style=for-the-badge" alt="TPAMI extended manuscript"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f?style=for-the-badge" alt="Apache-2.0"></a>

<br>

<a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-%E2%89%A53.10-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python >=3.10"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.1-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch >=2.1"></a>
<img src="https://img.shields.io/badge/Scope-2D%20%2B%203D-6f42c1?style=flat-square" alt="2D + 3D">
<img src="https://img.shields.io/badge/Paradigms-M2I%20%7C%20M%26I%20%7C%20T2I-8250df?style=flat-square" alt="M2I M&I T2I">
<img src="https://img.shields.io/badge/Training-Frozen%20DM%20%2B%20Anchor-0969da?style=flat-square" alt="Frozen DM and anchor">
<img src="https://img.shields.io/badge/Reproduction-Config--Driven-1f883d?style=flat-square" alt="Config-driven reproduction">

<br>

<a href="https://github.com/JackCD99/Native-Adversariality-Mining/stargazers"><img src="https://img.shields.io/github/stars/JackCD99/Native-Adversariality-Mining?style=flat-square&logo=github" alt="GitHub stars"></a>
<a href="https://github.com/JackCD99/Native-Adversariality-Mining/network/members"><img src="https://img.shields.io/github/forks/JackCD99/Native-Adversariality-Mining?style=flat-square&logo=github" alt="GitHub forks"></a>
<a href="https://github.com/JackCD99/Native-Adversariality-Mining/issues"><img src="https://img.shields.io/github/issues/JackCD99/Native-Adversariality-Mining?style=flat-square&logo=github" alt="GitHub issues"></a>
<a href="https://github.com/JackCD99/Native-Adversariality-Mining/commits/main"><img src="https://img.shields.io/github/last-commit/JackCD99/Native-Adversariality-Mining?style=flat-square&logo=github" alt="Last commit"></a>

<br><br>

📄 **[CVPR Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Diffusion-Based_Native_Adversarial_Synthesis_for_Enhanced_Medical_Segmentation_Generalization_CVPR_2026_paper.pdf)**
&nbsp;·&nbsp;
📎 **[Supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhang_Diffusion-Based_Native_Adversarial_CVPR_2026_supplemental.pdf)**
&nbsp;·&nbsp;
🧪 **[Table I Protocol](docs/TABLE1_PROTOCOL.md)**
&nbsp;·&nbsp;
📊 **[Evaluation Guide](docs/EVALUATION_VISUALIZATION.md)**
&nbsp;·&nbsp;
🐛 **[Issues](https://github.com/JackCD99/Native-Adversariality-Mining/issues)**

<br>

**[Overview](#overview)** · **[Method](#method)** · **[Results](#results)** · **[Install](#install)** · **[Data](#datasets)** · **[Generators](#generators)** · **[Reproduce](#reproduction)** · **[Evaluate](#evaluation)** · **[Baselines](#baselines)** · **[FAQ](#faq)** · **[Cite](#citation)**

</div>

---

<a id="news"></a>

## 📢 News

| Date | Update |
|---|---|
| **2026.08** | 🚀 Public release of the TPAMI-oriented NAM codebase: 2D/3D synthesis, M2I/M&I/T2I branches, transfer experiments, analysis utilities, and defective-mode mitigation. |
| **2026.06** | 🌟 *Diffusion-Based Native Adversarial Synthesis for Enhanced Medical Segmentation Generalization* presented at **CVPR 2026** and selected as a **Highlight** paper. |

---

<a id="overview"></a>

## 🌟 Overview

<p>
  <img src="https://img.shields.io/badge/Core-Seed--Level%20Mining-6f42c1?style=flat-square" alt="Seed-level mining">
  <img src="https://img.shields.io/badge/Generator-Frozen-0969da?style=flat-square" alt="Frozen generator">
  <img src="https://img.shields.io/badge/Anchor-Frozen-0969da?style=flat-square" alt="Frozen anchor">
  <img src="https://img.shields.io/badge/Goal-Downstream%20Generalization-1f883d?style=flat-square" alt="Downstream generalization">
</p>

Diffusion models can synthesize realistic medical images, but **visual fidelity alone does not guarantee downstream utility**. Random sampling tends to overproduce already well-learned modes, while informative hard modes can be sparse. **Native Adversariality Mining (NAM)** learns a lightweight miner that uses the frozen diffusion model's early denoising response to reselect initial noise, increasing the probability of generating hard yet generator-supported samples.

| **Core idea** | **Optimization** | **Coverage** | **Reproduction** |
|---|---|---|---|
| Mine informative seeds from a pretrained DM | Frozen generator + frozen anchor; miner only | 13 benchmarks · 10 generators · 2D/3D | Config-driven runs · fixed budgets · fixed seeds |

<p align="center">
  <img src="assets/nam_overview.png" width="920" alt="Overview of Native Adversariality Mining">
</p>

| 🧩 Aspect | 🎲 Base sampling | ⛏️ **NAM** | ⚔️ Adversarial guidance |
|---|---|---|---|
| Diffusion model | Frozen | **Frozen** | Frozen |
| Sampling trajectory | Original | **Original** | Often modified |
| Initial seed distribution | Gaussian | **Adapted by miner** | Usually unchanged |
| Downstream difficulty targeted | No | **Yes** | Yes |
| Trainable component | None | **NAM miner only** | Guidance/attack variables |
| Primary goal | Generic synthesis | **Native hard-mode mining** | Attack-style difficulty |
| Paper behavior | Reference distribution | **Near-base / mild shift** | Often larger alignment degradation |

> [!TIP]
> The most convenient end-to-end starting point is **Polyps + SiameseDiff + nnU-Net** in `configs/table1_2d.yaml`.

### ✨ Repository coverage

| NAM / synthesis | Data & tasks | Downstream | Analysis & reproduction |
|---|---|---|---|
| 2D & 3D miners | MRI / CT / CTA / RGB / X-ray / dermoscopy | nnU-Net / Swin-Unet / SwinUNETR / SAMed | Table I matrix + fixed seeds |
| M2I / M&I / T2I | In-domain / cross-center / cross-modality | ResNet-50 / ViT-S/16 | Adversariality / FID / t-SNE |
| Frozen generator + frozen anchor | Medical segmentation / classification | DeepLabV3 / Mask2Former | Cross-model consistency |
| HAT / QSF / LSRS / ASG | Natural-image segmentation transfer | Real + synthetic continuation | TensorBoard / JSONL / environment snapshots |

---

<a id="method"></a>

## 🧠 Method at a Glance

<p>
  <img src="https://img.shields.io/badge/Search%20Space-Initial%20Noise-8250df?style=flat-square" alt="Initial-noise search">
  <img src="https://img.shields.io/badge/Optimization-Miner%20Only-d29922?style=flat-square" alt="Miner-only optimization">
  <img src="https://img.shields.io/badge/Training-Truncated%20Rollout-bf8700?style=flat-square" alt="Truncated rollout">
  <img src="https://img.shields.io/badge/Sampling-Original%20DDIM-1f883d?style=flat-square" alt="Original DDIM">
</p>

<p align="center">
  <img src="assets/native_vs_artificial_adversariality.png" width="780" alt="Native versus artificial adversariality">
</p>

NAM treats the **initial noise distribution** as the mining space. The generator and downstream anchor remain frozen; only the miner is optimized. Final synthesis keeps the original diffusion sampler.

| 🔹 Stage | 📥 Input | ⚙️ Operation | 🔥 Trainable? | 📤 Output |
|---|---|---|---|---|
| Probe | `x_T ~ N(0,I)` + condition | Initial frozen-DM denoising response | No | Early score/response |
| Mining | Early response | Predict noise-prior shift | **NAM miner only** | Reselected initial noise |
| Truncated training rollout | Reselected noise | Early differentiable DDIM rollout | Miner receives gradients | Adversariality signal |
| Regularization | Mined prior | KL to base Gaussian + `kappa_up` control | Miner only | Stable native mining objective |
| Final synthesis | Trained miner + frozen DM | Full original deterministic DDIM | No | NAM synthetic set |
| Downstream continuation | Real + synthetic data | Architecture-specific training | Downstream model | Final evaluation checkpoint |

### 🔬 Main optimization settings

| Parameter | Default | Parameter | Default |
|---|---:|---|---:|
| Optimizer | AdamW | Learning rate | `1e-4` |
| Weight decay | `1e-2` | Miner iterations | `3,000` |
| KL weight `beta` | `0.001` | Adversariality cap `kappa_up` | `0.5` |
| Truncated rollout | `10` steps | Final sampler | deterministic DDIM-50 |
| Rewards | `lce`, `lcbce`, `ldice`, `lfocal` | Main seeds | `42`, `3407`, `2026` |

> [!NOTE]
> NAM is **not training-free**. It avoids diffusion-model retraining, but the miner is optimized for the selected generator/task/anchor configuration.

---

<a id="results"></a>

## 🏆 Representative Results

<p>
  <img src="https://img.shields.io/badge/Medical%20Segmentation-DSC%20↑-1f883d?style=flat-square" alt="DSC">
  <img src="https://img.shields.io/badge/Transfer-Cross--Center%20%7C%20Cross--Modality-0969da?style=flat-square" alt="Transfer">
  <img src="https://img.shields.io/badge/Scope-Classification%20%7C%20Natural%20Images-8250df?style=flat-square" alt="Task transfer">
</p>

The values below are manuscript-reported reference gains over the real-only downstream baseline.

| Dataset / generator | nnU-Net: Base → NAM | Swin(-Unet): Base → NAM | SAMed: Base → NAM |
|---|---:|---:|---:|
| ACDC / DiffBoost | +3.50 → **+8.26 DSC** | +2.28 → **+8.39 DSC** | — |
| Synapse / DiffBoost | +4.09 → **+9.49 DSC** | +3.17 → **+8.56 DSC** | — |
| Polyps / SiameseDiff | +5.05 → **+10.74 DSC** | +4.38 → **+9.80 DSC** | +4.60 → **+10.00 DSC** |
| LA / MAISI | +2.66 → **+4.84 DSC** | — | — |
| ImageCAS / MAISI | +3.06 → **+6.30 DSC** | — | — |

| Transfer coverage | Included settings | Reference efficiency (SiameseDiff, RTX 4090) |
|---|---|---|
| Cross-center | EndoScene/CVC-300, CVC-ColonDB, ETIS | Standard DDIM: **7.41 s/image** |
| Cross-modality | MMWHS MRI ↔ CT | **NAM: 7.85 s/image** |
| Classification | PneumoniaMNIST-224, ISIC | NatADiff reference: **176.91 s/image** |
| Natural-image transfer | PASCAL VOC 2012 + SBD | Hardware/implementation dependent |

---

<a id="coverage"></a>

## 🧪 Reproducible Experiment Coverage

| 🌐 Setting | 📦 Dataset(s) | 🩻 Modality / task | 🧬 Main generator(s) | 🧠 Downstream |
|---|---|---|---|---|
| In-domain 2D | ACDC | cardiac MRI segmentation | SegDiff, DiffBoost, JoDiffusion | nnU-Net, Swin-Unet, SAMed |
| In-domain 2D | Synapse | abdominal CT segmentation | FairDiff, DiffBoost, MedSegFactory | nnU-Net, Swin-Unet, SAMed |
| In-domain / source | Polyps | RGB polyp segmentation | SegDiff, DiffBoost, SiameseDiff, MedSegFactory | nnU-Net, Swin-Unet, SAMed |
| In-domain 3D | LA | left-atrium MRI segmentation | VolDiT, MAISI | nnU-Net, SwinUNETR, SAMed |
| In-domain 3D | ImageCAS | coronary CTA segmentation | VolDiT, MAISI | nnU-Net, SwinUNETR, SAMed |
| Cross-center | EndoScene/CVC-300, ColonDB, ETIS | polyp segmentation | source-trained generators | segmentation models |
| Cross-modality | MMWHS MRI ↔ CT | whole-heart segmentation | task-specific branches | segmentation models |
| Classification | PneumoniaMNIST-224, ISIC | X-ray / dermoscopy | SD-v1.5 + LoRA | ResNet-50, ViT-S/16 |
| Natural-image transfer | PASCAL VOC 2012 + SBD | semantic segmentation | ControlNet-SDXL | DeepLabV3, Mask2Former |

`configs/table1_matrix.yaml` stores the primary Table I matrix and seeds. List configured cells with:

```bash
python scripts/list_table1.py
```

---

<a id="install"></a>

## ⚙️ Installation & Quick Start

### Requirements

| Runtime | Requirement | Runtime | Requirement |
|---|---|---|---|
| Python | `>=3.10` | PyTorch | `>=2.1` |
| GPU | CUDA-capable recommended | OS | Linux recommended; WSL2 useful on Windows |
| Paper hardware | 4× RTX 4090 | Storage | datasets + generator weights + synthetic outputs |

Install the CUDA-compatible PyTorch build first, then:

```bash
git clone https://github.com/JackCD99/Native-Adversariality-Mining.git
cd Native-Adversariality-Mining

python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[medical,diffusion,visualization]"

python scripts/train_nam_2d.py --config configs/table1_2d.yaml --dry-run --print-config
python scripts/list_table1.py
```

> [!IMPORTANT]
> `--dry-run` validates configuration/imports only. Full experiments still require the corresponding dataset, upstream generator code when applicable, generator weights, and a real-data downstream checkpoint.

### Optional dependency groups

| Extra | Main packages | Typical use | Install |
|---|---|---|---|
| `medical` | MONAI, nibabel, h5py | medical / 3D data | `pip install -e ".[medical]"` |
| `diffusion` | diffusers, PEFT, safetensors | latent-diffusion pipelines | `pip install -e ".[diffusion]"` |
| `natural` | torchvision, transformers, accelerate | VOC + ControlNet/SDXL | `pip install -e ".[natural]"` |
| `classification` | torchvision, timm | ResNet / ViT | `pip install -e ".[classification]"` |
| `volumetric-diffusion` | MONAI, OmegaConf, einops, timm | VolDiT / MAISI | `pip install -e ".[volumetric-diffusion]"` |
| `visualization` | matplotlib, scikit-learn | t-SNE / plots | `pip install -e ".[visualization]"` |
| `xray` | torchxrayvision | X-ray evaluation | `pip install -e ".[xray]"` |
| `mitigation-vqa` | transformers, accelerate | QSF | `pip install -e ".[mitigation-vqa]"` |

Linux is the recommended environment for full reproduction because several upstream medical/diffusion projects assume Linux shell/CUDA tooling. Native Windows remains suitable for code inspection, configuration work, dry runs, and many pure-PyTorch components.

---

<a id="repo-map"></a>

## 🗂️ Repository & Configuration Map

```text
Native-Adversariality-Mining/
├── assets/                  # README figures
├── configs/                 # experiment YAMLs
├── docs/                    # reproduction / evaluation notes
├── nam/
│   ├── data/                # datasets + split manifests
│   ├── diffusion/           # 2D_M2I / 2D_M&I / 2D_T2I / 3D_M2I
│   ├── downstream/          # models + real/synthetic training
│   ├── engine/              # shared execution
│   ├── evaluation/          # metrics + visualization
│   ├── mitigation/          # HAT / QSF / LSRS / ASG
│   └── utils/               # config / logging / seeds / I/O
├── pretrained_weights/
├── scripts/
├── pyproject.toml
└── README.md
```

| Goal | Entry | Goal | Entry |
|---|---|---|---|
| Default 2D experiment | `configs/table1_2d.yaml` | Main matrix | `configs/table1_matrix.yaml` |
| Train NAM | `scripts/train_nam_2d.py`, `train_nam_3d.py` | Generate Base/NAM | `scripts/generate_2d.py`, `generate_3d.py` |
| Train downstream | `scripts/train_downstream_2d.py`, `train_downstream_3d.py` | Segmentation eval | `scripts/evaluate_2d.py`, `evaluate_3d.py` |
| Classification eval | `scripts/evaluate_classification.py` | Adversariality | `scripts/evaluate_adversariality.py` |
| FID | `scripts/evaluate_fid.py` | Table I protocol | `docs/TABLE1_PROTOCOL.md` |
| Visualization | `docs/EVALUATION_VISUALIZATION.md` | Dataset inventory | `nam/data/SPLITS.md` |

### CLI

| Option | Purpose | Example |
|---|---|---|
| `--config PATH` | experiment YAML/JSON | `--config configs/table1_2d.yaml` |
| `--set KEY=VALUE ...` | dotted overrides | `--set runtime.seed=3407 training.reward=lcbce` |
| `--dry-run` | validate config/imports without loading data/weights | recommended before GPU runs |
| `--print-config` | print resolved config | useful for provenance |
| `--log-level` | `DEBUG/INFO/WARNING/ERROR` | default `INFO` |

```bash
python scripts/train_nam_2d.py \
  --config configs/table1_2d.yaml \
  --set runtime.seed=3407 training.reward=lcbce \
  --dry-run --print-config
```

### Main configurations

| Configuration | Use | Configuration | Use |
|---|---|---|---|
| `table1_2d.yaml` | Polyps + SiameseDiff + nnU-Net reference | `table1_3d.yaml` | 3D reference |
| `table1_matrix.yaml` | main paper matrix | `segdiff_2d.yaml` | SegDiff |
| `diffboost_2d.yaml` | DiffBoost | `fairdiff_2d.yaml` | FairDiff |
| `jodiffusion_2d.yaml` | JoDiffusion | `medsegfactory_2d.yaml` | MedSegFactory |
| `voldit_3d.yaml` | VolDiT | `maisi_3d.yaml` | MAISI |
| `controlnet_sdxl_voc.yaml` | VOC/SBD transfer | `sd15_lora_pneumoniamnist.yaml` | PneumoniaMNIST |
| `sd15_lora_isic.yaml` | ISIC classification | `mitigation.yaml` | mitigation studies |

---

<a id="datasets"></a>

## 📦 Datasets

<p>
  <img src="https://img.shields.io/badge/Benchmarks-13-6f42c1?style=flat-square" alt="13 benchmarks">
  <img src="https://img.shields.io/badge/Dimensions-2D%20%2B%203D-0969da?style=flat-square" alt="2D and 3D">
  <img src="https://img.shields.io/badge/Splits-Train%20%7C%20Val%20%7C%20Test-1f883d?style=flat-square" alt="Dataset splits">
</p>

Raw datasets are **not redistributed**. Each local package contains its loader, manifests, and preparation notes. Keep split membership unchanged when comparing with the manuscript.

| Scenario | Dataset | Modality / task | Train | Val | Test | Resolution |
|---|---|---|---:|---:|---:|---|
| In-domain | ACDC | cardiac MRI segmentation | 4,010 | 572 | 1,146 | 256×256 |
| In-domain | Synapse | abdominal CT segmentation | 2,645 | 378 | 756 | 256×256 |
| Source / in-domain | CVC-ClinicDB + Kvasir-SEG | RGB polyp segmentation | 1,128 | 161 | 323 | 256×256 |
| 3D | LA | left-atrium MRI segmentation | 70 | 10 | 20 | 192×192×96 |
| 3D | ImageCAS | coronary CTA segmentation | 700 | 100 | 200 | 192×192×96 |
| Cross-center | EndoScene / CVC-300 | polyp evaluation | — | — | 60 | 256×256 |
| Cross-center | CVC-ColonDB | polyp evaluation | — | — | 380 | 256×256 |
| Cross-center | ETIS-LaribPolypDB | polyp evaluation | — | — | 196 | 256×256 |
| Cross-modality | MMWHS CT | whole-heart segmentation | 3,714 | 531 | 1,060 | 256×256 |
| Cross-modality | MMWHS MRI | whole-heart segmentation | 2,029 | 290 | 579 | 256×256 |
| Natural image | PASCAL VOC 2012 + SBD | semantic segmentation | 8,422 | 1,203 | 2,406 | 512×512 |
| Classification | PneumoniaMNIST-224 | chest X-ray classification | 4,099 | 585 | 1,171 | 224×224 |
| Classification | ISIC | dermoscopic classification | 1,925 | 275 | 550 | 256×256 |

Detailed split inventory: [`nam/data/SPLITS.md`](nam/data/SPLITS.md)

| Item | Format / location |
|---|---|
| Segmentation manifest | `sample_id relative/image/path relative/target/path [optional prompt]` |
| Classification manifest | `sample_id relative/image/path class_id [optional prompt]` |
| Typical package | `nam/data/<dataset>/{README.md,dataset.py,train.list,val.list,test.list,data/}` |
| Polyps example | `nam/data/polyps/README.md`; unified 7:1:2 split, seed 42, binary 256×256 masks |
| Local data | place below each package's ignored `data/` directory |
| Licensing | follow each original dataset's license/use conditions |

---

<a id="generators"></a>

## 🧬 Supported Generators

<p>
  <img src="https://img.shields.io/badge/Generators-10-6f42c1?style=flat-square" alt="10 generators">
  <img src="https://img.shields.io/badge/Paradigms-M2I%20%7C%20M%26I%20%7C%20T2I-8250df?style=flat-square" alt="Synthesis paradigms">
  <img src="https://img.shields.io/badge/Space-Pixel%20%7C%20Latent%20%7C%203D-0969da?style=flat-square" alt="Generator spaces">
</p>

Upstream repositories and large pretrained weights remain external. Each adapter README documents source code, required weights, and checkpoint locations.

| 🧬 Generator | 🔀 Paradigm | 🔢 Noise layout | 📦 Main datasets | 📘 Local guide | 🔗 Upstream |
|---|---|---|---|---|---|
| **SegDiff** | 2D mask → image, pixel DPM | image channels × 256×256 | ACDC, Polyps | [`README`](nam/diffusion/2D_M2I/segdiff/README.md) | [GitHub](https://github.com/mazurowski-lab/segmentation-guided-diffusion) |
| **DiffBoost** | 2D mask → image, latent | 4×32×32 | ACDC, Synapse, Polyps | [`README`](nam/diffusion/2D_M2I/diffboost/README.md) | [GitHub](https://github.com/NUBagciLab/DiffBoost) |
| **FairDiff** | 2D mask → image, latent | 4×32×32 | Synapse | [`README`](nam/diffusion/2D_M2I/fairdiff/README.md) | [GitHub](https://github.com/wenyi-li/FairDiff) |
| **SiameseDiff** | 2D mask → image, SD latent | 4×32×32 | Polyps | [`README`](nam/diffusion/2D_M2I/siamesediff/README.md) | [GitHub](https://github.com/Qiukunpeng/Siamese-Diffusion) |
| **JoDiffusion** | joint image + mask | 8×32×32 | ACDC | [`README`](nam/diffusion/2D_M%26I/jodiffusion/README.md) | [GitHub](https://github.com/00why00/JoDiffusion) |
| **MedSegFactory** | dual-stream image + mask | 4×32×32, dual noise | Synapse, Polyps | [`README`](nam/diffusion/2D_M%26I/medsegfactory/README.md) | [GitHub](https://github.com/jwmao1/MedSegFactory) |
| **VolDiT** | 3D mask → volume | 8×24×24×12 | LA, ImageCAS | [`README`](nam/diffusion/3D_M2I/voldit/README.md) | [GitHub](https://github.com/Cardio-AI/voldit) |
| **MAISI** | 3D mask → volume | 4×48×48×24 | LA, ImageCAS | [`README`](nam/diffusion/3D_M2I/maisi/README.md) | [GitHub](https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi) |
| **ControlNet-SDXL** | semantic mask → image | 2D latent | PASCAL VOC + SBD | [`README`](nam/diffusion/2D_M2I/controlnet_sdxl/README.md) | [ControlNet](https://github.com/lllyasviel/ControlNet) |
| **SD-v1.5 + LoRA** | text → image | 2D latent | PneumoniaMNIST, ISIC | [`README`](nam/diffusion/2D_T2I/sd15_lora/README.md) | [SD-v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |

Example upstream setup:

```bash
git clone https://github.com/Qiukunpeng/Siamese-Diffusion.git third_party/Siamese-Diffusion
```

---

<a id="downstream"></a>

## 🧠 Downstream Models

| 🧠 Model | 🎯 Task | 📐 Dim. | 🔗 Reference | 💾 Checkpoint role |
|---|---|---:|---|---|
| **nnU-Net** | medical segmentation | 2D / 3D | [GitHub](https://github.com/MIC-DKFZ/nnUNet) | main anchor / downstream |
| **Swin-Unet** | medical segmentation | 2D | [GitHub](https://github.com/HuCaoFighting/Swin-Unet) | transfer downstream |
| **SwinUNETR** | volumetric segmentation | 3D | [GitHub](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) | 3D downstream |
| **SAMed** | medical segmentation | 2D / slice-wise 3D eval | [GitHub](https://github.com/hitachinsk/SAMed) | transfer downstream |
| **DeepLabV3-R50** | natural-image segmentation | 2D | [Torchvision](https://pytorch.org/vision/stable/models/deeplabv3.html) | VOC transfer |
| **Mask2Former-R50** | natural-image segmentation | 2D | [GitHub](https://github.com/facebookresearch/Mask2Former) | VOC transfer |
| **ResNet-50** | medical classification | 2D | [Torchvision](https://pytorch.org/vision/stable/models/resnet.html) | classification |
| **ViT-S/16** | medical classification | 2D | [timm](https://github.com/huggingface/pytorch-image-models) | classification |

| Namespace | Path | Selection |
|---|---|---|
| Real baseline / NAM anchor | `nam/downstream/real_checkpoint/<dataset>/<model>/best.pt` | validation-selected |
| Synthetic-augmented | `nam/downstream/syn_checkpoint/<dataset>/<generator>/<model>/best.pt` | validation-selected |
| Resume state | corresponding `latest.pt` | latest training state |
| Periodic snapshots | `epoch_XXXX.pt` | archival |

Details: [`nam/downstream/README.md`](nam/downstream/README.md)

---

<a id="reproduction"></a>

## 🔁 Reproduction

<p>
  <img src="https://img.shields.io/badge/Reference-Polyps%20%2B%20SiameseDiff%20%2B%20nnU--Net-8250df?style=flat-square" alt="Reference experiment">
  <img src="https://img.shields.io/badge/Seeds-42%20%7C%203407%20%7C%202026-0969da?style=flat-square" alt="Paper seeds">
  <img src="https://img.shields.io/badge/Budget-Matched-1f883d?style=flat-square" alt="Matched synthetic budget">
</p>

### 🎯 Recommended first run: Polyps + SiameseDiff + nnU-Net

| Item | Default |
|---|---|
| Config | `configs/table1_2d.yaml` |
| Data | Polyps: `1128 / 161 / 323` train/val/test |
| Generator | SiameseDiff |
| Anchor / downstream | nnU-Net |
| Latent noise | 4×32×32 |
| NAM | 3,000 iterations; `beta=0.001`; `kappa_up=0.5`; 10-step rollout |
| Final synthesis | deterministic DDIM-50 |
| Synthetic budget | 1,128 |

```bash
# 1) data: follow nam/data/polyps/README.md
git clone https://github.com/Qiukunpeng/Siamese-Diffusion.git third_party/Siamese-Diffusion

# 2) train/register generator (skip if a compatible checkpoint is already available)
python scripts/train_diffusion_2d.py --config configs/table1_2d.yaml

# 3) real downstream baseline / frozen anchor
python scripts/train_downstream_2d.py --config configs/table1_2d.yaml --phase real

# 4) NAM
python scripts/train_nam_2d.py --config configs/table1_2d.yaml

# 5) matched synthesis
python scripts/generate_2d.py --config configs/table1_2d.yaml --method nam
python scripts/generate_2d.py --config configs/table1_2d.yaml --method base

# 6) synthetic continuation
python scripts/train_downstream_2d.py --config configs/table1_2d.yaml --phase synthetic

# 7) evaluation
python scripts/evaluate_2d.py --config configs/table1_2d.yaml --checkpoint-phase real
python scripts/evaluate_2d.py --config configs/table1_2d.yaml --checkpoint-phase syn
python scripts/evaluate_adversariality.py --config configs/table1_2d.yaml
python scripts/evaluate_fid.py --config configs/table1_2d.yaml
```

Expected default checkpoint locations:

| Component | Path |
|---|---|
| SiameseDiff task checkpoint | `nam/diffusion/2D_M2I/siamesediff/checkpoints/diffusion/polyps/best_fid.ckpt` |
| Real nnU-Net anchor | `nam/downstream/real_checkpoint/polyps/nnunet/best.pt` |
| NAM miner | configuration-dependent, typically `outputs/<experiment>/checkpoints/nam_latest.pt` |
| Synthetic downstream | `nam/downstream/syn_checkpoint/polyps/siamesediff/nnunet/best.pt` |

### 🧪 Workflow profiles

The operation order is shared across branches: **generator → real baseline → NAM → Base/NAM synthesis → synthetic continuation → evaluation**.

| Profile | Config | NAM / generation | Downstream / evaluation |
|---|---|---|---|
| 2D medical | `<2D_CONFIG>` | `train_nam_2d.py`, `generate_2d.py` | `train_downstream_2d.py`, `evaluate_2d.py` |
| 3D medical | `configs/table1_3d.yaml` | `train_nam_3d.py`, `generate_3d.py` | `train_downstream_3d.py`, `evaluate_3d.py` |
| Natural-image | `configs/controlnet_sdxl_voc.yaml` | 2D scripts | `evaluate_2d.py` |
| PneumoniaMNIST | `configs/sd15_lora_pneumoniamnist.yaml` | 2D scripts | `evaluate_classification.py` |
| ISIC | `configs/sd15_lora_isic.yaml` | 2D scripts | `evaluate_classification.py` |

### 📋 Main Table I protocol

| Control | Setting | Control | Setting |
|---|---|---|---|
| Generator | same task-specific checkpoint | NAM optimization | 3,000 iterations |
| Final sampler | deterministic DDIM-50 | Synthetic budget | matched to real train size |
| Real : synthetic | `1 : 1` | CutMix | `p=0.5` |
| Checkpoint selection | validation only | Final metrics | held-out test |
| Paper seeds | `42`, `3407`, `2026` | Full protocol | [`docs/TABLE1_PROTOCOL.md`](docs/TABLE1_PROTOCOL.md) |

---

<a id="evaluation"></a>

## 📊 Evaluation & Analysis

<p>
  <img src="https://img.shields.io/badge/Metrics-DSC%20%7C%20ASD%20%7C%20mIoU%20%7C%20Acc-1f883d?style=flat-square" alt="Task metrics">
  <img src="https://img.shields.io/badge/Synthesis-FID%20%7C%20Adversariality-0969da?style=flat-square" alt="Synthesis metrics">
  <img src="https://img.shields.io/badge/Analysis-t--SNE%20%7C%20Cross--Model-8250df?style=flat-square" alt="Analysis tools">
</p>

| 📊 Analysis | ▶️ Command / module | 📤 Main output | 🧪 Reproduction note |
|---|---|---|---|
| 2D segmentation | `scripts/evaluate_2d.py` | DSC / ASD | select real or synthetic checkpoint |
| 3D segmentation | `scripts/evaluate_3d.py` | volumetric metrics | use 3D config |
| Classification | `scripts/evaluate_classification.py` | accuracy + configured metrics | PneumoniaMNIST / ISIC |
| Adversariality | `scripts/evaluate_adversariality.py` | CSV + JSON summary | larger normalized score = harder |
| FID | `scripts/evaluate_fid.py` | feature-distance JSON | keep encoder/splits/budget fixed |
| t-SNE | `python -m nam.evaluation.Vis.tsne` | features + PDF/PNG | joint projection |
| Adversariality distribution | `python -m nam.evaluation.Vis.adversariality_distribution` | plots | exported sample scores |
| Cross-model consistency | `python -m nam.evaluation.Vis.cross_model_consistency` | Pearson / Spearman / MAD | align by exact sample `id` |

Adversariality exports may contain `id`, `path`, raw `lce/lcbce/ldice`, normalized `adv_*`, and selected `adv`. Example reward override:

```bash
python scripts/train_nam_2d.py --config configs/table1_2d.yaml --set training.reward=lcbce
```

For FID/Base/NAM comparisons keep the **feature extractor, real split, synthetic budget, preprocessing, and output resolution** fixed. Full analysis notes: [`docs/EVALUATION_VISUALIZATION.md`](docs/EVALUATION_VISUALIZATION.md)

---

<a id="mitigation"></a>

## 🛡️ Mitigation Strategies

High-adversariality mining can expose defective modes already present in the base generator. Mitigation is optional and applied after NAM training while generator, anchor, and miner remain frozen.

| Strategy | Main idea | Requirement | Location |
|---|---|---|---|
| **HAT** | replace samples above calibrated adversariality threshold; retry seeds | adversariality scoring | `nam/mitigation/hat.py` |
| **QSF** | VQA-based quality / semantic filtering | VQA model + optional deps | `nam/mitigation/qsf.py` |
| **LSRS** | rerank using full/unconditional/component-conditioned predictions | cached states / hooks | `nam/mitigation/lsrs.py` |
| **ASG** | suppress incompatible generations with cross/self-attention signals | attention access | `nam/mitigation/asg.py` |

Config: `configs/mitigation.yaml` · Guide: [`nam/mitigation/README.md`](nam/mitigation/README.md)

---

<a id="artifacts"></a>

## 💾 Checkpoints, Outputs & Paper-to-Code Map

### Paths and artifacts

| Item | Default / pattern | Item | Default / pattern |
|---|---|---|---|
| Real downstream | `nam/downstream/real_checkpoint/<dataset>/<model>/best.pt` | Synthetic downstream | `nam/downstream/syn_checkpoint/<dataset>/<generator>/<model>/best.pt` |
| NAM miner | `outputs/<experiment>/checkpoints/nam_latest.pt` | TensorBoard | `outputs/**/tensorboard/` |
| Metrics | `metrics.jsonl` | Resolved config | `config.json` |
| Environment | `environment.json` | Sample metadata | `samples.jsonl` |
| Visualizations | `visualizations/` | Large weights/data | excluded from Git |

```bash
tensorboard --logdir outputs --port 6006
```

Keep `config.json`, `environment.json`, code commit, seed, upstream generator revision, checkpoint IDs, and synthetic budget for reproducible comparisons.

### 🗺️ Paper-to-code

| Paper component | Entry | Paper component | Entry |
|---|---|---|---|
| 2D reference | `configs/table1_2d.yaml` | 3D reference | `configs/table1_3d.yaml` |
| Table I matrix | `configs/table1_matrix.yaml` | Table I notes | `docs/TABLE1_PROTOCOL.md` |
| NAM training | `scripts/train_nam_2d.py`, `train_nam_3d.py` | Base/NAM sampling | `scripts/generate_2d.py`, `generate_3d.py` |
| Downstream training | `scripts/train_downstream_*.py` | Segmentation eval | `scripts/evaluate_2d.py`, `evaluate_3d.py` |
| Classification | `scripts/evaluate_classification.py` | Adversariality | `scripts/evaluate_adversariality.py` |
| FID | `scripts/evaluate_fid.py` | Visualization guide | `docs/EVALUATION_VISUALIZATION.md` |
| t-SNE | `nam/evaluation/Vis/tsne.py` | Cross-model | `nam/evaluation/Vis/cross_model_consistency.py` |
| Split inventory | `nam/data/SPLITS.md` | Mitigation | `configs/mitigation.yaml`, `nam/mitigation/` |

---

<a id="checklist"></a>

## ✅ Reproducibility Checklist

| Category | Verify |
|---|---|
| **Data** | unchanged train/val/test manifests · case-level partitioning · resolution/label mapping · no target-domain leakage |
| **Generator** | same checkpoint · sampler · condition list · CFG/prediction type · upstream revision |
| **NAM** | frozen generator/anchor · correct 2D/3D miner · `beta/kappa_up/reward/truncated_steps` · matching miner checkpoint |
| **Synthetic data** | Base/NAM budget matched · stable IDs/seeds · no unreported filtering |
| **Downstream** | same real initialization · architecture-specific optimizer · same continuation schedule · validation-only selection |
| **Evaluation** | held-out test metrics · same FID encoder/preprocessing · same adversariality anchor · cross-model match by `id` · intended seed aggregation |

---

<a id="extending"></a>

## 🧰 Extending the Repository

| Extension | Minimum interface / files | Reproducibility notes |
|---|---|---|
| Dataset | `README.md`, `dataset.py`, train/val/test manifests, `build_dataset` factory | document source, preprocessing, labels, split, resolution, layout |
| Diffusion generator | model/checkpoint loading, noise layout, conditioning, sampling, early denoising signal, NAM integration | reuse an adapter with a similar synthesis paradigm |
| Downstream model | model, real training, synthetic continuation, validation selection, evaluation | NAM needs a frozen loss/adversariality-producing checkpoint |
| Reward | downstream hardness signal with **larger = harder** convention | keep normalization/direction consistent |
| Mitigation hooks | state / attention access when needed | required mainly for LSRS / ASG |

---

<a id="baselines"></a>

## ⚖️ Comparison Methods

<p>
  <img src="https://img.shields.io/badge/Families-Random%20%7C%20Heuristic%20%7C%20Diversity%20%7C%20Utility%20%7C%20Adversarial-6f42c1?style=flat-square" alt="Comparison families">
  <img src="https://img.shields.io/badge/Protocol-Frozen%20DM-0969da?style=flat-square" alt="Frozen DM protocol">
  <img src="https://img.shields.io/badge/Budget-Matched-1f883d?style=flat-square" alt="Matched budget">
</p>

We compare NAM with **random sampling**, **heuristic targeting**, **diversity-oriented augmentation**, **utility-aware generation**, and **adversarial guidance**. The last column gives the configuration used in our unified comparison; some methods require minimal adaptation to satisfy the shared frozen-DM / from-scratch synthesis protocol.

### 📚 Baselines and configurations

| 🧭 Family | 🧪 Method | 📄 Paper | 💻 Code | 💡 Core mechanism | ⚙️ Configuration used in our experiments |
|---|---|---|---|---|---|
| 🎲 Random | **Base** | — | — | Standard diffusion sampling | DDIM-50; `σ=0`; Gaussian noise; no guidance/filtering |
| 🎯 Heuristic | **UGDM** | [Paper](https://doi.org/10.1109/TPAMI.2024.3399098) | [GitHub](https://github.com/huanlemin/UGDM) | Uncertainty measurement guidance | `γ=3`; DDIM inversion removed |
| 🎯 Heuristic | **VGD** | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/38246) | — | Value-guided high-utility/boundary synthesis | DDIM-100; `σ=0.2`; `λ=1`, `τ=1`, `k=0.03`, `γ=0.90` |
| 🌈 Diversity | **AugPaint** | [Paper](https://arxiv.org/abs/2506.23038) | — | Task-aware inpainting | foreground masking + diffusion reconstruction |
| 🌈 Diversity | **DiffAug** | [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/24e8b46430df965674221665816a4964-Abstract-Conference.html) | — | Partial diffuse-and-denoise | Base sample; `t ~ Beta(2,4)·T` |
| 🌈 Diversity | **CIG / Diff-II** | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Inversion_Circle_Interpolation_Diffusion-Based_Image_Augmentation_for_Data-Scarce_Classification_CVPR_2025_paper.html) | [GitHub](https://github.com/scuwyh2000/Diff-II) | Circle interpolation | `"interp"`; real ref + forward noise to `T`; concept learning removed |
| 🌈 Diversity | **DA-Fusion** | [Paper](https://arxiv.org/abs/2302.07944) | [GitHub](https://github.com/brandontrabucco/da-fusion) | SDEdit + Mixup | `t₀/T∈{.25,.5,.75,1}`; Mixup `α=.5` |
| 📈 Utility | **GAL** | [Paper](https://proceedings.mlr.press/v235/zhu24b.html) | [GitHub](https://github.com/aim-uofa/DiverGen) | Offline utility filtering | threshold `τ=-0.05`; repeat until budget |
| 📈 Utility | **UtilGen-lite** | [Paper](https://papers.neurips.cc/paper_files/paper/2025/hash/2ea07a4acbf7e38913062fd69a70805f-Abstract-Conference.html) | — | Utility estimator + prompt/noise optimization | 1-hidden-layer MLP; MLCO removed; CFG `5.5/0` |
| ⚔️ Adv. guidance | **AdvDiffuser** | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_AdvDiffuser_Natural_Adversarial_Example_Synthesis_with_Diffusion_Models_ICCV_2023_paper.html) | — | Diffusion guidance + PGD | `T=50`; `η=.1`; `I=1` |
| ⚔️ Adv. guidance | **P2P** | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Medghalchi_Prompt2Perturb_P2P_Text-Guided_Diffusion-Based_Adversarial_Attack_on_Breast_Ultrasound_Images_CVPR_2025_paper.html) | [GitHub](https://github.com/moeinheidari7829/P2P) | Text-embedding adversarial optimization | `ε=.05`; AdamW `1e-5`; 500 iters |
| ⚔️ Adv. guidance | **Diff-PGD** | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/088463cd3126aef2002ffc69da42ec59-Abstract-Conference.html) | [GitHub](https://github.com/xavihart/Diff-PGD) | SDEdit + PGD | Base synthetic input; official/default attack parameters |
| ⚔️ Adv. guidance | **DiffAttack** | [Paper](https://arxiv.org/abs/2305.08192) | [GitHub](https://github.com/WindVChen/DiffAttack) | Latent attack + attention preservation | DDIM-20; start 15; 30 iters; AdamW `1e-2`; guidance `2.5`; weights `10/10000/100` |
| ⚔️ Adv. guidance | **NatADiff** | [Paper](https://arxiv.org/abs/2505.20934) | [GitHub](https://github.com/maxcollins1999/NatADiff) | Boundary guidance + time travel | DDIM-100; `ω=7.5`,`ρ=7.5`,`μ=.2`; `R=5`; `r_l/r_u=500/800`; `[c_l,c_u]=[0,700]`; `S=5`; `s=50` |
| ⛏️ Native | **NAM (ours)** | TPAMI extended manuscript | [GitHub](https://github.com/JackCD99/Native-Adversariality-Mining) | Seed-level native hard-mode mining | frozen DM/anchor; AdamW `1e-4`; 3k iters; `β=.001`; `κ_up=.5`; rollout 10; DDIM-50 |

> [!TIP]
> **DiffAug** here is *DiffAug: A Diffuse-and-Denoise Augmentation for Training Robust Classifiers* (NeurIPS 2024).

### 🧪 Unified protocol & adaptations

| Component | Setting | Component | Setting |
|---|---|---|---|
| Diffusion backbone | same task-specific frozen DM | Synthetic budget | real training-set size |
| Primary comparison seed | `42` | Conditions | same masks/prompts/labels |
| Downstream initialization | same real checkpoint | Real : synthetic | `1:1` |
| CutMix | `p=0.5` | Training schedule | identical per architecture |
| Checkpoint selection | validation only | Final evaluation | held-out test |
| Hyperparameters | official when valid | Tuning | minimal validity-only tuning |

| Method | Adaptation | Reason |
|---|---|---|
| **UGDM** | remove DDIM inversion | direct generation from initial noise |
| **CIG / Diff-II** | remove concept learning; replace inversion with forward noising to `T` | avoid method-specific adaptation; common generation protocol |
| **DiffAttack** | remove DDIM inversion | direct frozen-DM synthesis |
| **UtilGen-lite** | remove MLCO DM retraining; retain utility estimator + prompt/noise optimization | enforce frozen-DM comparison |

> [!IMPORTANT]
> We call the modified utility baseline **UtilGen-lite** because the original MLCO diffusion-model adaptation stage is excluded.

---

<a id="limitations"></a>

## ⚠️ Practical Limitations

| Limitation | Practical implication |
|---|---|
| **Generator alignment** | if the base DM poorly represents the target distribution, higher adversariality may not improve generalization |
| **Miner optimization cost** | NAM avoids DM retraining but still optimizes through a truncated diffusion path |
| **Defective-mode exposure** | hard generator defects can receive high downstream loss; optional mitigation may be needed |
| **Upstream dependency complexity** | full-matrix reproduction depends on several large external projects/checkpoints |
| **Metric sensitivity** | FID, adversariality, validity, and downstream gain measure different properties and should be interpreted together |

---

<a id="faq"></a>

## ❓ FAQ & Troubleshooting

| Question / symptom | Check / answer |
|---|---|
| Is NAM training-free? | **No.** The miner is trained; the diffusion generator remains frozen. |
| Must I train the diffusion model from scratch? | No, if a compatible task checkpoint is available and placed as documented by the adapter. |
| What is the downstream anchor? | A real-data-trained, frozen downstream model that provides the difficulty signal. |
| What does Base mean? | Standard sampling from the same frozen generator with random initial noise and no NAM reselection. |
| Does `--dry-run` test checkpoints/data? | No; it validates configuration and imports without loading large data/weights. |
| `best.pt` is missing | finish real-data training/validation first. |
| Dataset file not found | verify dataset README, `dataset.root`, manifest-relative paths, preprocessing, and ignored `data/` directory. |
| Diffusion checkpoint mismatch | verify upstream revision, model config, channels, scheduler/prediction type, filename, resolution. |
| NAM shape mismatch | match miner spatial dims/noise layout to the generator (e.g. 4×32×32 vs 3D latent tensors). |
| CUDA OOM | reduce batch size first; for 3D also inspect volume size, mixed precision, and visualization frequency. |
| FID differs strongly | match feature encoder, preprocessing/resolution, real split, and synthetic budget. |
| Base/NAM downstream results incomparable | match real initialization, budget, continuation epochs, augmentation, validation rule, and seeds. |
| Cross-model CSV order differs | align by exact sample `id`, not row position. |
| Reporting an issue | include command, config, `config.json`, `environment.json`, checkpoint IDs, dataset/preprocessing stage, and full traceback; never upload private medical data. |

> [!WARNING]
> Batch size may be reduced for memory, but do **not** silently reduce the final synthetic budget in a Base-vs-NAM comparison.

---

<a id="development"></a>

## 🧑‍💻 Development & Contributing

| Topic | Recommendation |
|---|---|
| Git contents | keep raw datasets, generated datasets, `*.pt/*.pth/*.ckpt/*.safetensors`, TensorBoard folders, large outputs, and third-party repos out of Git |
| Portability | prefer dataset-relative/config-driven paths over machine-specific absolute paths |
| Sanity check | `python scripts/train_nam_2d.py --config <CONFIG> --dry-run --print-config` |
| Provenance | record code commit, config, seed, upstream revision, checkpoints, environment, budget, metrics |
| Good contributions | reproducibility fixes, setup notes, new adapters/models, split-preserving loader fixes, evaluation/visualization, docs |
| PR description | state affected experiment/component, numerical behavior changes, tested config, and new dependencies |

Issues and pull requests are welcome. For substantial changes, opening an issue first helps align expected experiment behavior and compatibility.

---

<a id="publications"></a>

## 📚 Publications

| Venue | Publication |
|---|---|
| 📘 **TPAMI extended manuscript** | **Mining Native Adversariality in Diffusion Models for Medical Generalization** — Hongyu Zhang, Haipeng Chen, Zhimin Xu, Chengxin Yang, Yingda Lyu. Submitted to *IEEE TPAMI*, 2026. |
| 🌟 **CVPR 2026 Highlight** | **Diffusion-Based Native Adversarial Synthesis for Enhanced Medical Segmentation Generalization** — Hongyu Zhang, Haipeng Chen, Zhimin Xu, Chengxin Yang, Yingda Lyu. *CVPR*, 2026. |

---

<a id="citation"></a>

## 📝 Citation

If NAM or this repository is useful for your research, please cite the relevant paper and the original generator/downstream/dataset used in your reproduced experiment.

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

---

<a id="license"></a>

## 🙏 Acknowledgements & License

This project builds on open-source work in diffusion models, medical imaging, segmentation, classification, and evaluation. Upstream repositories are linked above; datasets, pretrained weights, external checkpoints, and third-party code remain subject to their original licenses and terms.

The NAM code is released under the [Apache License 2.0](LICENSE).

<div align="center">

---

### ⭐ If NAM supports your research, consider starring the repository.

<sub><b>Native Adversariality Mining</b> · seed-level hard-mode mining for diffusion-driven downstream generalization</sub>

<br>

<a href="https://github.com/JackCD99/Native-Adversariality-Mining/stargazers"><img src="https://img.shields.io/github/stars/JackCD99/Native-Adversariality-Mining?style=social" alt="GitHub stars"></a>

</div>
