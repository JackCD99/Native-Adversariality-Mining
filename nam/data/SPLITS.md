# Dataset split inventories

All manifests use portable paths relative to their dataset package. The split
sizes follow Table C.1 of the supplementary material. Dataset partitions that
were available in the experiment workspace retain their anonymized
sample identifiers. Missing validation partitions and unavailable processed
inventories were generated deterministically with seed 42.

| Dataset | Train | Validation | Test | Unit |
|---|---:|---:|---:|---|
| ACDC | 4,010 | 572 | 1,146 | foreground axial slices |
| Synapse | 2,645 | 378 | 756 | foreground axial slices |
| Polyps | 1,128 | 161 | 323 | image-mask pairs |
| EndoScene | - | - | 60 | evaluation-only pairs |
| CVC-ColonDB | - | - | 380 | evaluation-only pairs |
| ETIS | - | - | 196 | evaluation-only pairs |
| MMWHS CT | 3,714 | 531 | 1,060 | axial slices |
| MMWHS MRI | 2,029 | 290 | 579 | axial slices |
| LA | 70 | 10 | 20 | 3D volumes |
| ImageCAS | 700 | 100 | 200 | 3D volumes |
| PASCAL VOC + SBD | 8,422 | 1,203 | 2,406 | image-mask pairs |
| PneumoniaMNIST-224 | 4,099 | 585 | 1,171 | classified images |
| ISIC | 1,925 | 275 | 550 | classified images |

## Manifest format

Segmentation rows use:

```text
sample_id relative/image/path relative/target/path [optional prompt]
```

Classification rows use:

```text
sample_id relative/image/path class_id [optional prompt]
```

MMWHS provides `ct_{train,val,test}.list` and
`mri_{train,val,test}.list`. The unprefixed manifests concatenate the two
modalities for inventory inspection; the dataset factory selects the
modality-specific files at runtime.

EndoScene, ColonDB, and ETIS are held-out cross-center test datasets. Their
`train.list` and `val.list` contain an explanatory comment rather than sample
rows so that the files are non-empty without introducing evaluation leakage.

## Data placement

The checked-in manifests define the expected post-processing filenames. Place
or export raw data under each package's ignored `data/` directory using these
relative paths. If the downloaded release uses different filenames, rename the
processed files during conversion rather than editing split membership.
