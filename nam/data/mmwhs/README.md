# MMWHS

The Multi-Modality Whole Heart Segmentation challenge contains cardiac CT and MRI. Access and redistribution are governed by the [official MMWHS page](https://zmiclab.github.io/zxh/0/mmwhs/); do not commit raw images.

## NAM preparation

Process CT and MRI independently at patient level. Convert each volume to axial slices, discard slices without foreground, resize to 256 x 256, and map the official sparse labels `{0,205,420,500,550,600,820,850}` to contiguous IDs 0-7 for background, AA, LA, LV, myocardium, PA, RA, and RV. The paper reports CT 3,714/531/1,060 and MRI 2,029/290/579 slices.

Use `ct_train.list`, `ct_val.list`, `ct_test.list`, and their `mri_*` counterparts. `train.list`, `val.list`, and `test.list` document the combined inventory and may concatenate both modalities. Set `dataset.modality` to `CT` or `MRI`; set `raw_labels: true` only if masks still use the official sparse IDs.

Factory: `nam.data.mmwhs.dataset:build_dataset`.
