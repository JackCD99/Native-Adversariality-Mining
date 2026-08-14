# ACDC

ACDC is the cine-MRI cardiac segmentation benchmark used for the 2D experiments. Download it from the [official ACDC platform](https://www.creatis.insa-lyon.fr/Challenge/acdc/). The task has four labels: background, right ventricle, myocardium, and left ventricle.

## NAM preparation

Split patients with seed 42 at a 7:1:2 ratio before extracting slices. Use the end-diastolic and end-systolic annotated frames, convert each volume to axial slices, discard slices without foreground, resize images and labels to 256 x 256, and preserve label IDs 0-3. The expected paper statistics are 4,010/572/1,146 train/validation/test slices.

Store either paired files or HDF5 files under `data/`. A list row is `sample_id image_path target_path [prompt]`; an HDF5 row may repeat the same path because `dataset.py` reads the `image` and `label` arrays separately.

```text
patient001_frame01_slice001 data/slices/patient001_frame01_slice001.h5 data/slices/patient001_frame01_slice001.h5
```

Factory: `nam.data.acdc.dataset:build_dataset`.
