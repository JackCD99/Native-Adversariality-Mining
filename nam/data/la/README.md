# LA

The 2018 Atrial Segmentation Challenge provides 100 labeled 3D gadolinium-enhanced MRI scans. Download them from the [Cardiac Atlas challenge page](https://www.cardiacatlas.org/atriaseg2018-challenge/) and comply with its data terms.

## NAM preparation

Use the paper's fixed 70/10/20 patient split. Normalize each volume, find the foreground bounding box, extend it with a documented safety margin, crop the same ROI from image and mask, then resample/pad to 192 x 192 x 96. Preserve a binary mask. Store paired NIfTI, HDF5, NPY, NPZ, or PT files under `data/`.

```text
case001 data/images/case001.nii.gz data/labels/case001.nii.gz
```

Factory: `nam.data.la.dataset:build_dataset` with `spatial_dims: 3`.
