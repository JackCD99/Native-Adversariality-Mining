# ImageCAS

ImageCAS contains 1,000 coronary CTA volumes. Use the [official project](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT) and its linked download instructions.

## NAM preparation

Apply the paper's fixed 700/100/200 case split. Binarize all annotated coronary branches as foreground, robustly normalize CTA intensities, crop an ROI around the coronary annotation with a safety margin, and resample/pad paired volumes to 192 x 192 x 96. Do not derive split membership after patch extraction.

```text
10016975 data/10016975/img.nii.gz data/10016975/label.nii.gz
```

Factory: `nam.data.imagecas.dataset:build_dataset` with `spatial_dims: 3`.
