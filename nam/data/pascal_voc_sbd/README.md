# PASCAL VOC 2012 + Semantic Boundaries Dataset

Download [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
and the [Semantic Boundaries Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

## Preparation

Combine the VOC semantic-segmentation split with SBD, remove duplicate image
identifiers, and convert SBD MATLAB annotations to single-channel PNG or NumPy
label maps. Labels use the standard VOC IDs `0..20`; `255` remains the ignored
boundary label. The fixed experiment split contains 12,031 RGB images:

- 8,422 training images;
- 1,203 validation images;
- 2,406 test images.

Images and masks are resized to 512 x 512 with bilinear and nearest-neighbor
interpolation, respectively. Training applies synchronized geometric and color
augmentation. Prompts follow `a photo of [VOC object]` and list the foreground
classes present in each conditioning mask.

Place images and masks under `data/`. Every manifest row uses:

```text
sample_id relative/image/path relative/mask/path [optional prompt]
```

Factory: `nam.data.pascal_voc_sbd.dataset:build_dataset`.

The VOC and SBD licenses and terms of use apply to the downloaded data.
