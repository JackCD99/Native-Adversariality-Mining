# FairDiff pretrained weights

This directory stores external initialization weights only. Model weights are
not redistributed by this repository.

Expected layout:

```text
pretrained_weights/
└── control_sd15_seg.pth
```

The public FairDiff `MaskImageGen/train.py` references
`./models/control_sd15_seg.pth`, but the upstream Model Zoo is marked
"Coming Soon" and does not publish a verified download URL. Obtain this file
from the FairDiff authors or construct the segmentation ControlNet
initialization from legally acquired Stable-Diffusion v1.5 weights using the
upstream ControlNet procedure. Do not rename an unrelated checkpoint to make
the loader pass.

The default path is configured in `configs/fairdiff_2d.yaml`. An absolute path
can be supplied without modifying source code:

```bash
python -m nam.diffusion.2D_M2I.fairdiff.pre_training \
  --set fairdiff.pre_training.initialization_checkpoint=/path/to/control_sd15_seg.pth
```
