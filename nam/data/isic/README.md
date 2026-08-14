# ISIC-2017 Task 3

Download the image and diagnosis files from the [official ISIC Challenge archive](https://challenge.isic-archive.com/data/). The binary protocol follows Task 3: melanoma is malignant (1), while nevus and seborrheic keratosis are grouped as benign (0).

## NAM preparation

Combine the official images and labels, retain lesion IDs for patient-level grouping where available, and create the fixed 1,925/275/550 split. Resize to 256 x 256. Each list row is `sample_id image_path class_id [prompt]`. Training uses class-aware prompt variants and synchronized horizontal flipping. Raw data remain under the ignored `data/` directory.

```text
ISIC_0000000 data/images/ISIC_0000000.jpg 0 a dermoscopic image of a benign skin lesion
ISIC_0000002 data/images/ISIC_0000002.jpg 1 a dermoscopic image of a malignant skin lesion
```

Factory: `nam.data.isic.dataset:build_dataset`.
