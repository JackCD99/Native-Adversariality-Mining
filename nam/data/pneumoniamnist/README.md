# PneumoniaMNIST-224

Use the 224-resolution PneumoniaMNIST+ release from the [official MedMNIST website](https://medmnist.com/). The task groups bacterial and viral pneumonia into class 1 and uses normal as class 0.

## NAM preparation

Export images without changing official patient identities, then produce the paper's deterministic 4,099/585/1,171 split. Each manifest row is `sample_id image_path class_id [prompt]`; because this is classification, the third column is an integer class rather than a mask path. Images are loaded as RGB-compatible grayscale at 224 x 224. Training supports class-preserving prompt variants and optional caption dropout.

```text
000001 data/images/000001.png 0 a grayscale chest X-ray image of normal
000002 data/images/000002.png 1 a grayscale chest X-ray image of pneumonia
```

Factory: `nam.data.pneumoniamnist.dataset:build_dataset`.
