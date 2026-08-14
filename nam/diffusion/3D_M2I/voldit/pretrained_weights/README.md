# Official VolDiT weights

Download the public model bundle with:

```bash
hf download AICM-HD/voldit --local-dir pretrained_weights/voldit
```

The default configuration expects `vqgan_ds8.pth` and `dit_ds8_l4.pth` here. Verify filenames against the downloaded model card and update the YAML if the publisher changes archive names. Do not commit weight binaries to Git.
