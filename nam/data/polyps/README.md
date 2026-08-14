# Polyps

This benchmark combines [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/) (612 frames) and [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) (1,000 frames), following SiameseDiff. Respect both datasets' research-use conditions.

## NAM preparation

Deduplicate by source ID, retain the source name in `sample_id`, perform the unified 7:1:2 split with seed 42, convert masks to `{0,1}`, and resize paired RGB images/masks to 256 x 256. The expected split has 1,128/161/323 samples. The loader also maps conventional 255-valued foreground masks to 1.

```text
kvasir_0001 data/images/kvasir_0001.jpg data/masks/kvasir_0001.png
clinicdb_0001 data/images/clinicdb_0001.tif data/masks/clinicdb_0001.tif
```

Factory: `nam.data.polyps.dataset:build_dataset`.
