# Publication datasets

Every benchmark is self-contained under one package. Raw or processed data are intentionally excluded; place them under that package's `data/` directory and populate `train.list`, `val.list`, and `test.list` using relative paths. `test_prompts.jsonl` records class/anatomy prompts without binding them to private files.

The segmentation list schema is:

```text
sample_id relative/image/path relative/target/path [optional prompt]
```

The classification list schema is:

```text
sample_id relative/image/path integer_class_id [optional prompt]
```

JSONL rows may instead use `sample_id`, `image`, `target`, `class_id`, and `prompt`. Every `dataset.py` exposes `build_dataset(config, split, spatial_dims)` and `build_dataloader(...)`. Returned samples follow the canonical NAM fields: `image`, `target`, `condition`, `sample_id`, and `metadata`.

Patient-level split membership must be decided before extracting slices or patches. Training transforms are synchronized for images and masks; validation/test are deterministic. Natural-image PASCAL VOC + SBD is intentionally deferred.
