# Dataset interface

Each dataset package records its official download source and preprocessing
protocol. A configured factory receives `(config, split, spatial_dims)` and
returns a PyTorch `Dataset`. Every segmentation sample provides:
sample must provide:

```python
{
    "image": image_tensor_or_none,
    "target": integer_segmentation_mask,
    "condition": generator_specific_condition,
    "sample_id": stable_unique_identifier,
    "metadata": {"spacing": [sx, sy] or [sx, sy, sz]},
}
```

Use patient-level 70/10/20 splits created before preprocessing. For the 2D
experiments, remove foreground-free slices and resize to 256 x 256. For the 3D
experiments, crop the region of interest and normalize to 192 x 192 x 96. The
same split manifest must be shared by generator training, anchor training, miner
training, and final downstream evaluation.

The condition can be a tensor or a dictionary. The SiameseDiff bridge expects:

```python
condition = {
    "hint": mask_tensor,
    "txt": "a colonoscopy image of a polyp",
}
```

Classification datasets use a scalar integer `target` and class-aware prompt in
`condition`; their factories use the same function signature and collation
contract. Synthetic datasets store lossless tensor pairs and JSON provenance in
the method output directory. `nam.data.common:build_generated_dataset` reads
those pairs directly for downstream continuation.

PASCAL VOC+SBD uses the same segmentation contract with `ignore_index=255`, 21
class IDs, 512 x 512 tensors, and prompts derived from foreground classes in the
conditioning mask.
