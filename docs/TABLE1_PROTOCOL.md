# Table I protocol

Table I uses five datasets, eight diffusion pipelines, and three downstream
architectures. Every experiment cell follows the same dependency order:

1. train the generator and downstream baseline from the same real training split;
2. freeze the generator and the nnU-Net anchor;
3. optimize NAM for 3,000 iterations;
4. synthesize a fixed budget with deterministic DDIM-50;
5. continue each downstream baseline with matched real and synthetic batches;
6. evaluate the converged checkpoint on the held-out test split.

NAM uses AdamW with learning rate `1e-4`, weight decay `1e-2`, KL weight
`beta=0.001`, adversariality cap `kappa_up=0.5`, and a ten-step truncated
rollout. Downstream continuation uses a 1:1 real-to-synthetic ratio and paired
CutMix with probability 0.5.

The real-data optimizers remain architecture-specific:

- nnU-Net v2 uses Nesterov SGD, polynomial decay, Dice plus cross entropy, and
  deep supervision;
- Swin-Unet uses SGD and polynomial decay, while SwinUNETR uses AdamW,
  warmup-cosine scheduling, and sliding-window validation;
- SAMed uses rank-4 LoRA, AdamW, warmup-polynomial decay, and its native decoder
  loss resolution.

`configs/table1_matrix.yaml` is the canonical dataset-generator-downstream-seed
matrix. `python scripts/list_table1.py` prints every cell and its executable base
configuration. Dataset-specific fields such as `factory`, `root`, `num_classes`,
prompts, budget, and checkpoint paths are set together for each run.

Checkpoint selection uses the validation split. DSC and ASD are reported only
on the held-out test split. FID validation and final alignment reporting follow
the split definitions recorded in the experiment configuration.
