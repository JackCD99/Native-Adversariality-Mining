# Adversariality Evaluation and Visualization

## Fixed-budget evaluation

`scripts/evaluate_adversariality.py` evaluates the configured synthetic dataset
in deterministic dataset order. It only truncates the dataset to the requested
budget; it does not resize, perturb, filter, or augment stored samples. The
downstream checkpoint is frozen and used only for inference.

```bash
python scripts/evaluate_adversariality.py
python scripts/evaluate_adversariality.py --proxy lce
python scripts/evaluate_adversariality.py --config configs/table1_3d.yaml --spatial-dims 3
```

The CSV contains `id`, `path`, raw `lce`, `lcbce`, and `ldice`, their normalized
`adv_*` scores, the selected `adv` score, and an audit Dice value. CE losses are
mapped with `1-exp(-loss)`; Dice loss is already bounded. Consequently all
`adv_*` values share the convention that a larger value means a harder sample.
The companion JSON records `mean_adv`, every proxy mean, budget, split, model,
and output path.

## Publication figures

Install the optional plotting dependencies once:

```bash
pip install -e ".[visualization]"
```

Then run the figure modules from the repository root:

```bash
python -m nam.evaluation.Vis.tsne
python -m nam.evaluation.Vis.adversariality_distribution
python -m nam.evaluation.Vis.cross_model_consistency
```

The t-SNE module hooks the frozen downstream encoder bottleneck and stores both
the pooled source features and the projected coordinates. All configured groups
are projected jointly with one seed. It uses no synthetic feature interpolation,
noise, dropout, point removal, or coordinate manipulation.

Cross-model consistency aligns CSV files by exact `id` intersection rather than
row position. It reports Pearson correlation, Spearman correlation, and mean
absolute difference. Every figure is saved as vector PDF and review PNG.
