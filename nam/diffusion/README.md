# Diffusion implementation layout

Methods are grouped by synthesis contract rather than architecture name:

- `2D_M2I/`: a two-dimensional mask conditions image generation.
- `3D_M2I/`: a volumetric mask conditions volume generation.
- `2D_M&I/`: image and mask are generated jointly.

Each method package owns its official-code adapter and, once specialized, its
`model.py`, `pre_training.py`, `sampling.py`, `NAM_training.py`, and `utils/`
implementation. Shared mathematical contracts remain in `base.py`; the public
`build_diffusion(config)` entry remains stable through `registry.py`.

Python identifiers cannot contain `&` or start with a digit. For this reason,
external code should not import category paths with a normal `from` statement.
Use the stable registry API:

```python
from nam.diffusion import build_diffusion

adapter = build_diffusion(config.diffusion)
```

The registry uses `importlib` internally and can load the requested physical
folder names without exposing this filesystem naming constraint to experiments.

The `2D_M2I/controlnet_sdxl` package contains the 512 x 512 natural-image
transfer branch: PASCAL VOC+SBD masks condition a task-tuned ControlNet on the
SDXL base model, and a latent 4-channel NAM miner performs seed reselection.
