# Adapting an official diffusion repository

NAM keeps third-party source code outside this repository. Each diffusion file
under `nam/diffusion/` records the official source and creates a validated
adapter. The configured bridge is a small module that stays either in the
official checkout or under `nam/bridges/`.

The bridge factory has the following signature:

```python
def build_bridge(config, metadata):
    return Bridge(config, metadata)
```

The returned object must expose:

```python
class Bridge:
    model: torch.nn.Module

    def prepare_condition(self, batch): ...
    def initial_score(self, probe_noise, condition, cfg_scale): ...
    def truncated_rollout(self, initial_noise, condition, steps, cfg_scale): ...
    def sample(self, initial_noise, condition, steps, cfg_scale): ...
```

`initial_score` returns either `ScoreOutput` or a dictionary with `score`,
`timestep`, and optional `raw_prediction`. The score tensor must have the same
spatial layout as the selected seed. For epsilon-prediction models, use
`nam.diffusion.base.epsilon_to_score`. For v-prediction models, first call
`v_prediction_to_epsilon`.

`truncated_rollout` is the only delicate method. It must:

1. start at the first timestep of the full 50-step deterministic DDIM schedule;
2. execute only the first `steps` reverse updates;
3. evaluate the frozen score model on `x_t.detach()`;
4. detach the score prediction before the DDIM update;
5. retain the direct differentiable path from the selected seed to the clean
   estimate;
6. decode the clean estimate without `torch.no_grad()`.

M2I bridges return the decoded image tensor. M&I bridges return
`(decoded_image, decoded_target)` so the adversariality reward follows the
jointly generated image-mask pair instead of the input condition.

This implements the temporal stop-gradient approximation described in the
paper. `nam/diffusion/2D_M2I/siamesediff/model.py` is the concrete 2D reference.

For shared-noise M&I models, concatenate image and mask score channels and split
the generated joint tensor in `sample`. For dual-noise M&I models, concatenate
the two initial scores for conditioning, then return branch-specific noise
offsets through a bridge-specific packed tensor convention documented in that
adapter.

Pin every official checkout to a commit in the final release. Do not commit its
source or checkpoint unless its license explicitly allows redistribution.
