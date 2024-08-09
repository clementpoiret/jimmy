# Draws inspiration from Timm's ModelEmaV2
# <https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py#L83>

from copy import deepcopy

import jax
from flax import nnx


class EmaModel(nnx.Module):
    model: nnx.Module
    decay: float = 0.9999

    def __init__(self, model: nnx.Module, decay: float = 0.9999):
        super().__init__()
        self.model = deepcopy(model)
        self.decay = decay
        self.ema_params = None

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def initialize_ema(self, params):
        self.ema_params = jax.tree.map(lambda x: x.copy(), params)

    def update(self, params):
        if self.ema_params is None:
            self.initialize_ema(params)
        else:
            self.ema_params = jax.tree.map(
                lambda ema, new: self.decay * ema + (1.0 - self.decay) * new,
                self.ema_params,
                params,
            )

        nnx.update(self.model, self.ema_params)

    def set(self, params):
        self.ema_params = jax.tree.map(lambda x: x.copy(), params)

    def get_ema_params(self):
        return self.ema_params
