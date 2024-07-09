from typing import Callable

import jax.numpy as jnp
from flax import nnx


class Mlp(nnx.Module):
    """Simple MLP for Vision Transformers."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable = nnx.gelu,
        dropout_rate: float = 0.0,
        bias: bool = True,
        rngs: nnx.Rngs = None,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nnx.Linear(in_features,
                              hidden_features,
                              use_bias=bias,
                              rngs=rngs)
        self.act = act_layer
        self.drop1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features,
                              out_features,
                              use_bias=bias,
                              rngs=rngs)
        self.drop2 = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x
