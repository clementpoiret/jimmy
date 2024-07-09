from typing import Callable

import jax.numpy as jnp
from flax import nnx


class SwiGLU(nnx.Module):
    """SwiGLU FFN block from Google Brain."""

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

        self.w12 = nnx.Linear(in_features,
                              2 * hidden_features,
                              use_bias=bias,
                              rngs=rngs)
        self.w3 = nnx.Linear(hidden_features,
                             out_features,
                             use_bias=bias,
                             rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nnx.silu(x1) * x2

        return self.w3(hidden)
