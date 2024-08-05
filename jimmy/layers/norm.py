from typing import Optional

import jax.numpy as jnp
from flax import nnx
from jax import lax


class RMSNormGated(nnx.Module):

    def __init__(self, d: int, eps: float = 1e-5, rngs: nnx.Rngs = None):
        self.eps = eps
        self.w = nnx.Param(nnx.initializers.ones(rngs.params(), [d]))

    def __call__(self, x: jnp.ndarray, z: Optional[jnp.ndarray] = None):
        if z is not None:
            x *= z

        y = x.astype(jnp.float32)
        norm = y * lax.rsqrt(jnp.mean(y * y, -1, keepdims=True) + self.eps)

        return self.w.value * norm.astype(x.dtype)
