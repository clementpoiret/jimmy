from typing import Optional

import jax.numpy as jnp
from flax import nnx
from jax import lax


class RMSNormGated(nnx.Module):
    eps: float = 1e-5

    def __init__(self, d: int, rngs: nnx.Rngs, **kwargs):
        self.__dict__.update(**kwargs)

        self.w = nnx.Param(nnx.initializers.ones(rngs.params(), [d]))

    def __call__(self, x: jnp.ndarray, z: Optional[jnp.ndarray] = None):
        if z is not None:
            x *= z

        y = x.astype(jnp.float32)
        norm = y * lax.rsqrt(jnp.mean(y * y, -1, keepdims=True) + self.eps)

        return self.w.value * norm.astype(x.dtype)


class LayerScale(nnx.Module):
    """Layer scale module for scaling the output of a layer."""

    init_values: float = 1e-5

    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        """Initialize the LayerScale module.

        Args:
            dim (int): The dimension of the input.
            init_values (float, optional): Initial value for scaling. Defaults to 1e-5.
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        self.__dict__.update(**kwargs)

        self.gamma = nnx.Param(
            self.init_values * nnx.initializers.ones(rngs.params(), [dim]), )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply layer scaling to the input.

        Args:
            x (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Scaled output.
        """
        return x * self.gamma
