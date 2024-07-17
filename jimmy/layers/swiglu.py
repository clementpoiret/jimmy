from typing import Callable

import jax.numpy as jnp
from flax import nnx


class SwiGLU(nnx.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) FFN block from Google Brain.

    This module implements the SwiGLU activation function, which is a variant of the GLU
    (Gated Linear Unit) that uses the Swish activation function.

    Args:
        in_features (int): Number of input features.
        hidden_features (int | None, optional): Number of hidden features. If None, set to in_features. Defaults to None.
        out_features (int | None, optional): Number of output features. If None, set to in_features. Defaults to None.
        act_layer (Callable, optional): Activation function to use. Defaults to nnx.gelu.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        rngs (nnx.Rngs, optional): Random number generators. Defaults to None.
    """

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
                              hidden_features,
                              use_bias=bias,
                              rngs=rngs)
        self.w3 = nnx.Linear(hidden_features // 2,
                             out_features,
                             use_bias=bias,
                             rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the SwiGLU module.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after applying SwiGLU.
        """
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nnx.silu(x1) * x2

        return self.w3(hidden)
