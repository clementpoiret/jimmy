from typing import Callable

import jax.numpy as jnp
from flax import nnx


class Mlp(nnx.Module):
    """
    Simple MLP (Multi-Layer Perceptron) for Vision Transformers.

    This module implements a two-layer MLP with configurable hidden size,
    activation function, and dropout.

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
        """
        Forward pass of the MLP.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after passing through the MLP.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x
