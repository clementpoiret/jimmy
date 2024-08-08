from typing import Callable

import jax.numpy as jnp
from flax import nnx


class Mlp(nnx.Module):
    """
    Simple MLP (Multi-Layer Perceptron) for Vision Transformers.

    This module implements a two-layer MLP with configurable hidden size,
    activation function, and dropout.

    Attributes:
        hidden_features (int | None): Number of hidden features. If None, set to in_features.
        out_features (int | None): Number of output features. If None, set to in_features.
        act_layer (Callable): Activation function to use.
        dropout_rate (float): Dropout rate.
        bias (bool): Whether to use bias in linear layers.

    Args:
        in_features (int): Number of input features.
        rngs (nnx.Rngs): Random number generators.
        **kwargs: Additional keyword arguments to override default attributes.
    """

    hidden_features: int | None = None
    out_features: int | None = None
    act_layer: Callable = nnx.gelu
    dropout_rate: float = 0.0
    bias: bool = True

    def __init__(
        self,
        in_features: int,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.__dict__.update(**kwargs)

        out_features = self.out_features or in_features
        hidden_features = self.hidden_features or in_features

        self.fc1 = nnx.Linear(
            in_features, hidden_features, use_bias=self.bias, rngs=rngs
        )
        self.act = self.act_layer
        self.drop1 = nnx.Dropout(self.dropout_rate, rngs=rngs)
        self.fc2 = nnx.Linear(
            hidden_features, out_features, use_bias=self.bias, rngs=rngs
        )
        self.drop2 = nnx.Dropout(self.dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the MLP.

        Args:
            x (jnp.ndarray): Input tensor of shape (B, ..., in_features).

        Returns:
            jnp.ndarray: Output tensor of shape (B, ..., out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x
