import jax.numpy as jnp
from flax import nnx

from .configs import TransformerConfig


class Attention(nnx.Module):
    """
    Multi-head Attention module.

    This module implements multi-head attention mechanism as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        dim (int): The input and output dimension of the module.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value projections. Defaults to True.
        proj_bias (bool, optional): If True, add a learnable bias to output projection. Defaults to True.
        qk_norm (bool, optional): If True, apply layer normalization to query and key. Defaults to False.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.
        proj_drop (float, optional): Dropout rate for output projection. Defaults to 0.
        norm_layer (nnx.Module, optional): Normalization layer to use. Defaults to nnx.LayerNorm.
        rngs (nnx.Rngs, optional): Random number generators. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config

        self.qkv = nnx.Linear(
            config.dim, config.dim * 3, use_bias=config.qkv_bias, rngs=rngs
        )
        self.attn_drop = nnx.Dropout(config.attn_drop, rngs=rngs)
        self.proj = nnx.Linear(
            config.dim, config.dim, use_bias=config.proj_bias, rngs=rngs
        )
        self.proj_drop = nnx.Dropout(config.proj_drop, rngs=rngs)

        self.q_norm = (
            config.norm_layer(config.head_dim, rngs=rngs) if config.qk_norm else None
        )
        self.k_norm = (
            config.norm_layer(config.head_dim, rngs=rngs) if config.qk_norm else None
        )

    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the attention module.

        Args:
            x (jnp.ndarray): Input tensor of shape (B, N, C) where B is batch size,
                             N is sequence length, and C is input dimension.

        Returns:
            jnp.ndarray: Output tensor of shape (B, N, C).

        Raises:
            AssertionError: If input embedding dimension doesn't match layer embedding dimension.
        """
        B, N, C = x.shape
        if C != self.config.dim:
            raise AssertionError(
                f"Input embedding dimension ({C}) should match layer embedding dimension ({self.config.dim})."
            )

        qkv = self.qkv(x)
        qkv = jnp.reshape(
            qkv, (B, N, 3, self.config.num_heads, C // self.config.num_heads)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)
        if self.config.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # TODO: implement fused attention for better performance
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(self.config.head_dim)
        attn = nnx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
