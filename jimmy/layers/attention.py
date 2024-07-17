import jax.numpy as jnp
from flax import nnx


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
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nnx.Module = nnx.LayerNorm,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, use_bias=proj_bias, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

        self.q_norm = norm_layer(self.head_dim, rngs=rngs) if qk_norm else None
        self.k_norm = norm_layer(self.head_dim, rngs=rngs) if qk_norm else None

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
        if (C != self.dim):
            raise AssertionError(
                f"Input embedding dimension ({C}) should match layer embedding dimension ({self.dim})."
            )

        qkv = self.qkv(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # TODO: implement fused attention for better performance
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        attn = nnx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
