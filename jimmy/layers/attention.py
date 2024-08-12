import jax.numpy as jnp
from einops import rearrange, reduce
from flax import nnx

from .configs import AttentionConfig
from .norm import RMSNormGated
from .posemb import PosEmbMLPSwinv2D, RoPE


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
        config: AttentionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        match config.norm_layer:
            case "layernorm":
                norm_layer = nnx.LayerNorm
            case "rmsnormgated":
                norm_layer = RMSNormGated
            case "batchnorm":
                norm_layer = nnx.BatchNorm
            case _:
                raise ValueError(f"Unknown norm `{config.norm_layer}`")

        self.qkv = nnx.Linear(config.dim,
                              config.dim * 3,
                              use_bias=config.qkv_bias,
                              rngs=rngs)
        self.attn_drop = nnx.Dropout(config.attn_drop, rngs=rngs)
        self.proj = nnx.Linear(config.dim,
                               config.dim,
                               use_bias=config.proj_bias,
                               rngs=rngs)
        self.proj_drop = nnx.Dropout(config.proj_drop, rngs=rngs)

        self.q_norm = norm_layer(config.head_dim,
                                 rngs=rngs) if config.qk_norm else None
        self.k_norm = norm_layer(config.head_dim,
                                 rngs=rngs) if config.qk_norm else None

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
            qkv, (B, N, 3, self.config.num_heads, C // self.config.num_heads))
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


class LinearAttention(nnx.Module):
    """Linear Attention from Mamba-like Linear Attention (MLLA) paper."""

    def __init__(
        self,
        config: AttentionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config

        self.qk = nnx.Linear(config.dim, config.dim * 2, rngs=rngs)
        self.lepe = nnx.Conv(
            in_features=config.dim,
            out_features=config.dim,
            kernel_size=(3, 3),
            padding=(1, 1),
            feature_group_count=config.dim,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        b, n, c = x.shape
        h = int(n**0.5)
        w = int(n**0.5)
        num_heads = self.config.num_heads

        q, k = rearrange(self.qk(x),
                         "b n (qk h d) -> qk b h n d",
                         qk=2,
                         h=num_heads)
        v = rearrange(x, "b n (h d) -> b h n d", h=num_heads)

        q = nnx.elu(q) + 1.0
        k = nnx.elu(k) + 1.0

        # TODO: Try to define rope here to avoid setting input_resolution a priori
        rope = RoPE(shape=(h, w, c))
        q_2d = rearrange(q, "b h (x y) d -> b x y (h d)", x=h, y=w)
        k_2d = rearrange(k, "b h (x y) d -> b x y (h d)", x=h, y=w)

        q_rope = rearrange(rope(q_2d),
                           "b x y (h d) -> b h (x y) d",
                           h=num_heads)
        k_rope = rearrange(rope(k_2d),
                           "b x y (h d) -> b h (x y) d",
                           h=num_heads)

        # Compute attention
        z = 1 / (jnp.einsum("bhnd,bhd->bhn", q,
                            reduce(k, "b h n d -> b h d", "mean")) + 1e-6)
        kv = jnp.einsum("bhnd,bhne->bhde", k_rope * (n**-0.5), v * (n**-0.5))
        x = jnp.einsum("bhnd,bhde->bhne", q_rope, kv) * z[..., None]

        # Reshape output
        x = rearrange(x, "b h n d -> b n (h d)")

        # Apply LePE
        v_2d = rearrange(v, "b h (x y) d -> b x y (h d)", x=h, y=w)
        lepe_out = self.lepe(v_2d)

        lepe_out = rearrange(lepe_out,
                             "b x y (h d) -> b (x y) (h d)",
                             h=num_heads)

        # Combine attention output and LePE
        x = x + lepe_out

        return x


class WindowedAttention(nnx.Module):
    """
    Windowed Attention module.

    This module implements Window attention based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention"

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
        resolution: int,
        seq_len: int,
        config: AttentionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.resolution = resolution
        match config.norm_layer:
            case "layernorm":
                norm_layer = nnx.LayerNorm
            case "rmsnormgated":
                norm_layer = RMSNormGated
            case "batchnorm":
                norm_layer = nnx.BatchNorm
            case _:
                raise ValueError(f"Unknown norm `{config.norm_layer}`")

        self.qkv = nnx.Linear(config.dim,
                              config.dim * 3,
                              use_bias=config.qkv_bias,
                              rngs=rngs)
        self.attn_drop = nnx.Dropout(config.attn_drop, rngs=rngs)
        self.proj = nnx.Linear(config.dim,
                               config.dim,
                               use_bias=config.proj_bias,
                               rngs=rngs)
        self.proj_drop = nnx.Dropout(config.proj_drop, rngs=rngs)

        # Attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv2D(
            window_size=[resolution, resolution],
            pretrained_window_size=[resolution, resolution],
            num_heads=config.num_heads,
            seq_len=seq_len,
            rngs=rngs,
        )

        self.q_norm = norm_layer(config.head_dim,
                                 rngs=rngs) if config.qk_norm else None
        self.k_norm = norm_layer(config.head_dim,
                                 rngs=rngs) if config.qk_norm else None

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
            qkv, (B, N, 3, self.config.num_heads, C // self.config.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)
        if self.config.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # TODO: implement fused attention for better performance
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(self.config.head_dim)
        attn = self.pos_emb_funct(attn, self.resolution**2)
        attn = nnx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
