from typing import Optional

import jax.numpy as jnp
from flax import nnx


class PatchEmbed(nnx.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nnx.Module] = None,
        flatten: bool = True,
        use_bias: bool = True,
        rngs: nnx.Rngs = None,
    ):
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.patch_H, self.patch_W = patch_size, patch_size

        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=(self.patch_H, self.patch_W),
            strides=(self.patch_H, self.patch_W),
            padding="VALID",
            use_bias=use_bias,
            rngs=rngs,
        )

        self.norm = norm_layer(num_features=embed_dim,
                               rngs=rngs) if norm_layer else None

    def __call__(self, x: jnp.ndarray):
        _, H, W, C = x.shape
        assert (
            H % self.patch_H == 0 and W % self.patch_W == 0
        ), f"Image size ({H}*{W}) cannot be evenly divided by patch size ({self.patch_H}*{self.patch_W})."

        x = self.proj(x)

        _, H, W, _ = x.shape
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        if self.norm:
            x = self.norm(x)

        if not self.flatten:
            x = jnp.reshape(x, (-1, H, W, self.embed_dim))

        return x
