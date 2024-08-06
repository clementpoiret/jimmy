from typing import Callable, Optional

import jax.numpy as jnp
from flax import nnx

from .blocks import MllaBlock
from .mlp import Mlp


# TODO: merge with VMamba2Layer
class MllaLayer(nnx.Module):
    """A basic MLLA layer for one stage"""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_drop: float = 0.0,
        drop_path: float | list = 0.0,
        act_layer: Callable = nnx.silu,
        norm_layer: Callable = nnx.LayerNorm,
        ffn_layer: Callable = Mlp,
        downsample: Optional[nnx.Module] = None,
        rngs: nnx.Rngs = None,
    ):
        self.blocks = [
            MllaBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                proj_drop=proj_drop,
                drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                act_layer=act_layer,
                norm_layer=norm_layer,
                ffn_layer=ffn_layer,
                rngs=rngs,
            )
            for i in range(depth)
        ]

        self.downsample = (
            downsample(dim=dim, rngs=rngs) if downsample is not None else None
        )

    def __call__(self, x: jnp.ndarray):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
