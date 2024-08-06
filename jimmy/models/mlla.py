from typing import List

import jax.numpy as jnp
from einops import reduce
from flax import nnx

from jimmy.layers import (ConvStem, MllaLayer, PatchMerging, SimpleConvStem,
                          SimplePatchMerging)


# TODO: Merge with VMamba2
class Mlla(nnx.Module):

    def __init__(
        self,
        patch_size: int = 4,
        in_features: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 64,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        mlp_ratio: float = 4.0,
        qkv_bias=True,
        qk_norm: bool = False,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        pos_drop_rate=0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nnx.Module = nnx.LayerNorm,
        simple_downsample: bool = False,
        simple_patch_embed: bool = False,
        rngs: nnx.Rngs = None,
    ):
        num_layers = len(depths)
        num_features = int(embed_dim * 2**(num_layers - 1))

        stem = SimpleConvStem if simple_patch_embed else ConvStem
        self.patch_embed = stem(
            patch_size=patch_size,
            in_features=in_features,
            embed_dim=embed_dim,
            rngs=rngs,
        )

        patch_merging_block = SimplePatchMerging if simple_downsample else PatchMerging

        self.pos_drop = nnx.Dropout(pos_drop_rate, rngs=rngs)
        dpr = list(jnp.linspace(0, drop_path_rate, sum(depths)))

        self.levels = [
            MllaLayer(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                proj_drop=proj_drop,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=patch_merging_block if i < num_layers - 1 else None,
                rngs=rngs,
            ) for i in range(num_layers)
        ]

        self.norm = norm_layer(num_features, rngs=rngs)
        self.head = nnx.Linear(num_features, num_classes, rngs=rngs)

    def forward_features(self, x: jnp.ndarray):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for i, level in enumerate(self.levels):
            x = level(x)

        x = reduce(self.norm(x), "b l c -> b c", "mean")

        return x

    def __call__(self, x: jnp.ndarray):
        x = self.forward_features(x)
        x = self.head(x)

        return x
