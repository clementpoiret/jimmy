from typing import List

import jax.numpy as jnp
from einops import rearrange, reduce
from flax import nnx

from jimmy.layers import (ConvStem, GenericLayer, MllaBlock, PatchMerging,
                          SimpleConvStem, SimplePatchMerging)
from jimmy.layers.builders import get_norm
from jimmy.layers.configs import ViTBlockConfig


class Mlla(nnx.Module):
    num_classes: int = 1000
    simple_downsample: bool = False
    simple_patch_embed: bool = False
    pos_drop_rate = 0.0
    drop_path_rate: float = 0.2
    norm_layer = "layernorm"
    block_types: List[str] = [
        "linearattention",
        "linearattention",
        "linearattention",
        "linearattention",
    ]

    block_config = {
        "mlp_ratio": 4.0,
        "ffn_layer": "mlp",
        "ffn_bias": True,
        "act_layer": "silu",  # gelu in VMamba2
        "init_values": None,
        "use_dwc": True,  # false in VMamba2
    }
    attention_config = {
        "qkv_bias": True,
        "qk_norm": True,
        "proj_bias": True,
        "proj_drop": 0.0,
        "attn_drop": 0.0,
        "norm_layer": "layernorm",
    }
    mamba_config = {
        "d_state": 64,
        "d_conv": 3,
        "expand": 2,
        "linear_attn_duality": True,
        "chunk_size": 256,
    }

    def __init__(
        self,
        depths: List[int],  # = [2, 4, 12, 4],
        patch_size: int,  # = 4,
        in_features: int,  # = 3,
        embed_dim: int,  # = 64,
        num_heads: List[int],  # = [2, 4, 8, 16],
        layer_window_sizes: List[int],
        *,
        rngs: nnx.Rngs,
        block_kwargs: dict = {},
        attention_kwargs: dict = {},
        mamba_kwargs: dict = {},
        **kwargs,
    ):
        self.__dict__.update(**kwargs)
        self.attention_config.update(**attention_kwargs)
        self.mamba_config.update(**mamba_kwargs)
        self.block_config.update(**block_kwargs)

        assert len(self.block_types) == len(depths)

        num_layers = len(depths)
        num_features = int(embed_dim * 2**(num_layers - 1))

        stem = SimpleConvStem if self.simple_patch_embed else ConvStem
        self.patch_embed = stem(
            patch_size=patch_size,
            in_features=in_features,
            embed_dim=embed_dim,
            flatten=False,
            rngs=rngs,
        )

        patch_merging_block = (SimplePatchMerging
                               if self.simple_downsample else PatchMerging)

        self.pos_drop = nnx.Dropout(self.pos_drop_rate, rngs=rngs)
        dpr = list(jnp.linspace(0, self.drop_path_rate, sum(depths)))

        self.levels = [
            GenericLayer(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                block=MllaBlock,
                block_config=ViTBlockConfig,
                layer_window_size=layer_window_sizes[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                block_types=[self.block_types[i]],
                downsample=i < num_layers - 1,
                downsampler=patch_merging_block,
                config_kwargs={
                    **self.block_config,
                },
                block_kwargs={
                    "attention_kwargs": {
                        "num_heads": num_heads[i],
                        **self.attention_config,
                    },
                    "mamba_kwargs": {
                        **self.mamba_config,
                    },
                },
                rngs=rngs,
            ) for i in range(num_layers)
        ]

        self.norm = get_norm(self.norm_layer)(num_features, rngs=rngs)
        self.head = nnx.Linear(num_features, self.num_classes, rngs=rngs)

    def forward_features(self, x: jnp.ndarray):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for i, level in enumerate(self.levels):
            x = level(x)

        x = self.norm(rearrange(x, "b h w c -> b (h w) c"))
        x = reduce(x, "b l c -> b c", "mean")

        return x

    def __call__(self, x: jnp.ndarray):
        x = self.forward_features(x)
        x = self.head(x)

        return x


# TODO: Merge with VMamba2
# class Mlla(nnx.Module):

#     def __init__(
#         self,
#         patch_size: int = 4,
#         in_features: int = 3,
#         num_classes: int = 1000,
#         embed_dim: int = 64,
#         depths: List[int] = [2, 2, 6, 2],
#         num_heads: List[int] = [3, 6, 12, 24],
#         mlp_ratio: float = 4.0,
#         qkv_bias=True,
#         qk_norm: bool = False,
#         ffn_bias: bool = True,
#         proj_bias: bool = True,
#         pos_drop_rate=0.0,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         drop_path_rate: float = 0.1,
#         norm_layer: nnx.Module = nnx.LayerNorm,
#         simple_downsample: bool = False,
#         simple_patch_embed: bool = False,
#         rngs: nnx.Rngs = None,
#     ):
#         num_layers = len(depths)
#         num_features = int(embed_dim * 2 ** (num_layers - 1))

#         stem = SimpleConvStem if simple_patch_embed else ConvStem
#         self.patch_embed = stem(
#             patch_size=patch_size,
#             in_features=in_features,
#             embed_dim=embed_dim,
#             rngs=rngs,
#         )

#         patch_merging_block = SimplePatchMerging if simple_downsample else PatchMerging

#         self.pos_drop = nnx.Dropout(pos_drop_rate, rngs=rngs)
#         dpr = list(jnp.linspace(0, drop_path_rate, sum(depths)))

#         self.levels = [
#             MllaLayer(
#                 dim=int(embed_dim * 2**i),
#                 depth=depths[i],
#                 num_heads=num_heads[i],
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 ffn_bias=ffn_bias,
#                 proj_drop=proj_drop,
#                 drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
#                 norm_layer=norm_layer,
#                 downsample=patch_merging_block if i < num_layers - 1 else None,
#                 rngs=rngs,
#             )
#             for i in range(num_layers)
#         ]

#         self.norm = norm_layer(num_features, rngs=rngs)
#         self.head = nnx.Linear(num_features, num_classes, rngs=rngs)

#     def forward_features(self, x: jnp.ndarray):
#         x = self.patch_embed(x)
#         x = self.pos_drop(x)

#         for i, level in enumerate(self.levels):
#             x = level(x)

#         x = reduce(self.norm(x), "b l c -> b c", "mean")

#         return x

#     def __call__(self, x: jnp.ndarray):
#         x = self.forward_features(x)
#         x = self.head(x)

#         return x
