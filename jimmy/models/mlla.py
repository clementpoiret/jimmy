from typing import List

import jax.numpy as jnp
from einops import rearrange, reduce
from flax import nnx

from jimmy.layers import (
    ConvStem,
    GenericLayer,
    MllaBlock,
    PatchMerging,
    SimpleConvStem,
    SimplePatchMerging,
)
from jimmy.layers.builders import get_norm
from jimmy.layers.configs import ViTBlockConfig


class Mlla(nnx.Module):
    """
    Mamba-like Linear Attention (MLLA) model, implemented from Han, et al., 2024 [1].

    This class implements the MLLA architecture, which combines linear attention
    mechanisms with Mamba-inspired blocks for efficient and effective vision processing.

    Attributes:
        num_classes (int): Number of output classes for classification.
        simple_downsample (bool): If True, use simple downsampling instead of patch merging.
        simple_patch_embed (bool): If True, use simple patch embedding instead of convolutional stem.
        pos_drop_rate (float): Dropout rate for positional embeddings.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (str): Type of normalization layer to use.
        block_types (List[str]): Types of blocks to use in each stage.
        block_config (dict): Configuration for the blocks.
        attention_config (dict): Configuration for the attention mechanism.
        mamba_config (dict): Configuration for the Mamba-inspired blocks.

    Args:
        depths (List[int]): Number of blocks in each stage.
        patch_size (int): Size of the image patches.
        in_features (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        num_heads (List[int]): Number of attention heads in each stage.
        layer_window_sizes (List[int]): Window sizes for each layer.
        rngs (nnx.Rngs): Random number generators.
        block_kwargs (dict, optional): Additional keyword arguments for blocks.
        attention_kwargs (dict, optional): Additional keyword arguments for attention.
        mamba_kwargs (dict, optional): Additional keyword arguments for Mamba blocks.
        **kwargs: Additional keyword arguments.

    References:
        [1] Demystify Mamba in Vision: A Linear Attention Perspective
    """

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
        depths: List[int],
        patch_size: int,
        in_features: int,
        embed_dim: int,
        num_heads: List[int],
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
        num_features = int(embed_dim * 2 ** (num_layers - 1))

        stem = SimpleConvStem if self.simple_patch_embed else ConvStem
        self.patch_embed = stem(
            patch_size=patch_size,
            in_features=in_features,
            embed_dim=embed_dim,
            flatten=False,
            rngs=rngs,
        )

        patch_merging_block = (
            SimplePatchMerging if self.simple_downsample else PatchMerging
        )

        self.pos_drop = nnx.Dropout(self.pos_drop_rate, rngs=rngs)
        dpr = list(jnp.linspace(0, self.drop_path_rate, sum(depths)))

        self.levels = [
            GenericLayer(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                block=MllaBlock,
                block_config=ViTBlockConfig,
                layer_window_size=layer_window_sizes[i],
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
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
            )
            for i in range(num_layers)
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
