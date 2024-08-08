from typing import List

import jax.numpy as jnp
from einops import reduce
from flax import nnx

from jimmy.layers import (ConvBlock, ConvPatchEmbed, GenericLayer, Identity,
                          ViTBlock)
from jimmy.layers.builders import get_norm
from jimmy.layers.configs import ConvBlockConfig, ViTBlockConfig

from .mlla import Mlla


class MambaVision(nnx.Module):
    """
    MambaVision model architecture.

    This class implements the MambaVision model, which combines elements of
    vision transformers and convolutional neural networks.

    Args:
        in_features (int): Number of input channels.
        dim (int): Base dimension of the model.
        in_dim (int): Input dimension for the patch embedding.
        depths (List[int]): Number of blocks in each stage.
        window_size (List[int]): Window sizes for each stage.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_heads (List[int]): Number of attention heads in each stage.
        drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
        qk_norm (bool, optional): If True, apply normalization to query, key, value. Defaults to True.
        ffn_bias (bool, optional): If True, use bias in the feed-forward network. Defaults to True.
        proj_bias (bool, optional): If True, use bias in the projection layers. Defaults to True.
        proj_drop (float, optional): Dropout rate for projection layers. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate for attention. Defaults to 0.0.
        init_values (float | None, optional): Initial layer scale value. Defaults to None.
        init_values_conv (float | None, optional): Initial layer scale value for conv blocks. Defaults to None.
        transformer_attention (Callable, optional): Attention mechanism to use. Defaults to Attention.
        mamba_mixer (Callable, optional): Mamba mixer to use. Defaults to MambaVisionMixer.
        act_layer (Callable, optional): Activation function to use. Defaults to nnx.gelu.
        norm_layer (Callable, optional): Normalization layer to use. Defaults to nnx.LayerNorm.
        ffn_layer (Callable, optional): Feed-forward network layer to use. Defaults to Mlp.
        num_classes (int, optional): Number of classes for classification. Defaults to 1000.
    """

    msa_blocktype = "attention"
    mamba_blocktype = "mambavisionmixer"

    block_config = {
        "mlp_ratio": 4.0,
        "ffn_layer": "mlp",
        "ffn_bias": True,
        "act_layer": "gelu",
        "init_values": 1e-5,
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
        "d_state": 8,
        "d_conv": 3,
        "expand": 1,
    }

    norm_layer = "batchnorm"  # check w/ layernorm?

    drop_path_rate: float = 0.2
    num_classes: int = 1000

    ls_convblock: bool = False

    def __init__(
        self,
        in_features: int,
        dim: int,
        in_dim: int,
        depths: List[int],
        layer_window_sizes: List[int],
        msa_window_sizes: List[int],
        num_heads: List[int],
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

        num_features = int(dim * 2**(len(depths) - 1))
        self.num_classes = self.num_classes

        self.patch_embed = ConvPatchEmbed(in_features=in_features,
                                          hidden_features=in_dim,
                                          out_features=dim,
                                          rngs=rngs)
        dpr = list(jnp.linspace(0, self.drop_path_rate, sum(depths)))

        self.levels = []
        for i, item in enumerate(depths):
            conv = i > 2
            _config = {
                "init_values": None
            } if not self.ls_convblock and conv else {}

            level = GenericLayer(
                dim=int(dim * 2**i),
                depth=item,
                block=ConvBlock if conv else ViTBlock,
                block_config=ConvBlockConfig if conv else ViTBlockConfig,
                layer_window_size=layer_window_sizes[i],
                msa_window_size=msa_window_sizes[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                block_types=[None] if conv else self._get_block_types(item),
                downsample=i < len(depths) - 1,
                config_kwargs={
                    **(self.block_config | _config),
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
            self.levels.append(level)

        self.norm = get_norm(self.norm_layer)(num_features=num_features,
                                              rngs=rngs)
        self.head = (nnx.Linear(num_features, self.num_classes, rngs=rngs)
                     if self.num_classes else Identity())

    def _get_block_types(self, l: int):
        """
        Generate a list of block types for a layer.
        Used to alternate between Multi-Head Self Attention (MSA), and Mamba blocks

        Args:
            l (int): Total number of blocks in the layer.

        Returns:
            List[str]: A list of block types, with the first half being "mambavisionmixer"
                       and the second half being "attention".
        """
        first_half_size = (l + 1) // 2
        second_half_size = l // 2
        return [self.mamba_blocktype] * first_half_size + [self.msa_blocktype
                                                           ] * second_half_size

    def forward_features(self, x: jnp.ndarray):
        """
        Compute features through the network.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            jnp.ndarray: Output features of shape (batch_size, num_features).
        """
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = reduce(x, "b h w c -> b c", "mean")
        x = jnp.reshape(x, (x.shape[0], -1))

        return x

    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the MambaVision model.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, num_classes).
        """
        x = self.forward_features(x)
        x = self.head(x)

        return x


class VMamba2(Mlla):
    block_types: list = [
        "mamba2mixer",
        "mamba2mixer",
        "mamba2mixer",
        "attention",
    ]
    block_config = {
        "mlp_ratio": 4.0,
        "ffn_layer": "mlp",
        "ffn_bias": True,
        "act_layer": "gelu",  # silu in Mlla
        "init_values": None,
        "use_dwc": False,  # true in Mlla
    }
