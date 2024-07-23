from typing import Callable, List

import jax.numpy as jnp
from einops import reduce
from flax import nnx

from jimmy.layers import (Attention, ConvPatchEmbed, Identity,
                          MambaVisionLayer, MambaVisionMixer, Mlp)


def adaptive_avg_pool2d(x: jnp.ndarray):
    """
    Perform adaptive average pooling on a 4D input tensor.

    Args:
        x (jnp.ndarray): Input tensor of shape (batch_size, height, width, channels).

    Returns:
        jnp.ndarray: Output tensor of shape (batch_size, channels, 1, 1).
    """
    return reduce(x, "b h w c -> b c 1 1", "mean")


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

    def __init__(
        self,
        in_features: int,
        dim: int,
        in_dim: int,
        depths: List[int],
        window_size: List[int],
        mlp_ratio: float,
        num_heads: List[int],
        drop_path_rate: float = 0.2,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        init_values_conv: float | None = None,
        transformer_attention: Callable = Attention,
        mamba_mixer: Callable = MambaVisionMixer,
        act_layer: Callable = nnx.gelu,
        norm_layer: Callable = nnx.LayerNorm,
        ffn_layer: Callable = Mlp,
        num_classes: int = 1000,
        rngs: nnx.Rngs = None,
    ):
        num_features = int(dim * 2**(len(depths) - 1))
        self.num_classes = num_classes

        self.patch_embed = ConvPatchEmbed(in_features=in_features,
                                          hidden_features=in_dim,
                                          out_features=dim,
                                          rngs=rngs)
        dpr = list(jnp.linspace(0, drop_path_rate, sum(depths)))

        self.levels = []
        for i, item in enumerate(depths):
            conv = i > 2
            level = MambaVisionLayer(
                dim=int(dim * 2**i),
                depth=item,
                num_heads=num_heads[i],
                window_size=window_size[i],
                conv=conv,
                downsample=i < 3,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                ffn_bias=ffn_bias,
                proj_bias=proj_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                init_values=init_values,
                init_values_conv=init_values_conv,
                transformer_attention=transformer_attention,
                mamba_mixer=mamba_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                ffn_layer=ffn_layer,
                block_types=self._get_block_types(item),
                rngs=rngs,
            )
            self.levels.append(level)

        self.norm = nnx.BatchNorm(num_features=num_features, rngs=rngs)
        self.head = (nnx.Linear(num_features, num_classes, rngs=rngs)
                     if num_classes else Identity())

    def _get_block_types(self, l: int):
        """
        Generate a list of block types for a layer.

        Args:
            l (int): Total number of blocks in the layer.

        Returns:
            List[str]: A list of block types, with the first half being "mambavisionmixer"
                       and the second half being "attention".
        """
        first_half_size = (l + 1) // 2
        second_half_size = l // 2
        return ["mambavisionmixer"] * first_half_size + ["attention"
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
        x = adaptive_avg_pool2d(x)
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
