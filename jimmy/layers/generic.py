from typing import Callable

import jax.numpy as jnp
from flax import nnx

from jimmy.utils import window_partition, window_reverse

from .blocks import ConvBlock, ViTBlock
from .configs import ViTBlockConfig
from .misc import Downsample


class GenericLayer(nnx.Module):
    """Base layer.

    This class implements a layer of the MambaVision architecture, which can be either
    a convolutional block or a transformer block, optionally followed by a downsampling operation.
    It supports both traditional transformer attention and Mamba-style mixing mechanisms.

    Attributes:
        conv (bool): Whether to use convolutional blocks instead of transformer blocks.
        blocks (list): List of Block or ConvBlock instances.
        transformer_block (bool): Whether the layer uses transformer blocks.
        downsample (Downsample | None): Downsampling operation, if applicable.
        do_gt (bool): Flag for global token usage (currently not implemented).
        window_size (int): Size of the window for windowed attention in transformer blocks.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks in the layer.
        num_heads (int): Number of attention heads in transformer blocks.
        window_size (int): Size of the window for windowed attention.
        conv (bool, optional): Whether to use convolutional blocks. Defaults to False.
        downsample (bool, optional): Whether to apply downsampling after the blocks. Defaults to True.
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
        qk_norm (bool, optional): Whether to apply normalization to query and key. Defaults to False.
        ffn_bias (bool, optional): If True, use bias in the feed-forward network. Defaults to True.
        proj_bias (bool, optional): If True, use bias in the projection layers. Defaults to True.
        proj_drop (float, optional): Dropout rate for projection layers. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate for attention. Defaults to 0.0.
        drop_path (float | list, optional): Stochastic depth rate. Defaults to 0.0.
        init_values (float | None, optional): Initial layer scale value. Defaults to None.
        init_values_conv (float | None, optional): Initial layer scale value for conv blocks. Defaults to None.
        transformer_attention (Callable, optional): Attention mechanism to use for transformer blocks. Defaults to Attention.
        mamba_mixer (Callable, optional): Mamba mixing mechanism to use. Defaults to MambaVisionMixer.
        act_layer (Callable, optional): Activation function to use. Defaults to nnx.gelu.
        norm_layer (Callable, optional): Normalization layer to use. Defaults to nnx.LayerNorm.
        ffn_layer (Callable, optional): Feed-forward network layer to use. Defaults to Mlp.
        block_types (list, optional): List of block types to use in the layer. Defaults to [].
        rngs (nnx.Rngs, optional): Random number generators. Defaults to None.
    """

    downsampler: nnx.Module = Downsample

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        rngs: nnx.Rngs,
        block: nnx.Module = ViTBlock,
        block_config: Callable = ViTBlockConfig,
        layer_window_size: int = -1,
        msa_window_size: int = -1,
        drop_path: float | list = 0.0,
        block_types: list = [],
        downsample: bool = True,
        block_kwargs: dict = {},
        config_kwargs: dict = {},
        **kwargs,
    ):
        self.__dict__.update(**kwargs)

        self.reshape = not isinstance(block, ConvBlock)

        self.blocks = []
        for i in range(depth):
            if len(block_types) != depth:
                if len(block_types) != 1:
                    raise ValueError(
                        "Length mismatch between `block_types` and `depth`."
                    )
                block_type = block_types[0]
            else:
                block_type = block_types[i]

            cfg = config_kwargs | {
                "drop_path": drop_path[i] if isinstance(drop_path, list) else drop_path,
                "attention": block_type,
                "msa_window_size": msa_window_size,
            }
            cfg = {k: v for k, v in cfg.items() if k in block_config.__dict__.keys()}

            self.blocks.append(
                block(
                    dim=dim,
                    config=block_config(**cfg),
                    rngs=rngs,
                    **block_kwargs,
                )
            )

        self.downsample = (
            None if not downsample else self.downsampler(dim=dim, rngs=rngs)
        )
        self.do_gt = False
        self.window_size = layer_window_size

    def __call__(self, x: jnp.ndarray):
        shape = x.shape

        if self.reshape:
            assert len(shape) == 4
            _, H, W, _ = x.shape
            ws = max(H, W) if self.window_size == -1 else self.window_size

            pad_b = (ws - H % ws) % ws
            pad_r = (ws - W % ws) % ws
            if pad_r > 0 or pad_b > 0:
                x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
                _, Hp, Wp, _ = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, ws)

        for blk in self.blocks:
            x = blk(x)

        if self.reshape:
            x = window_reverse(x, ws, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]

        if self.downsample is not None:
            x = self.downsample(x)

        return x
