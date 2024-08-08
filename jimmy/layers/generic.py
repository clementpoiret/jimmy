from typing import Callable

import jax.numpy as jnp
from flax import nnx

from jimmy.utils import window_partition, window_reverse

from .blocks import ConvBlock, ViTBlock
from .configs import ViTBlockConfig
from .misc import Downsample


class GenericLayer(nnx.Module):
    """
    Generic layer for vision models.

    This class implements a flexible layer that can be used in various vision model
    architectures. It supports both convolutional and transformer-style blocks,
    with optional downsampling.

    Attributes:
        reshape (bool): Whether to reshape the input for windowed attention.
        blocks (List[nnx.Module]): List of block modules.
        downsample (nnx.Module | None): Downsampling module, if applicable.
        do_gt (bool): Flag for global token usage (not implemented).
        window_size (int): Size of the window for windowed attention.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks in the layer.
        rngs (nnx.Rngs): Random number generators.
        block (nnx.Module): Block module to use (e.g., ViTBlock or ConvBlock).
        block_config (Callable): Configuration for the block.
        layer_window_size (int): Size of the layer window for attention.
        msa_window_size (int): Size of the multi-head self-attention window.
        drop_path (float | list): Stochastic depth rate.
        block_types (list): Types of blocks to use in the layer.
        downsample (bool): Whether to apply downsampling after the blocks.
        block_kwargs (dict): Additional keyword arguments for blocks.
        config_kwargs (dict): Additional keyword arguments for block configuration.
        **kwargs: Additional keyword arguments.
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
