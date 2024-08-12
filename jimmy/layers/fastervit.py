from typing import Callable

import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from flax.linen import avg_pool


from .blocks import HATBlock
from .configs import ViTBlockConfig
from .misc import Downsample


class TokenInitializer(nnx.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    ct_size: int = 1

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        window_size: int,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window
        """
        self.__dict__.update(**kwargs)

        output_size = int((self.ct_size) * input_resolution / window_size)
        self.strides = int(input_resolution / output_size)
        self.kernel = input_resolution - (output_size - 1) * self.strides

        self.pos_embed = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            feature_group_count=dim,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        x = avg_pool(
            self.pos_embed(x),
            window_shape=(self.kernel, self.kernel),
            strides=(self.strides, self.strides),
        )

        ct = rearrange(
            x,
            "b (h h1) (w w1) c -> b (h h1 w w1) c",
            h1=self.ct_size,
            w1=self.ct_size,
        )
        return ct


# TODO: merge with GenericLayer?
class FasterViTLayer(nnx.Module):
    """
    FasterViT layer for vision models.

    This class implements a FasterViT layer that implementing Hierarchical
    Attention (HAT).

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
    only_local: bool = False
    hierarchy: bool = True

    def __init__(
        self,
        dim: int,
        depth: int,
        input_resolution: int,
        window_size: int,
        *,
        rngs: nnx.Rngs,
        block: nnx.Module = HATBlock,
        block_config: Callable = ViTBlockConfig,
        msa_window_size: int = -1,
        drop_path: float | list = 0.0,
        downsample: bool = True,
        block_kwargs: dict = {},
        config_kwargs: dict = {},
        **kwargs,
    ):
        self.__dict__.update(**kwargs)

        self.is_hat = block is HATBlock

        self.blocks = []
        for i in range(depth):
            cfg = config_kwargs | {
                "drop_path": drop_path[i] if isinstance(drop_path, list) else drop_path,
            }
            cfg = {k: v for k, v in cfg.items() if k in block_config.__dict__.keys()}

            if self.is_hat:
                block_kwargs = block_kwargs | {
                    "sr_ratio": (
                        input_resolution // window_size if not self.only_local else 1
                    ),
                    "window_size": window_size,
                    "last": i == depth - 1,
                    "ct_size": self.ct_size,
                }

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

        if (
            len(self.blocks)
            and not self.only_local
            and input_resolution // window_size > 1
            and self.hierarchy
            and self.is_hat
        ):
            self.global_tokenizer = TokenInitializer(
                dim,
                input_resolution,
                window_size,
                ct_size=self.ct_size,
                rngs=rngs,
            )
            self.do_gt = True
        else:
            self.do_gt = False

        self.window_size = window_size

    def window_partition(self, x: jnp.ndarray, window_size: int):
        return rearrange(
            x, "b (h h1) (w w1) c -> (b h w) (h1 w1) c", h1=window_size, w1=window_size
        )

    def window_reverse(self, x: jnp.ndarray, window_size: int, h: int, w: int):
        return rearrange(
            x,
            "(b h w) (h1 w1) c -> b (h h1) (w w1) c",
            h=h // window_size,
            w=w // window_size,
            h1=window_size,
            w1=window_size,
        )

    def __call__(self, x: jnp.ndarray):
        ct = self.global_tokenizer(x) if self.do_gt else None
        b, h, w, c = x.shape

        if self.is_hat:
            x = self.window_partition(x, self.window_size)
            for blk in self.blocks:
                x, ct = blk(x, ct)
            x = self.window_reverse(x, self.window_size, h, w)
        else:
            for blk in self.blocks:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
