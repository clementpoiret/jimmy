import math
from typing import Callable, List, Optional, Tuple, Union

import jax.numpy as jnp
from einops import rearrange
from flax import nnx


class PatchEmbed(nnx.Module):
    """
    Image to Patch Embedding, inspired from Timm.

    This module converts an image into a sequence of embedded patches.

    Args:
        img_size (int | None, optional): Size of the input image (assumed square). Defaults to None.
        patch_size (Union[List[int], int], optional): Size of the patches. Defaults to 16.
        in_channels (int, optional): Number of input channels. Defaults to 3.
        embed_dim (int, optional): Dimension of the embedded patches. Defaults to 768.
        norm_layer (Optional[nnx.Module], optional): Normalization layer. Defaults to None.
        flatten (bool, optional): Whether to flatten the output. Defaults to True.
        dynamic_img_size (bool, optional): Whether to allow dynamic image sizes. Defaults to False.
        dynamic_img_pad (bool, optional): Whether to use dynamic padding. Defaults to False.
        use_bias (bool, optional): Whether to use bias in the projection. Defaults to True.
        rngs (nnx.Rngs, optional): Random number generators. Defaults to None.
    """

    def __init__(
        self,
        img_size: int | None = None,
        patch_size: Union[List[int], int] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nnx.Module] = None,
        flatten: bool = True,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        use_bias: bool = True,
        rngs: nnx.Rngs = None,
    ):
        self.patch_size = (patch_size if isinstance(patch_size, list) else
                           [patch_size, patch_size])

        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad
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

        self.norm = (norm_layer(num_features=embed_dim, rngs=rngs)
                     if norm_layer else None)

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get grid (feature) size for given image size taking account of dynamic padding.
        Taken as is from timm

        Args:
            img_size (Tuple[int, int]): Size of the input image.

        Returns:
            Tuple[int, int]: Grid size after applying patches.
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(
                img_size[1] / self.patch_size[1])
        return img_size[0] // self.patch_size[0], img_size[
            1] // self.patch_size[1]

    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the PatchEmbed module.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            jnp.ndarray: Output tensor of embedded patches.

        Raises:
            AssertionError: If input dimensions don't match the expected dimensions.
        """
        _, H, W, C = x.shape

        if self.img_size is not None:
            if not self.dynamic_img_size:
                if H != self.img_size[0]:
                    raise AssertionError(
                        f"Input height ({H}) doesn't match model ({self.img_size[0]})"
                    )
                if W != self.img_size[1]:
                    raise AssertionError(
                        f"Input width ({W}) doesn't match model ({self.img_size[1]})"
                    )
            elif not self.dynamic_img_pad:
                if H % self.patch_size[0] != 0:
                    raise AssertionError(
                        f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})"
                    )
                if W % self.patch_size[1] != 0:
                    raise AssertionError(
                        f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})"
                    )

        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] -
                     H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] -
                     W % self.patch_size[1]) % self.patch_size[1]
            x = jnp.pad(x, pad_width=((0, 0), (0, pad_h), (0, pad_w), (0, 0)))

        x = self.proj(x)

        _, H, W, _ = x.shape
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        if self.norm:
            x = self.norm(x)

        if not self.flatten:
            x = jnp.reshape(x, (-1, H, W, self.embed_dim))

        return x


class ConvPatchEmbed(nnx.Module):
    """
    Convolutional Patch Embedding, used in MambaVision.

    This module applies a series of convolutional layers to embed patches.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        out_features (int): Number of output features.
        act_layer (Callable, optional): Activation function to use. Defaults to nnx.relu.
        norm_layer (Callable, optional): Normalization layer to use. Defaults to nnx.BatchNorm.
        norm_params (dict, optional): Parameters for the normalization layer. Defaults to {"epsilon": 1e-4}.
        rngs (nnx.Rngs, optional): Random number generators. Defaults to None.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable = nnx.relu,
        norm_layer: Callable = nnx.BatchNorm,
        norm_params: dict = {"epsilon": 1e-4},
        rngs: nnx.Rngs = None,
    ):
        self.conv_down = nnx.Sequential(
            nnx.Conv(
                in_features=in_features,
                out_features=hidden_features,
                kernel_size=(3, 3),
                strides=2,
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            ),
            norm_layer(num_features=hidden_features, rngs=rngs, **norm_params),
            act_layer,
            nnx.Conv(
                in_features=hidden_features,
                out_features=out_features,
                kernel_size=(3, 3),
                strides=2,
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            ),
            norm_layer(num_features=out_features, rngs=rngs, **norm_params),
            act_layer,
        )

    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the ConvPatchEmbed module.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after applying convolutional patch embedding.
        """
        return self.conv_down(x)


class Conv(nnx.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "SAME",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        norm: nnx.Module | None = nnx.BatchNorm,
        act: Callable = nnx.relu,
        rngs: nnx.Rngs = None,
    ):
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding=padding,
            use_bias=bias,
            feature_group_count=groups,
            rngs=rngs,
        )
        self.norm = norm(out_features, rngs=rngs) if norm is not None else None
        self.act = act

    def __call__(self, x: jnp.ndarray):
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x


class SimpleConvStem(nnx.Module):
    """Simple patch embed from Mlla paper"""

    def __init__(
        self,
        patch_size: int = 4,
        in_features: int = 3,
        embed_dim=96,
        rngs: nnx.Rngs = None,
    ):
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        x = rearrange(self.conv1(x), "b h w c -> b (h w) c")
        return self.norm(x)


class ConvStem(nnx.Module):
    """Convolutional patch embed from Mlla paper"""

    def __init__(
        self,
        patch_size: int = 4,
        in_features: int = 3,
        embed_dim=96,
        rngs: nnx.Rngs = None,
    ):
        self.conv1 = Conv(in_features,
                          embed_dim // 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False)
        self.conv2 = nnx.Sequential([
            Conv(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                stride=1,
                bias=False,
                rngs=rngs,
            ),
            Conv(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                stride=1,
                bias=False,
                act=None,
                rngs=rngs,
            ),
        ])
        self.conv3 = nnx.Sequential([
            Conv(
                embed_dim // 2,
                embed_dim * 4,
                kernel_size=3,
                stride=2,
                bias=False,
                rngs=rngs,
            ),
            Conv(
                embed_dim * 4,
                embed_dim,
                kernel_size=1,
                bias=False,
                act=None,
                rngs=rngs,
            ),
        ])

    def __call__(self, x: jnp.ndarray):
        x = self.conv1(x)
        x += self.conv2(x)
        x = self.conv3(x)

        x = rearrange(x, "b h w c -> b (h w) c")

        return x


class SimplePatchMerging(nnx.Module):
    """Simple patch merging from Mlla paper"""

    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs = None,
    ):
        self.conv = Conv(dim,
                         2 * dim,
                         kernel_size=3,
                         stride=2,
                         norm=None,
                         rngs=rngs)
        self.norm = nnx.LayerNorm(2 * dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        _, l, _ = x.shape

        h = w = int(l**0.5)

        x = rearrange(
            self.conv(rearrange(x, "b (h w) c -> b h w c", h=h, w=w)),
            "b h w c -> b (h w) c",
        )
        return self.norm(x)


class PatchMerging(nnx.Module):
    """Patch merging from Mlla paper"""

    def __init__(
        self,
        dim: int,
        ratio: float = 4.0,
        rngs: nnx.Rngs = None,
    ):
        self.conv = nnx.Sequential([
            Conv(
                dim,
                2 * dim * ratio,
                kernel_size=1,
                norm=None,
                rngs=rngs,
            ),
            Conv(
                2 * dim * ratio,
                2 * dim * ratio,
                kernel_size=3,
                stride=2,
                groups=int(2 * dim * ratio),
                norm=None,
                rngs=rngs,
            ),
            Conv(
                2 * dim * ratio,
                2 * dim,
                kernel_size=1,
                act=None,
                rngs=rngs,
            ),
        ])

    def __call__(self, x: jnp.ndarray):
        _, l, _ = x.shape

        h = w = int(l**0.5)

        x = rearrange(
            self.conv(rearrange(x, "b (h w) c -> b h w c", h=h, w=w)),
            "b h w c -> b (h w) c",
        )
        return x
