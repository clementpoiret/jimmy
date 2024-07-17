import math
from typing import Callable, List, Optional, Tuple, Union

import jax.numpy as jnp
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
        self.patch_size = patch_size if isinstance(
            patch_size, list) else [patch_size, patch_size]

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

        self.norm = norm_layer(num_features=embed_dim,
                               rngs=rngs) if norm_layer else None

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
        else:
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
                if H != self.img_size[
                    0]:
                    raise AssertionError(f"Input height ({H}) doesn't match model ({self.img_size[0]})")
                if W != self.img_size[
                    1]:
                    raise AssertionError(f"Input width ({W}) doesn't match model ({self.img_size[1]})")
            elif not self.dynamic_img_pad:
                if H % self.patch_size[
                    0] != 0:
                    raise AssertionError(f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})")
                if W % self.patch_size[
                    1] != 0:
                    raise AssertionError(f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})")

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
        self.conv_down = nnx.Sequential([
            nnx.Conv2d(in_features=in_features,
                       out_features=hidden_features,
                       kernel_size=(3, 3),
                       strides=2,
                       padding="SAME",
                       use_bias=False,
                       rngs=rngs),
            norm_layer(num_features=hidden_features, rngs=rngs, **norm_params),
            act_layer,
            nnx.Conv2d(in_features=hidden_features,
                       out_features=out_features,
                       kernel_size=(3, 3),
                       strides=2,
                       padding="SAME",
                       use_bias=False,
                       rngs=rngs),
            norm_layer(num_features=out_features, rngs=rngs, **norm_params),
            act_layer,
        ])

    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the ConvPatchEmbed module.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after applying convolutional patch embedding.
        """
        return self.conv_down(x)
