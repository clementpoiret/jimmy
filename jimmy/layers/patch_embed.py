from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
from flax import nnx


class PatchEmbed(nnx.Module):
    """Image to Patch Embedding"""

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
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        Taken as is from timm
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(
                img_size[1] / self.patch_size[1])
        else:
            return img_size[0] // self.patch_size[0], img_size[
                1] // self.patch_size[1]

    def __call__(self, x: jnp.ndarray):
        _, H, W, C = x.shape

        # Here again, the logic for dynamic_img_size and dynamic_img_pad is
        # taken from timm and adapted for jax.numpy
        if self.img_size is not None:
            if not self.dynamic_img_size:
                assert H == self.img_size[
                    0], f"Input height ({H}) doesn't match model ({self.img_size[0]})"
                assert W == self.img_size[
                    1], f"Input width ({W}) doesn't match model ({self.img_size[1]})"
            elif not self.dynamic_img_pad:
                assert H % self.patch_size[
                    0] == 0, f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})"
                assert W % self.patch_size[
                    1] == 0, f"Input height ({W}) should be divisible by patch size ({self.patch_size[1]})"

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
