from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from jimmy.layers.attention import Attention
from jimmy.layers.blocks import Block, Identity
from jimmy.layers.mlp import Mlp
from jimmy.layers.patch_embed import PatchEmbed

# TODO: handle dynamic image size
# TODO: dynamic_img_size
# TODO: dynamic_img_pad
# TODO: interpolate_pos_encoding
# TODO: pos_drop
# TODO: compare to prepare_tokens_with_masks


class DinoV2(nnx.Module):

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = True,
        class_token: bool = True,
        reg_tokens: int = 1,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        # dynamic_img_size: bool = False,
        # dynamic_img_pad: bool = False,
        embed_layer: nnx.Module = PatchEmbed,
        act_layer: Callable = nnx.gelu,
        block: nnx.Module = Block,
        attention: nnx.Module = Attention,
        ffn_layer: nnx.Module = Mlp,
        init_values: float | None = None,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        rngs: nnx.Rngs = None,
    ):
        assert pos_embed in ("", "none", "learn")

        # self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        # self.dynamic_img_size = dynamic_img_size

        self.patch_embed = embed_layer(patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dim=embed_dim,
                                       rngs=rngs)
        self.cls_token = nnx.Param(
            nnx.initializers.zeros(rngs.params(),
                                   [1, 1, embed_dim]),) if class_token else None
        self.reg_tokens = nnx.Param(
            nnx.initializers.zeros(
                rngs.params(),
                [1, reg_tokens, embed_dim],
            )) if reg_tokens else None

        num_patches = (img_size // patch_size)**2
        self.embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == "none":
            self.pos_embed = None
        else:
            self.pos_embed = nnx.Param(
                nnx.initializers.normal(.02)(rngs.params(),
                                             [1, self.embed_len, embed_dim]))

        # Stochastic depth decay rule
        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [rate for rate in jnp.linspace(0, drop_path_rate, depth)]

        # To respect the original naming
        for i in range(depth):
            _block = block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                attention=attention,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                rngs=rngs,
            )
            setattr(self, f"blocks.{i}", _block)

        self.norm = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.head = Identity()

        # self.mask_token = nnx.Param(
        #     nnx.initializers.zeros(
        #         rngs.params(),
        #         [1, embed_dim],
        #     ))

    def interpolate_pos_encoding(self, x: jnp.ndarray, h: int, w: int):
        previous_dtype = x.dtype

        # TODO: VERIFY REG TOKENS
        npatch = x.shape[1] - 1
        N = self.embed_len - 1

        if npatch == N and h == w:
            return self.pos_embed

        pos_embed = self.pos_embed.value.astype("float32")
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size
        M = int(jnp.sqrt(N))  # Recover the number of patches in each dimension

        assert N == M * M

        patch_pos_embed = jax.image.resize(
            patch_pos_embed.reshape(1, M, M, dim),
            (1, h0, w0, dim),
            method="bicubic",
        )
        assert (h0, w0) == patch_pos_embed.shape[1:3]

        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, -1, dim))

        return jnp.concatenate((class_pos_embed[None], patch_pos_embed),
                               axis=1).astype(previous_dtype)

    def _pos_embed(self, x: jnp.ndarray, h: int, w: int):
        if self.pos_embed is None:
            return jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        to_cat = []
        if self.cls_token is not None:
            # Broadcast cls_token to match batch size
            cls_token_value = self.cls_token.value
            expanded_cls_token = jnp.broadcast_to(
                cls_token_value, (x.shape[0], 1, cls_token_value.shape[-1]))
            to_cat.append(expanded_cls_token)

        if self.reg_tokens is not None:
            reg_tokens_value = self.reg_tokens.value
            expanded_reg_tokens = jnp.broadcast_to(
                reg_tokens_value, (x.shape[0], 1, reg_tokens_value.shape[-1]))
            to_cat.append(expanded_reg_tokens)

        # WARNING: it seems different compared to
        # https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
        if self.no_embed_class:
            x = x + self.interpolate_pos_encoding(self.pos_embed, h, w)
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
        else:
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
            x = x + self.interpolate_pos_encoding(self.pos_embed, h, w)

        return x

    def __call__(self, x: jnp.ndarray):
        N, H, W, C = x.shape

        x = self.patch_embed(x)
        x = self._pos_embed(x, h=H, w=W)

        for i in range(self.depth):
            _block = getattr(self, f"blocks.{i}")
            x = _block(x)

        x = self.norm(x)
        x = self.head(x)

        return x
