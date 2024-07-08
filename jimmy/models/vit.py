import math
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from jimmy.layers.attention import Attention
from jimmy.layers.blocks import Block, Identity
from jimmy.layers.mlp import Mlp
from jimmy.layers.patch_embed import PatchEmbed

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
        pos_embed_reg_tokens: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        embed_layer: nnx.Module = PatchEmbed,
        act_layer: Callable = nnx.gelu,
        block: nnx.Module = Block,
        attention: nnx.Module = Attention,
        ffn_layer: nnx.Module = Mlp,
        init_values: float | None = None,
        interpolate_antialias=False,
        rngs: nnx.Rngs = None,
    ):
        assert pos_embed in ("", "none", "learn")

        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_embedded_prefix_tokens = 0
        self.num_register_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.interpolate_antialias = interpolate_antialias
        self.dynamic_img_size = dynamic_img_size
        self.pos_embed_reg_tokens = pos_embed_reg_tokens

        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dim=embed_dim,
                                       flatten=not dynamic_img_size,
                                       dynamic_img_size=dynamic_img_size,
                                       dynamic_img_pad=dynamic_img_pad,
                                       rngs=rngs)
        self.cls_token = nnx.Param(
            nnx.initializers.zeros(rngs.params(),
                                   [1, 1, embed_dim]),) if class_token else None
        self.register_tokens = nnx.Param(
            nnx.initializers.zeros(
                rngs.params(),
                [1, reg_tokens, embed_dim],
            )) if reg_tokens else None

        self.num_patches = (img_size // patch_size)**2

        if no_embed_class:
            self.embed_len = self.num_patches
        elif self.pos_embed_reg_tokens:
            self.embed_len = self.num_patches + self.num_prefix_tokens
            self.num_embedded_prefix_tokens += self.num_prefix_tokens
        else:
            self.num_embedded_prefix_tokens += 1
            self.embed_len = self.num_patches + 1

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

    def resample_pos_embed(
        self,
        pos_embed: jnp.ndarray,
        new_size: Tuple[int],
        old_size: Tuple[int] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ):
        previous_dtype = pos_embed.value.dtype

        num_new_tokens = new_size[0] * new_size[
            1] + self.num_embedded_prefix_tokens

        if num_new_tokens == self.embed_len and new_size[0] == new_size[1]:
            return pos_embed

        if old_size is None:
            hw = int(math.sqrt(self.num_patches))
            old_size = hw, hw

        prefix_embed = pos_embed[:, :self.
                                 num_prefix_tokens] if self.num_prefix_tokens else None
        pos_embed = pos_embed[:, self.num_prefix_tokens:]

        pos_embed = pos_embed.astype("float32")
        pos_embed = jnp.reshape(pos_embed,
                                (1, old_size[0], old_size[1], self.embed_dim))

        pos_embed = jax.image.resize(
            pos_embed,
            (1, h0, w0, dim),
            method=interpolation,
            antialias=antialias,
        )
        pos_embed = pos_embed.reshape(1, -1, embed_dim).astype(previous_dtype)

        if prefix_embed is not None:
            pro_embed = jnp.concatenate(prefix_embed + [pos_embed], axis=1)

        return pos_embed

    def _pos_embed(self, x: jnp.ndarray, h: int, w: int):
        if self.pos_embed is None:
            return jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = self.resample_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                antialias=self.interpolate_antialias)
            x = jnp.reshape(x, (B, -1, C))
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            # Broadcast cls_token to match batch size
            cls_token_value = self.cls_token.value
            expanded_cls_token = jnp.broadcast_to(
                cls_token_value, (x.shape[0], 1, cls_token_value.shape[-1]))
            to_cat.append(expanded_cls_token)

        # TODO: review cat orders, and pos_embed on x without reg tokens
        if self.register_tokens is not None:
            register_tokens_value = self.register_tokens.value
            expanded_register_tokens = jnp.broadcast_to(
                register_tokens_value, (x.shape[0], self.num_register_tokens,
                                        register_tokens_value.shape[-1]))
            to_cat.append(expanded_register_tokens)

        if self.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
        elif self.pos_embed_reg_tokens:
            # Timm impl.
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
            x = x + pos_embed
        else:
            # Similar to original dinov2 impl.
            # https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
            x = jnp.concatenate(to_cat[:1] + [x], axis=1)
            x = x + pos_embed
            if self.register_tokens is not None:
                x = jnp.concatenate([x[:, :1], to_cat[1], x[:, 1:]], axis=1)

        return x

    def features(self, x):
        N, H, W, C = x.shape

        x = self.patch_embed(x)
        x = self._pos_embed(x, h=H, w=W)

        for i in range(self.depth):
            _block = getattr(self, f"blocks.{i}")
            x = _block(x)

        return x

    def forward_features(self, x):
        x = self.features(x)
        x_norm = self.norm(x)

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1:self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
            "x_prenorm": x,
        }

    def __call__(self, x: jnp.ndarray):
        x = self.features(x)
        x = self.norm(x)
        x = self.head(x)

        return x
