import jax
import jax.numpy as jnp
from flax import nnx

from jimmy.layers.attention import Attention
from jimmy.layers.blocks import Block
from jimmy.layers.mlp import Mlp
from jimmy.layers.patch_embed import PatchEmbed

# TODO: dynamic_img_size
# TODO: dynamic_img_pad


class DinoV2(nnx.Module):

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 14,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        class_token: bool = True,
        # reg_tokens: int = 1,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        # dynamic_img_size: bool = False,
        # dynamic_img_pad: bool = False,
        block: nnx.Module = Block,
        attention: nnx.Module = Attention,
        mlp: nnx.Module = Mlp,
        embed_layer: nnx.Module = PatchEmbed,
        rngs: nnx.Rngs = None,
    ):
        assert pos_embed in ("", "none", "learn")

        # self.embed_dim = embed_dim
        self.depth = depth
        self.num_prefix_tokens = 1 if class_token else 0
        # self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = 0  # reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        # self.dynamic_img_size = dynamic_img_size

        self.patch_embed = embed_layer(patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dim=embed_dim,
                                       rngs=rngs)
        self.cls_token = nnx.Param(
            nnx.initializers.zeros(rngs.params(),
                                   [1, 1, embed_dim]),) if class_token else None
        # self.reg_tokens = nnx.Param(
        #     nnx.initializers.zeros(
        #         rngs.params(),
        #         [1, reg_tokens, embed_dim],
        #     )) if reg_tokens else None

        num_patches = (img_size // patch_size)**2
        embed_len = num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == "none":
            self.pos_embed = None
        else:
            self.pos_embed = nnx.Param(
                nnx.initializers.normal(.02)(rngs.params(),
                                             [1, embed_len, embed_dim]))

        # TODO: interpolate_pos_encoding

        # To respect the original naming
        for i in range(depth):
            _block = block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
                attention=attention,
                mlp_layer=mlp,
                init_values=1.0,
                rngs=rngs,
            )
            setattr(self, f"blocks.{i}", _block)

        self.norm = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)

    def _pos_embed(self, x: jnp.ndarray):
        if self.pos_embed is None:
            return jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        # TODO: handle dynamic image size
        to_cat = []
        if self.cls_token is not None:
            # Broadcast cls_token to match batch size
            cls_token_value = self.cls_token.value
            expanded_cls_token = jnp.broadcast_to(
                cls_token_value, (x.shape[0], 1, cls_token_value.shape[-1]))
            to_cat.append(expanded_cls_token)

        if self.no_embed_class:
            x = x + self.pos_embed
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
        else:
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
            x = x + self.pos_embed

        # TODO: pos_drop
        return x

    def __call__(self, x: jnp.ndarray):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        for i in range(self.depth):
            _block = getattr(self, f"blocks.{i}")
            x = _block(x)

        x = self.norm(x)

        return x
