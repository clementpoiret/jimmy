import jax
import jax.numpy as jnp
from flax import nnx

from jimmy.layers.attention import Attention
from jimmy.layers.blocks import Block
from jimmy.layers.mlp import Mlp
from jimmy.layers.patch_embed import PatchEmbed

nnx.Linear(3, 32, rngs=nnx.Rngs(0), name="test")


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
        reg_token: int = 1,
        pos_embed: str = "learn",
        block: nnx.Module = Block,
        attention: nnx.Module = Attention,
        mlp: nnx.Module = Mlp,
        embed_layer: nnx.Module = PatchEmbed,
        rngs: nnx.Rngs = None,
    ):
        assert pos_embed in ("", "none", "learn")

        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token

        self.patch_embed = PatchEmbed(patch_size=patch_size,
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
            ) if reg_tokens else None)

        embed_len = (img_size // patch_size)**2 + self.num_prefix_tokens
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
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path_rate=drop_path_rate,
                rngs=rngs,
            )
            setattr(self, f"blocks.{i}", _block)

        self.norm = nnx.LayerNorm(num_features=dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = self.patch_embed(x)

        # TODO: pos embed

        for i in range(self.depth):
            _block = getattr(self, f"blocks.{i}")
            x = _block(x)

        x = self.norm(x)

        return x
