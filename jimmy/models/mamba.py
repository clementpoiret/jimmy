from typing import Callable, List

import jax.numpy as jnp
from einops import reduce
from flax import nnx

from jimmy.layers import (Attention, ConvPatchEmbed, Identity, MambaVisionLayer,
                          MambaVisionMixer, Mlp)


def adaptive_avg_pool2d(x: jnp.ndarray):
    return reduce(x, "b h w c -> b c 1 1", "mean")


class MambaVision(nnx.Module):

    def __init__(
        self,
        in_features: int,
        dim: int,
        in_dim: int,
        depths: List[int],
        window_size: List[int],
        mlp_ratio: float,
        num_heads: List[int],
        drop_path_rate: float = 0.2,
        qkv_bias: bool = True,
        qkv_norm: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: float | None = None,
        init_values_conv: float | None = None,
        transformer_attention: Callable = Attention,
        mamba_mixer: Callable = MambaVisionMixer,
        act_layer: Callable = nnx.gelu,
        norm_layer: Callable = nnx.LayerNorm,
        ffn_layer: Callable = Mlp,
        num_classes: int = 1000,
    ):
        num_features = int(dim * 2**(len(depths) - 1))
        self.num_classes = num_classes

        self.patch_embed = ConvPatchEmbed(in_features=in_features,
                                          hidden_features=in_dim,
                                          out_features=dim,
                                          rngs=rngs)
        dpr = jnp.linspace(0, drop_path_rate, sum(depths))

        self.levels = []
        for i in range(len(depths)):
            conv = i > 2
            level = MambaVisionLayer(
                dim=int(dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                conv=conv,
                downsample=i < 3,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                ffn_bias=ffn_bias,
                proj_bias=proj_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                init_values=init_values,
                init_values_conv=init_values_conv,
                transformer_attention=transformer_attention,
                mamba_mixer=mamba_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                ffn_layer=ffn_layer,
                block_types=self._get_block_types(depths[i]),
                rngs=rngs)
            self.levels.append(level)

        self.norm = nnx.BatchNorm(num_features=num_features, rngs=rngs)
        self.head = nnx.Linear(num_features, num_classes,
                               rngs=rngs) if num_classes else Identity()

    def _get_block_types(l: int):
        first_half_size = (l + 1) // 2
        second_half_size = l // 2
        return ["mambavisionmixer"] * first_half_size + ["attention"
                                                        ] * second_half_size

    def forward_features(self, x: jnp.ndarray):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = adaptive_avg_pool2d(x)
        x = jnp.reshape(x, (x.shape[0], -1))

        return x

    def __call__(self, x: jnp.ndarray):
        x = self.forward_features(x)
        x = self.head(x)

        return x
