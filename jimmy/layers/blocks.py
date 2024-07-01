from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nnx.module import first_from

from jimmy.layers.attention import Attention
from jimmy.layers.mlp import Mlp


class Identity(nnx.Module):

    def __call__(self, x: jnp.ndarray):
        return x


class LayerScale(nnx.Module):

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        self.gamma = nnx.Param(
            init_values * nnx.initializers.ones(rngs.params(), [dim]),)

    def __call__(self, x: jnp.ndarray):
        return x * self.gamma


class DropPath(nnx.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self,
                 drop_prob: float = 0.,
                 scale_by_keep: bool = True,
                 deterministic: bool = False,
                 rng_collection: str = "dropout",
                 rngs: nnx.Rngs = None):
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.deterministic = deterministic
        self.rng_collection = rng_collection
        self.rngs = rngs

        Ellipsis

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to Dropout as
            either a __call__ argument or class attribute""",
        )

        if (self.drop_prob == 0.0) or deterministic:
            return x

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.drop_prob == 1.0:
            return jnp.zeros_like(x)

        rngs = first_from(
            rngs,
            self.rngs,
            error_msg=
            """`deterministic` is False, but no `rngs` argument was provided to
            Dropout as either a __call__ argument or class attribute.""",
        )
        rng = rngs[self.rng_collection]()

        keep_prob = 1.0 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jax.random.bernoulli(rng, p=keep_prob, shape=shape)

        if keep_prob > 0.0 and scale_by_keep:
            random_tensor /= keep_prob

        return x * random_tensor
        # return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class Block(nnx.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        attention: nnx.Module = Attention,
        act_layer: Callable = nnx.gelu,
        norm_layer: nnx.Module = nnx.LayerNorm,
        ffn_layer: nnx.Module = Mlp,
        rngs: nnx.Rngs = None,
    ):
        self.norm1 = norm_layer(num_features=dim, rngs=rngs)
        self.attn = attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.ls1 = LayerScale(dim, init_values,
                              rngs=rngs) if init_values else Identity()
        self.drop_path1 = DropPath(drop_path,
                                   rngs=rngs) if drop_path > 0. else Identity()

        self.norm2 = norm_layer(num_features=dim, rngs=rngs)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            dropout_rate=proj_drop,
            bias=ffn_bias,
            rngs=rngs,
        )
        self.ls2 = LayerScale(dim, init_values,
                              rngs=rngs) if init_values else Identity()
        self.drop_path2 = DropPath(drop_path,
                                   rngs=rngs) if drop_path > 0. else Identity()

    def __call__(self, x: jnp.ndarray):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
