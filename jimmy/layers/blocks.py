from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nnx.module import first_from

from jimmy.layers.attention import Attention
from jimmy.layers.mlp import Mlp


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
                 deterministic: bool = False,
                 rng_collection: str = "dropout",
                 rngs: nnx.Rngs = None):
        self.drop_prob = drop_prob
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

        if (self.rate == 0.0) or deterministic:
            return x

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
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
        shape = (x.shape[0], 1, 1, 1)
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=shape)

        # return x / keep_prob * random_tensor
        return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class Block(nnx.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: Callable = nnx.gelu,
        norm_layer: nnx.Module = nnx.LayerNorm,
        mlp_layer: nnx.Module = Mlp,
        rngs: nnx.Rngs = None,
    ):
        self.norm1 = norm_layer(num_features=dim, rngs=rngs)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.ls1 = None
        if init_values:
            self.ls1 = LayerScale(dim, init_values, rngs=rngs)
        self.drop_path1 = None
        if drop_path > 0.:
            self.drop_path1 = DropPath(drop_path, rngs=rngs)

        self.norm2 = norm_layer(num_features=dim, rngs=rngs)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            dropout_rate=proj_drop,
            rngs=rngs,
        )
        self.ls2 = None
        if init_values:
            self.ls2 = LayerScale(dim, init_values, rngs=rngs)
        self.drop_path2 = None
        if drop_path > 0.:
            self.drop_path2 = DropPath(drop_path, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = self.ls1(self.attn(self.norm1(x)))
        if self.drop_path1:
            x = self.drop_path1(x)

        x = self.ls2(self.mlp(self.norm2(x)))
        if self.drop_path2:
            x = self.drop_path2(x)

        return x
