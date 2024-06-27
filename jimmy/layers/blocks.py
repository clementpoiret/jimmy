import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nnx.module import first_from


class LayerScale(nnx.Module):

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        self.gamma = nnx.Param(
            init_values * nnx.initializers.ones(rngs.params(), [dim]),)

    def __call__(self, x):
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
        self.deterministic = deterministic
        self.rng_collection = rng_collection
        self.rngs = rngs

        Ellipsis

    def __call__(
        self,
        x,
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
