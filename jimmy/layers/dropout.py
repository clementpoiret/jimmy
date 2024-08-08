import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nnx.module import first_from


class DropPath(nnx.Module):
    """Drop paths (Stochastic Depth) per sample."""

    scale_by_keep: bool = True
    deterministic: bool = False
    rng_collection: str = "dropout"

    def __init__(
        self,
        drop_prob: float,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        """Initialize the DropPath module.

        Args:
            drop_prob (float, optional): Probability of dropping a path. Defaults to 0.
            scale_by_keep (bool, optional): Whether to scale the kept values. Defaults to True.
            deterministic (bool, optional): Whether to use deterministic behavior. Defaults to False.
            rng_collection (str, optional): Name of the RNG collection. Defaults to "dropout".
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        self.__dict__.update(**kwargs)

        self.drop_prob = drop_prob
        self.rngs = rngs

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> jnp.ndarray:
        """Apply DropPath to the input.

        Args:
            x (jnp.ndarray): Input array.
            deterministic (bool | None, optional): Override for deterministic behavior. Defaults to None.
            rngs (nnx.Rngs | None, optional): Override for random number generator state. Defaults to None.

        Returns:
            jnp.ndarray: Output after applying DropPath.
        """
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
            error_msg="""`deterministic` is False, but no `rngs` argument was provided to
            Dropout as either a __call__ argument or class attribute.""",
        )
        rng = rngs[self.rng_collection]()

        keep_prob = 1.0 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jax.random.bernoulli(rng, p=keep_prob, shape=shape)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor /= keep_prob

        return x * random_tensor
        # return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))
