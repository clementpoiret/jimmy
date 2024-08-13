import jax.numpy as jnp
from flax import nnx


class LoRA(nnx.Module):
    """
    Very simple implementation of a LoRA[1] layer.

    References:
        [1] Hu, et al., LoRA: Low-Rank Adaptation of Large Language Models (2021).
    """

    rank: int = 8
    alpha: int = 16
    drop_rate: float = 0.0

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs=nnx.Rngs,
        **kwargs,
    ):
        self.__dict__.update(**kwargs)

        self.A = nnx.Linear(in_features, self.rank, use_bias=False, rngs=rngs)
        self.B = nnx.Linear(
            self.rank,
            out_features,
            use_bias=False,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(self.drop_rate, rngs=rngs)

        self.scaling = self.alpha / self.rank

    def __call__(self, x: jnp.ndarray):
        x = self.dropout(x)

        x = self.B(self.A(x))

        return x * self.scaling
