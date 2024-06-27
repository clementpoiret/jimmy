from flax import nnx


class LayerScale:

    def __init__(
        self,
        rngs: nnx.Rngs,
        dim: int,
        init_values: float = 1e-5,
    ):
        self.gamma = nnx.Param(
            init_values * nnx.initializers.ones(rngs.params(), [dim]),)

    def __call__(self, x):
        return x * self.gamma
