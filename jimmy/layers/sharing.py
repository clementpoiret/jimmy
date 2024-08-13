import jax.numpy as jnp
from flax import nnx
from jax.lax import fori_loop, switch


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


class LayerSharing(nnx.Module):
    """
    Layer Sharing wrapper responsible for repeating a Callable,
    while learning LoRA params. This is similar to what is done in MobileLLM [1]
    and Zamba2 [2]. It allows to use more FLOPs without having to store more params.

    Depending on the position of this wrapper (block-level or layer-level), given three
    callables A, B and C, you can produce those to kind of repetition patterns:
    - ABCABC,
    - AABBCC.

    Note:
        Repeating a layer twice requires it to produce outputs of the same shape as its inputs

    References:
        [1] Liu, et al., MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases (2024).
        [2] https://github.com/Zyphra/Zamba2
    """

    lora: bool = True
    repetitions: int = 2
    lora_kwargs = {"rank": 8, "alpha": 16, "drop_rate": 0.0}

    def __init__(
        self,
        dim: int,
        f: nnx.Module,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.__dict__.update(**kwargs)

        self.loras = ([
            LoRA(
                in_features=dim,
                out_features=dim,
                rngs=rngs,
                **self.lora_kwargs,
            ) for i in range(self.repetitions)
        ] if self.lora else None)

        self.f = f

    def forward(self, i: int, x: jnp.ndarray):
        if self.loras is None:
            return self.f(x)

        def apply_lora(j, x):
            return self.loras[j](x)

        lora_output = switch(
            i, [lambda x: apply_lora(j, x) for j in range(self.repetitions)],
            x)
        return self.f(x) + lora_output

    def __call__(self, x: jnp.ndarray):
        x = fori_loop(0, self.repetitions, self.forward, x)

        return x
