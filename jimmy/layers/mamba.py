import math
from typing import Callable

import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import dtypes, random

from jimmy.ops.scan import selective_scan


def custom_uniform(scale, dtype=jnp.float_):
    """Builds an initializer that returns real uniformly-distributed random arrays.

    Args:
        scale: optional; the upper bound of the random distribution.
        dtype: optional; the initializer's default dtype.

    Returns:
        An initializer that returns arrays whose values are uniformly distributed in
        the range ``(-scale, scale)``.
    """

    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return random.uniform(key, shape, dtype, minval=-scale, maxval=scale)

    return init


def custom_tensor(tensor, dtype=jnp.float_):
    """Builds an initializer that returns real uniformly-distributed random arrays.

    Args:
        scale: optional; the upper bound of the random distribution.
        dtype: optional; the initializer's default dtype.

    Returns:
        An initializer that returns arrays whose values are uniformly distributed in
        the range ``(-scale, scale)``.
    """

    def init():
        return tensor

    return init


class MambaVisionMixer(nnx.Module):
    """MambaVision Mixer from Ali Hatamizadeh and Jan Kautz.

    Definitions:
        - b: batch size (`B` in [1]),
        - l: sequence length (`L` in [1]),
        - d and d_model: hidden dim,
        - n and d_state: latent space dim (`N` in [1]),
        - expand: expansion factor (`E` in [1]),
        - d_in and d_inner: d*expand (`D` in [1]),
        - A, B, C, D: state space parameters (See any state space representation formula),
        - dt or delta: input-dependent step size.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: int | None = None,
        rngs: nnx.Rngs = None,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model /
                                 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nnx.Linear(d_model,
                                  self.d_inner,
                                  use_bias=bias,
                                  rngs=rngs)
        self.x_proj = nnx.Linear(self.d_inner // 2,
                                 self.dt_rank + self.d_state * 2,
                                 use_bias=False,
                                 rngs=rngs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            kernel_init = nnx.initializers.constant(dt_init_std)
        elif dt_init == "random":
            kernel_init = custom_uniform(dt_init_std)
        else:
            raise NotImplementedError

        key = rngs.params()
        rand_vals = random.uniform(key, (self.d_inner // 2,))
        dt = jnp.exp(rand_vals * (math.log(dt_max) - math.log(dt_min)) +
                     math.log(dt_min))
        dt = jnp.clip(dt, a_min=dt_init_floor)

        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_proj = nnx.Linear(self.dt_rank,
                                  self.d_inner // 2,
                                  use_bias=True,
                                  kernel_init=kernel_init,
                                  bias_init=lambda *_: inv_dt,
                                  rngs=rngs)
        # WARNING: check no reinit for dt_proj bias

        A = jnp.arange(1, self.d_state + 1, dtype=jnp.float32)
        A = jnp.tile(A, (self.d_inner // 2, 1))

        A_log = jnp.log(A)
        self.A_log = nnx.Param(custom_tensor(A_log))
        self.D = nnx.Param(
            nnx.initializers.ones(rngs.params(), [self.d_inner // 2]))

        self.out_proj = nnx.Linear(self.d_inner,
                                   self.d_model,
                                   use_bias=True,
                                   rngs=rngs)

        self.conv1d_x = nnx.Conv(in_features=self.d_inner // 2,
                                 out_features=self.d_inner // 2,
                                 kernel_size=(self.d_conv,),
                                 feature_group_count=self.d_inner // 2,
                                 use_bias=conv_bias // 2 > 0,
                                 padding="SAME",
                                 rngs=rngs)
        self.conv1d_z = nnx.Conv(in_features=self.d_inner // 2,
                                 out_features=self.d_inner // 2,
                                 kernel_size=(self.d_conv,),
                                 feature_group_count=self.d_inner // 2,
                                 use_bias=conv_bias // 2 > 0,
                                 padding="SAME",
                                 rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        _, L, _ = x.shape

        xz = self.in_proj(x)  # 1,64, 32
        x, z = jnp.split(xz, 2, axis=-1)  # 1, 64, 16 x2
        A = -jnp.exp(self.A_log.value())  # 16, 16
        x = nnx.silu(self.conv1d_x(x))  # 1, 64, 16
        z = nnx.silu(self.conv1d_z(z))  # 1, 64, 16
        x_dbl = self.x_proj(rearrange(x, "b l d -> (b l) d"))  # 64, 33
        dt, B, C = jnp.split(x_dbl, [self.dt_rank, self.d_state + self.dt_rank],
                             axis=-1)  # 64, 1; 64, 16; 64, 16
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=L)  # 1, 16, 64
        B = rearrange(B, "(b l) d -> b d l", l=L)  # 1, 16, 64
        C = rearrange(C, "(b l) d -> b d l", l=L)  # 1, 16, 64

        x = rearrange(x, "b l d -> b d l")  # 1, 16, 64
        z = rearrange(z, "b l d -> b d l")  # 1, 16, 64
        y = selective_scan(x,
                           dt,
                           A,
                           B,
                           C,
                           self.D.value,
                           delta_bias=self.dt_proj.bias.value,
                           delta_softplus=True)  # 1, 16, 64

        y = jnp.concatenate([y, z], axis=1)  # 1, 32, 64
        y = rearrange(y, "b d l -> b l d")  # 1, 64, 32

        out = self.out_proj(y)  # 1, 64, 16

        return out
