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
        scale (float): The upper bound of the random distribution.
        dtype (jnp.dtype, optional): The initializer's default dtype. Defaults to jnp.float_.

    Returns:
        Callable: An initializer function that returns arrays whose values are uniformly
                  distributed in the range ``(-scale, scale)``.
    """

    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return random.uniform(key, shape, dtype, minval=-scale, maxval=scale)

    return init


def custom_tensor(tensor, dtype=jnp.float_):
    """Builds an initializer that returns a predefined tensor.

    Args:
        tensor (jnp.ndarray): The tensor to be returned by the initializer.
        dtype (jnp.dtype, optional): The initializer's default dtype. Defaults to jnp.float_.

    Returns:
        Callable: An initializer function that returns the predefined tensor.
    """

    def init():
        return tensor

    return init


class MambaVisionMixer(nnx.Module):
    """MambaVision Mixer from Ali Hatamizadeh and Jan Kautz.

    This class implements the MambaVision Mixer, a novel architecture for vision tasks.

    Attributes:
        d_model (int): Hidden dimension size.
        d_state (int): Latent space dimension size.
        d_conv (int): Convolution dimension size.
        expand (int): Expansion factor.
        d_inner (int): Inner dimension size (d_model * expand).
        dt_rank (int): Rank for delta time projection.
        use_fast_path (bool): Whether to use the fast path computation.
        layer_idx (int | None): Layer index, if applicable.

    Args:
        d_model (int): Hidden dimension size.
        d_state (int, optional): Latent space dimension size. Defaults to 16.
        d_conv (int, optional): Convolution dimension size. Defaults to 4.
        expand (int, optional): Expansion factor. Defaults to 2.
        dt_rank (str, optional): Rank for delta time projection. Defaults to "auto".
        dt_min (float, optional): Minimum delta time value. Defaults to 0.001.
        dt_max (float, optional): Maximum delta time value. Defaults to 0.1.
        dt_init (str, optional): Initialization method for delta time. Defaults to "random".
        dt_scale (float, optional): Scaling factor for delta time. Defaults to 1.0.
        dt_init_floor (float, optional): Floor value for delta time initialization. Defaults to 1e-4.
        conv_bias (bool, optional): Whether to use bias in convolutions. Defaults to True.
        bias (bool, optional): Whether to use bias in linear layers. Defaults to False.
        use_fast_path (bool, optional): Whether to use the fast path computation. Defaults to True.
        layer_idx (int | None, optional): Layer index, if applicable. Defaults to None.
        rngs (nnx.Rngs, optional): Random number generators. Defaults to None.

    Notes:
        - b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
        - l: sequence length                  (`L` in [1] Algorithm 2)
        - d or d_model: hidden dim
        - n or d_state: latent state dim      (`N` in [1] Algorithm 2)
        - expand: expansion factor            (`E` in [1] Section 3.4)
        - d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
        - A, B, C, D: state space parameters  (See any state space representation formula)
            (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
        - dt or delta: input-dependent step size
        - dt_rank: rank of dt                 (See [1] Section 3.6 "Parameterization of ∆")

    References:
        [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
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
        """Forward pass of the MambaVisionMixer.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        _, L, _ = x.shape

        xz = self.in_proj(x)
        x, z = jnp.split(xz, 2, axis=-1)
        A = -jnp.exp(self.A_log.value())
        x = nnx.silu(self.conv1d_x(x))
        z = nnx.silu(self.conv1d_z(z))
        x_dbl = self.x_proj(rearrange(x, "b l d -> (b l) d"))
        dt, B, C = jnp.split(x_dbl, [self.dt_rank, self.d_state + self.dt_rank],
                             axis=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=L)
        B = rearrange(B, "(b l) d -> b d l", l=L)
        C = rearrange(C, "(b l) d -> b d l", l=L)

        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")
        y = selective_scan(x,
                           dt,
                           A,
                           B,
                           C,
                           self.D.value,
                           delta_bias=self.dt_proj.bias.value,
                           delta_softplus=True)

        y = jnp.concatenate([y, z], axis=1)
        y = rearrange(y, "b d l -> b l d")

        out = self.out_proj(y)

        return out
