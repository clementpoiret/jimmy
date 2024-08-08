import math

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import nnx
from jax import random

from jimmy.ops.scan import non_casual_linear_attn, selective_scan, ssd
from jimmy.utils import custom_uniform

from .configs import MambaConfig
from .norm import RMSNormGated

# TODO: Inference cache as in https://github.com/walln/scratch/blob/ab0b6b891830375b7aa64c8e46e77783b843f5ca/src/scratch/language_modeling/mamba/mamba.py
# TODO: Learnable init state (as in mamba.py)


class MambaVisionMixer(nnx.Module):
    """MambaVision Mixer from Ali Hatamizadeh and Jan Kautz.

    This class implements the MambaVision Mixer, a novel architecture for vision tasks
    that combines the strengths of state space models and vision transformers.

    Attributes:
        config (MambaConfig): Configuration object containing model hyperparameters.
        in_proj (nnx.Linear): Input projection layer.
        x_proj (nnx.Linear): Projection layer for x.
        dt_proj (nnx.Linear): Projection layer for delta time.
        A_log (nnx.Param): Logarithm of the state transition matrix A.
        D (nnx.Param): Direct feedthrough matrix D.
        out_proj (nnx.Linear): Output projection layer.
        conv1d_x (nnx.Conv): 1D convolution for x.
        conv1d_z (nnx.Conv): 1D convolution for z.

    Args:
        config (MambaConfig): Configuration object for the MambaVisionMixer.
        rngs (nnx.Rngs): Random number generators for parameter initialization.

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
            (https://arxiv.org/abs/2312.00752)
    """

    def __init__(
        self,
        config: MambaConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config

        self.in_proj = nnx.Linear(config.d_model,
                                  config.d_inner,
                                  use_bias=config.bias,
                                  rngs=rngs)
        self.x_proj = nnx.Linear(
            config.d_inner // 2,
            config.dt_rank + config.d_state * 2,
            use_bias=False,
            rngs=rngs,
        )

        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            kernel_init = nnx.initializers.constant(dt_init_std)
        elif config.dt_init == "random":
            kernel_init = custom_uniform(dt_init_std)
        else:
            raise NotImplementedError

        key = rngs.params()
        rand_vals = random.uniform(key, (config.d_inner // 2, ))
        dt = jnp.exp(rand_vals *
                     (math.log(config.dt_max) - math.log(config.dt_min)) +
                     math.log(config.dt_min))
        dt = jnp.clip(dt, a_min=config.dt_init_floor)

        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_proj = nnx.Linear(
            config.dt_rank,
            config.d_inner // 2,
            use_bias=True,
            kernel_init=kernel_init,
            bias_init=lambda *_: inv_dt,
            rngs=rngs,
        )
        # WARNING: check no reinit for dt_proj bias

        A = jnp.arange(1, config.d_state + 1, dtype=jnp.float32)
        A = jnp.tile(A, (config.d_inner // 2, 1))

        A_log = jnp.log(A)
        self.A_log = nnx.Param(A_log)
        self.D = nnx.Param(
            nnx.initializers.ones(rngs.params(), [config.d_inner // 2]))

        self.out_proj = nnx.Linear(config.d_inner,
                                   config.d_model,
                                   use_bias=True,
                                   rngs=rngs)

        self.conv1d_x = nnx.Conv(
            in_features=config.d_inner // 2,
            out_features=config.d_inner // 2,
            kernel_size=(config.d_conv, ),
            feature_group_count=config.d_inner // 2,
            use_bias=config.conv_bias,
            padding="SAME",
            rngs=rngs,
        )
        self.conv1d_z = nnx.Conv(
            in_features=config.d_inner // 2,
            out_features=config.d_inner // 2,
            kernel_size=(config.d_conv, ),
            feature_group_count=config.d_inner // 2,
            use_bias=config.conv_bias,
            padding="SAME",
            rngs=rngs,
        )

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
        A = -jnp.exp(self.A_log.value)
        x = nnx.silu(self.conv1d_x(x))
        z = nnx.silu(self.conv1d_z(z))
        x_dbl = self.x_proj(rearrange(x, "b l d -> (b l) d"))
        dt, B, C = jnp.split(
            x_dbl,
            [self.config.dt_rank, self.config.d_state + self.config.dt_rank],
            axis=-1,
        )

        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=L)
        B = rearrange(B, "(b l) d -> b d l", l=L)
        C = rearrange(C, "(b l) d -> b d l", l=L)

        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")
        y = selective_scan(
            x,
            dt,
            A,
            B,
            C,
            self.D.value,
            delta_bias=self.dt_proj.bias.value,
            delta_softplus=True,
        )

        y = jnp.concatenate([y, z], axis=1)
        y = rearrange(y, "b d l -> b l d")

        out = self.out_proj(y)

        return out


class Mamba2Mixer(nnx.Module):
    """Mamba2 Mixer.

    This class implements the a Mamba2 Mixer using State Space Duality (SSD),
    from Mamba2 [1]. Also supports implementation details from Visual State Space
    Duality (VSSD) [2].

    Attributes:
        config (MambaConfig): Configuration object containing model hyperparameters.
        in_proj (nnx.Linear): Input projection layer.
        conv (nnx.Conv): Convolutional layer for processing input.
        dt_bias (nnx.Param): Bias for delta time.
        A_log (nnx.Param): Logarithm of the state transition matrix A.
        D (nnx.Param): Direct feedthrough matrix D.
        norm (nnx.RMSNorm): Root Mean Square Layer Normalization.
        out_proj (nnx.Linear): Output projection layer.

    Args:
        config (MambaConfig): Configuration object for the Mamba2VisionMixer.
        rngs (nnx.Rngs): Random number generators for parameter initialization.

    Notes:
        This implementation is heavily based on wlln/scratch.

    References:
        [1] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
            (https://arxiv.org/abs/2401.04054)
        [2] VSSD: Vision Mamba with Non-Casual State Space Duality
            (https://arxiv.org/abs/2407.18559)
    """

    def __init__(
        self,
        config: MambaConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config

        d_in_proj = (2 * config.d_inner +
                     2 * config.n_groups * config.d_state + config.n_heads)
        self.in_proj = nnx.Linear(config.d_model,
                                  d_in_proj,
                                  use_bias=config.bias,
                                  rngs=rngs)

        conv_dim = config.d_inner + 2 * config.n_groups * config.d_state
        self.conv = nnx.Conv(
            in_features=conv_dim,
            out_features=conv_dim,
            kernel_size=(config.d_conv, ),
            feature_group_count=conv_dim,
            padding=[((config.d_conv - 1) // 2, (config.d_conv - 1) // 2)],
            use_bias=config.conv_bias,
            rngs=rngs,
        )

        if config.learnable_init_states:
            self.init_states = nnx.Param(jnp.ones(config.n_heads), rngs=rngs)

        key = rngs.params()
        rand_vals = random.uniform(key, (config.n_heads, ))
        dt = jnp.exp(rand_vals *
                     (math.log(config.dt_max) - math.log(config.dt_min)) +
                     math.log(config.dt_min))
        dt = jnp.clip(dt, a_min=config.dt_init_floor)
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_bias = nnx.Param(inv_dt, rngs=rngs)

        A_min, A_max = config.A_init_range
        key = rngs.params()
        A = random.uniform(key, (config.n_heads, ), minval=A_min, maxval=A_max)
        A_log = jnp.log(A)
        self.A_log = nnx.Param(A_log, rngs=rngs)

        self.D = nnx.Param(jnp.ones(config.n_heads), rngs=rngs)

        self.norm = nnx.LayerNorm(config.d_inner, rngs=rngs)
        self.out_proj = nnx.Linear(config.d_inner,
                                   config.d_model,
                                   use_bias=config.bias,
                                   rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        """Forward pass of the VMamba2Mixer using non-casual attention duality.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        batch = x.shape[0]

        A = -jnp.exp(self.A_log.value)
        zxbcdt = self.in_proj(x)

        z, xbc, dt = jnp.split(
            zxbcdt,
            [self.config.d_inner, zxbcdt.shape[-1] - self.config.n_heads],
            axis=-1,
        )

        dt = jax.nn.softplus(dt + self.dt_bias.value)

        # Pad or truncate the xbc tensor to match the conv kernel size
        # xbc_rearranged = rearrange(xbc, "b l d -> b d l")
        # conv_state = pad_or_truncate_to_length(xbc_rearranged, self.config.d_conv)

        # apply 1d convolution and silu activation
        xbc_conv = self.conv(xbc)
        xbc_silu = jax.nn.silu(xbc_conv[:, :x.shape[1], :])

        # split the conv state into the conv kernel and the conv state
        x, B, C = jnp.split(xbc_silu, self.config.indices_xBC, axis=-1)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.config.head_dim)

        if self.config.linear_attn_duality:
            y = non_casual_linear_attn(x,
                                       dt=dt,
                                       A=A,
                                       B=B,
                                       C=C,
                                       D=self.D.value,
                                       n_groups=self.config.n_groups)
        else:
            # apply ssd function
            # TODO: Bidirectional
            initial_states = (repeat(
                self.init_states.value,
                "h -> b c h p n",
                b=batch,
                c=x.shape[1] // self.config.chunk_size,
                p=x.shape[-1],
                n=B.shape[-1],
            ) if self.config.learnable_init_states else None)
            y, ssm_state = ssd(
                x * jnp.expand_dims(dt, axis=-1),
                A * dt,
                rearrange(B, "b l (g n) -> b l g n", g=self.config.n_groups),
                rearrange(C, "b l (g n) -> b l g n", g=self.config.n_groups),
                self.config.chunk_size,
                initial_states=initial_states,
            )

            # Combine the output of the ssd function with the input and rearrange
            y = y + x * jnp.expand_dims(self.D.value, axis=-1)

        y = rearrange(y, "b l h p -> b l (h p)")

        # apply the output projection
        if isinstance(self.norm, RMSNormGated):
            y = self.norm(y, nnx.silu(z))
        else:
            # Should be LayerNorm
            y = self.norm(y) * z

        y = self.out_proj(y)

        return y


class Mamba2VisionMixer(nnx.Module):
    """Mamba2Vision Mixer.

    This class implements the a Mamba2Vision Mixer using State Space Duality (SSD),
    from Mamba2 [1]. It extends the MambaVisionMixer by replacing SSM with SSD, leading to
    enhanced efficiency, and maybe accuracy.

    Attributes:
        config (MambaConfig): Configuration object containing model hyperparameters.
        in_proj (nnx.Linear): Input projection layer.
        conv (nnx.Conv): Convolutional layer for processing input.
        dt_bias (nnx.Param): Bias for delta time.
        A_log (nnx.Param): Logarithm of the state transition matrix A.
        D (nnx.Param): Direct feedthrough matrix D.
        norm (nnx.RMSNorm): Root Mean Square Layer Normalization.
        out_proj (nnx.Linear): Output projection layer.

    Args:
        config (MambaConfig): Configuration object for the Mamba2VisionMixer.
        rngs (nnx.Rngs): Random number generators for parameter initialization.

    Notes:
        This implementation is heavily based on wlln/scratch.

    References:
        [1] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
            (https://arxiv.org/abs/2401.04054)
    """

    def __init__(
        self,
        config: MambaConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config

        # d_in_proj = 2 * config.d_inner + 2 * config.d_state + config.n_heads
        self.in_proj = nnx.Linear(config.d_model,
                                  2 * config.d_inner,
                                  use_bias=config.bias,
                                  rngs=rngs)
        self.x_proj = nnx.Linear(
            config.d_inner,
            config.n_heads + config.d_state * 2,
            use_bias=False,
            rngs=rngs,
        )
        self.conv1d_x = nnx.Conv(
            in_features=config.d_inner,
            out_features=config.d_inner,
            kernel_size=(config.d_conv, ),
            feature_group_count=config.d_inner,
            use_bias=config.conv_bias,
            padding="SAME",
            rngs=rngs,
        )
        self.conv1d_z = nnx.Conv(
            in_features=config.d_inner,
            out_features=config.d_inner,
            kernel_size=(config.d_conv, ),
            feature_group_count=config.d_inner,
            use_bias=config.conv_bias,
            padding="SAME",
            rngs=rngs,
        )

        # self.in_proj = nnx.Linear(
        #     config.d_model, d_in_proj, use_bias=config.bias, rngs=rngs
        # )

        key = rngs.params()
        rand_vals = random.uniform(key, (config.n_heads, ))
        dt = jnp.exp(rand_vals *
                     (math.log(config.dt_max) - math.log(config.dt_min)) +
                     math.log(config.dt_min))
        dt = jnp.clip(dt, a_min=config.dt_init_floor)
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_bias = nnx.Param(inv_dt)

        A_min, A_max = config.A_init_range
        key = rngs.params()
        A = random.uniform(key, (config.n_heads, ), minval=A_min, maxval=A_max)
        A_log = jnp.log(A)
        self.A_log = nnx.Param(A_log, rngs=rngs)

        self.D = nnx.Param(jnp.ones(config.n_heads), rngs=rngs)

        self.norm = RMSNormGated(config.d_inner, rngs=rngs)
        self.out_proj = nnx.Linear(config.d_inner,
                                   config.d_model,
                                   use_bias=config.bias,
                                   rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        """Forward pass of the MambaVisionMixer.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        _, L, _ = x.shape

        A = -jnp.exp(self.A_log.value)

        xz = self.in_proj(x)
        x, z = jnp.split(xz, 2, axis=-1)
        x = nnx.silu(self.conv1d_x(x))
        z = nnx.silu(self.conv1d_z(z))

        x_dbl = self.x_proj(rearrange(x, "b l d -> (b l) d"))
        dt, B, C = jnp.split(
            x_dbl,
            [self.config.n_heads, self.config.d_state + self.config.n_heads],
            axis=-1,
        )
        dt = rearrange(dt, "(b l) n -> b l n", l=L)
        dt = jax.nn.softplus(dt + self.dt_bias.value)

        B = rearrange(B, "(b l) n -> b l n", l=L)
        C = rearrange(C, "(b l) n -> b l n", l=L)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.config.head_dim)

        # apply ssd function
        y, ssm_state = ssd(
            x * jnp.expand_dims(dt, axis=-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.config.chunk_size,
        )

        # Combine the output of the ssd function with the input and rearrange
        y = y + x * jnp.expand_dims(self.D.value, axis=-1)
        y = rearrange(y, "b l h p -> b l (h p)")

        # apply the output projection
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y
