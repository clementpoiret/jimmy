import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import random

from jimmy.ops.scan import selective_scan, ssd
from jimmy.utils import custom_uniform, window_partition, window_reverse

from .attention import Attention
from .blocks import Block, ConvBlock
from .configs import MambaConfig
from .mlp import Mlp

# TODO: Inference cache


class Downsample(nnx.Module):
    """Downsampling block."""

    def __init__(self, dim: int, keep_dim: bool = False, rngs: nnx.Rngs = None):
        """Initialize the Block.

        Args:
            dim (int): Input dimension.
            keep_dim (bool): If True, maintain the resolution.
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nnx.Conv(
            in_features=dim,
            out_features=dim_out,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        return self.reduction(x)


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
        config: MambaConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config

        self.in_proj = nnx.Linear(
            config.d_model, config.d_inner, use_bias=config.bias, rngs=rngs
        )
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
        rand_vals = random.uniform(key, (config.d_inner // 2,))
        dt = jnp.exp(
            rand_vals * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
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
        self.D = nnx.Param(nnx.initializers.ones(rngs.params(), [config.d_inner // 2]))

        self.out_proj = nnx.Linear(
            config.d_inner, config.d_model, use_bias=True, rngs=rngs
        )

        self.conv1d_x = nnx.Conv(
            in_features=config.d_inner // 2,
            out_features=config.d_inner // 2,
            kernel_size=(config.d_conv,),
            feature_group_count=config.d_inner // 2,
            use_bias=config.conv_bias // 2 > 0,
            padding="SAME",
            rngs=rngs,
        )
        self.conv1d_z = nnx.Conv(
            in_features=config.d_inner // 2,
            out_features=config.d_inner // 2,
            kernel_size=(config.d_conv,),
            feature_group_count=config.d_inner // 2,
            use_bias=config.conv_bias // 2 > 0,
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


class InferenceCache(NamedTuple):
    """Inference Cache.

    Attributes:
        conv_state (batch_size, d_inner + 2 * d_state, d_conv): convolution state
        ssm_state (batch_size, n_heads, head_dim, d_state): SSM state
    """

    conv_state: jnp.ndarray
    ssm_state: jnp.ndarray

    @staticmethod
    def allocate(
        batch_size: int,
        config: MambaConfig,
    ):
        """Allocate InferenceCache.

        Args:
            batch_size: batch size
            config: MambaConfig

        Returns:
            InferenceCache
        """
        return InferenceCache(
            jnp.zeros((batch_size, config.d_inner + 2 * config.d_state, config.d_conv)),
            jnp.zeros((batch_size, config.n_heads, config.head_dim, config.d_state)),
        )


class Mamba2VisionMixer(nnx.Module):
    """Mamba2Vision Mixer.

    This class implements the MambaVision Mixer using Structured State Space Duality (SSD),
    from Mamba2 [1].

    Attributes:

    Args:
        rngs (nnx.Rngs, optional): Random number generators. Defaults to None.

    Notes:
        Heavily based on wlln/scratch.

    References:
        [1] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
    """

    def __init__(
        self,
        config: MambaConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config

        d_in_proj = 2 * config.d_inner + 2 * config.d_state + config.n_heads
        self.in_proj = nnx.Linear(
            config.d_model, d_in_proj, use_bias=config.bias, rngs=rngs
        )

        conv_dim = config.d_inner + 2 * config.d_state
        self.conv = nnx.Conv(
            in_features=conv_dim,
            out_features=conv_dim,
            kernel_size=(config.d_conv,),
            feature_group_count=conv_dim,
            padding=[(config.d_conv - 1, config.d_conv - 1)],
            rngs=rngs,
        )

        self.dt_bias = nnx.Param(jnp.zeros(config.n_heads), rngs=rngs)

        A_min, A_max = config.A_init_range
        key = rngs.params()
        A = random.uniform(key, (config.n_heads,), minval=A_min, maxval=A_max)
        A_log = jnp.log(A)
        self.A_log = nnx.Param(A_log, rngs=rngs)

        self.D = nnx.Param(jnp.ones(config.n_heads), rngs=rngs)

        self.norm = nnx.RMSNorm(config.d_inner, rngs=rngs)
        self.out_proj = nnx.Linear(
            config.d_inner, config.d_model, use_bias=config.bias, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray, cache: InferenceCache | None = None):
        """Forward pass of the MambaVisionMixer.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, sequence_length, d_model).
            cache: hidden states for inference step. If None, hidden states are
                initialized to zeros.

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        if cache:
            return self.step(x, cache)

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
        xbc_silu = jax.nn.silu(xbc_conv[:, : x.shape[1], :])

        # split the conv state into the conv kernel and the conv state
        x, b, c = jnp.split(xbc_silu, self.config.indices_xBC, axis=-1)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.config.head_dim)

        # apply ssd function
        y, ssm_state = ssd(
            x * jnp.expand_dims(dt, axis=-1),
            A * dt,
            rearrange(b, "b l n -> b l 1 n"),
            rearrange(c, "b l n -> b l 1 n"),
            self.config.chunk_size,
        )

        # Combine the output of the ssd function with the input and rearrange
        y = y + x * jnp.expand_dims(self.D.value, axis=-1)
        y = rearrange(y, "b l h p -> b l (h p)")

        # apply the output projection
        y = self.out_proj(y)

        # hidden_state = InferenceCache(conv_state, ssm_state)
        # return y, hidden_state
        return y

    def step(
        self, x: jnp.ndarray, cache: InferenceCache
    ) -> tuple[jnp.ndarray, InferenceCache]:
        """Forward pass through a single step of the Mamba layer.

        This function implements a single step of the Mamba layer, which consists
        of a projection, a depthwise convolution, and an SSD function. It takes in
        the input tensor, x, and the hidden states, cache, and returns the output
        tensor, y, and the updated hidden states. The hidden states are updated
        using the SSD function. This function is used when the hidden states are
        provided.

        Args:
            x (batch_size, 1, d_model): input tensor
            cache: hidden states for inference step. If None, hidden states are
                initialized to zeros.

        Returns:
            output tensor of shape (batch_size, 1, d_model) and updated hidden
            states.
        """
        assert x.shape[1] == 1, "Only supports single token inputs"

        # Squeeze dimension 1 from x
        x_squeezed = jnp.squeeze(x, axis=1)

        # Project input using in_proj
        zxbcdt = self.in_proj(x_squeezed)  # (batch, d_in_proj)

        # Split zxbcdt into z, xBC, and dt
        sizes = [
            self.config.d_inner,
            self.config.d_inner + 2 * self.config.d_state,
            self.config.n_heads,
        ]
        indices = jnp.cumsum(jnp.array(sizes))[:-1]
        z, xBC, dt = jnp.split(zxbcdt, indices, axis=-1)

        conv_state = cache.conv_state
        ssm_state = cache.ssm_state

        # Advance convolution input
        conv_state = jnp.roll(conv_state, shift=-1, axis=-1)
        conv_state = conv_state.at[:, :, -1].set(xBC)

        # Convolution step
        conv_weight_rearranged = rearrange(self.conv.layer.kernel, "d 1 w -> d w")
        xBC = jnp.sum(conv_state * conv_weight_rearranged, axis=-1)
        xBC += self.conv.layer.bias
        xBC = jax.nn.silu(xBC)

        # Split xBC into x, B, and C
        x, B, C = jnp.split(xBC, self.config.indices_xBC, axis=-1)

        # Exponential of A_log
        A = -jnp.exp(self.A_log.value)  # (nheads,)

        # SSM step
        dt = jax.nn.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = jnp.exp(dt * A)  # (batch, nheads)

        # Rearrange x
        x = rearrange(x, "b (h p) -> b h p", p=self.config.head_dim)

        # Compute dBx
        dBx = jnp.einsum("bh, bn, bhp -> bhpn", dt, B, x)

        # Update ssm_state
        ssm_state = ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx

        # Compute y
        y = jnp.einsum("bhpn, bn -> bhp", ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x

        # Rearrange y
        y = rearrange(y, "b h p -> b (h p)")

        # Apply normalization and output projection
        # y = self.norm(y, z)
        y = self.out_proj(y)

        return y, InferenceCache(conv_state, ssm_state)


class MambaVisionLayer(nnx.Module):
    """Base layer of MambaVision.

    This class implements a layer of the MambaVision architecture, which can be either
    a convolutional block or a transformer block, optionally followed by a downsampling operation.

    Attributes:
        conv (bool): Whether to use convolutional blocks instead of transformer blocks.
        blocks (list): List of Block or ConvBlock instances.
        transformer_block (bool): Whether the layer uses transformer blocks.
        downsample (Downsample | None): Downsampling operation, if applicable.
        do_gt (bool): Flag for global token usage (currently not implemented).
        window_size (int): Size of the window for windowed attention in transformer blocks.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks in the layer.
        num_heads (int): Number of attention heads in transformer blocks.
        window_size (int): Size of the window for windowed attention.
        conv (bool, optional): Whether to use convolutional blocks. Defaults to False.
        downsample (bool, optional): Whether to apply downsampling after the blocks. Defaults to True.
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
        qk_norm (bool, optional): Whether to apply normalization to query and key. Defaults to False.
        ffn_bias (bool, optional): If True, use bias in the feed-forward network. Defaults to True.
        proj_bias (bool, optional): If True, use bias in the projection layers. Defaults to True.
        proj_drop (float, optional): Dropout rate for projection layers. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate for attention. Defaults to 0.0.
        drop_path (float | list, optional): Stochastic depth rate. Defaults to 0.0.
        init_values (float | None, optional): Initial layer scale value. Defaults to None.
        init_values_conv (float | None, optional): Initial layer scale value for conv blocks. Defaults to None.
        attention (Callable, optional): Attention mechanism to use. Defaults to Attention.
        act_layer (Callable, optional): Activation function to use. Defaults to nnx.gelu.
        norm_layer (Callable, optional): Normalization layer to use. Defaults to nnx.LayerNorm.
        ffn_layer (Callable, optional): Feed-forward network layer to use. Defaults to Mlp.
        block_types (list, optional): List of transformer block types. Defaults to [].
        rngs (nnx.Rngs, optional): Random number generators. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        conv: bool = False,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list = 0.0,
        init_values: float | None = None,
        init_values_conv: float | None = None,
        transformer_attention: Callable = Attention,
        mamba_mixer: Callable = MambaVisionMixer,
        act_layer: Callable = nnx.gelu,
        norm_layer: Callable = nnx.LayerNorm,
        ffn_layer: Callable = Mlp,
        block_types: list = [],
        rngs: nnx.Rngs = None,
    ):
        self.conv = conv

        if conv:
            self.blocks = [
                ConvBlock(
                    dim=dim,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    init_values=init_values_conv,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
            self.transformer_block = False
        else:
            self.blocks = [
                Block(
                    dim=dim,
                    block_type=block_type,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    ffn_bias=ffn_bias,
                    proj_bias=proj_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    init_values=init_values,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    attention=(
                        transformer_attention
                        if block_type == transformer_attention.__name__
                        else mamba_mixer
                    ),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    ffn_layer=ffn_layer,
                    rngs=rngs,
                )
                for i, block_type in enumerate(block_types)
            ]
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim, rngs=rngs)
        self.do_gt = False
        self.window_size = window_size

    def __call__(self, x: jnp.ndarray):
        _, H, W, _ = x.shape
        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
                _, Hp, Wp, _ = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x = blk(x)

        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]

        if self.downsample is not None:
            x = self.downsample(x)

        return x
