import math
from dataclasses import dataclass

from flax import nnx


@dataclass
class TransformerConfig:
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    proj_bias: bool = True
    qk_norm: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    norm_layer: nnx.Module = nnx.LayerNorm

    def __post_init__(self):
        self.head_dim = self.dim // self.num_heads


@dataclass
class MambaConfig:
    """Mamba configuration.

    Args:
        d_model: model dimension (D)
        n_layers: number of mamba layers in the model
        d_state: state dimension (N)
        d_conv: convolution kernel size
        expand: expansion factor (E)
        head_dim: head dimension (P)
        chunk_size: matrix partition size (Q)
        vocab_size: vocabulary size
        pad_vocab_size_multiplier: padding
        d_inner: inner dimension
        n_heads: number of heads
    """

    d_model: int
    d_state: int = 128
    d_conv: int = 4
    dt_rank: str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    expand: int = 2
    head_dim: int = 64
    chunk_size: int = 64
    use_fast_path: bool = True
    layer_idx: int | None = (None,)
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        """Compute inner dimension and number of heads."""
        # Shared between Mamba-2 and Mamba-1
        self.d_inner = int(self.d_model * self.expand)
        assert self.d_inner % self.head_dim == 0

        # Mamba-2
        self.n_heads = self.d_inner // self.head_dim
        self.indices_xBC = [self.d_inner, self.d_inner + self.d_state]

        # Mamba-1
        self.dt_rank = (
            math.ceil(self.d_model / 16) if self.dt_rank == "auto" else self.dt_rank
        )
