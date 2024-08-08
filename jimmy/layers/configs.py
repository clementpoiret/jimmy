import math
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ConvBlockConfig:
    kernel_size = (3, 3)
    act_layer: str = "gelu"
    norm_layer: str = "batchnorm"
    drop_path: float = 0.0
    init_values: Optional[float] = None


@dataclass
class ViTBlockConfig:
    mlp_ratio: float = 4.0
    drop_path: float | list = 0.0
    act_layer: str = "gelu"
    norm_layer: str = "layernorm"
    attention: str = "attention"
    ffn_layer: str = "mlp"
    ffn_bias: bool = True
    proj_drop: float = 0.0
    init_values: Optional[float] = None
    msa_window_size: int = -1

    def __post_init__(self):
        allowed_attentions = [
            "attention",
            "linearattention",
            "mambavisionmixer",
            "mamba2mixer",
            "mamba2visionmixer",
        ]
        allowed_ffns = ["mlp", "swiglu"]
        if self.attention not in allowed_attentions:
            raise ValueError(
                f"Unsupported attention. Got `{self.attention}`, expected one of {allowed_attentions}."
            )
        if self.ffn_layer not in allowed_ffns:
            raise ValueError(
                f"Unsupported FFN. Got `{self.ffn_layer}`, expected one of {allowed_ffns}."
            )

        if isinstance(self.drop_path, list):
            if len(self.drop_path) != 2:
                raise AssertionError(
                    f"`drop_path` needs to have 2 elements, got {len(self.drop_path)}."
                )
            dr1, dr2 = self.drop_path
            self.dr1 = float(dr1)
            self.dr2 = float(dr2)
        else:
            self.dr1 = self.dr2 = float(self.drop_path)


@dataclass
class AttentionConfig:
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    proj_bias: bool = True
    qk_norm: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    norm_layer: str = "layernorm"

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
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    n_groups: int = 1
    learnable_init_states: bool = False
    expand: int = 2
    head_dim: int = 64
    chunk_size: int = 64
    A_init_range: Tuple[int, int] = (1, 16)
    linear_attn_duality: bool = True
    use_fast_path: bool = True
    layer_idx: int | None = None
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        """Compute inner dimension and number of heads."""
        # Shared between Mamba-2 and Mamba-1
        self.d_inner = int(self.d_model * self.expand)
        assert self.d_inner % self.head_dim == 0

        if self.n_groups == -1:
            self.n_groups = (
                self.d_inner // self.headdim
            )  # equivalent to multi-head attention

        # Mamba-2
        A_min, A_max = self.A_init_range
        assert 0 < A_min < A_max
        self.n_heads = self.d_inner // self.head_dim
        self.indices_xBC = [self.d_inner, self.d_inner + self.n_groups * self.d_state]
        # self.indices_xBC = [
        #     self.d_inner,
        #     self.n_groups * self.d_state,
        #     self.n_groups * self.d_state,
        # ]

        # Mamba-1
        self.dt_rank = (
            math.ceil(self.d_model / 16) if self.dt_rank == "auto" else self.dt_rank
        )
