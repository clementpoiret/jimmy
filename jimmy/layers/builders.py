from typing import Callable

from flax import nnx

from .attention import Attention, LinearAttention
from .mamba import Mamba2Mixer, Mamba2VisionMixer, MambaVisionMixer
from .mlp import Mlp
from .norm import RMSNormGated
from .swiglu import SwiGLU


def get_act(act: str) -> Callable:
    match act:
        case "gelu":
            return nnx.gelu
        case "silu":
            return nnx.silu
        case "relu":
            return nnx.relu
        case _:
            raise NotImplementedError(f"Unknown activation. Got: `{act}`.")


def get_norm(norm: str) -> nnx.Module:
    match norm:
        case "batchnorm":
            return nnx.BatchNorm
        case "layernorm":
            return nnx.LayerNorm
        case "rmsnormgated":
            return RMSNormGated
        case _:
            raise NotImplementedError(f"Unknown norm. Got: `{norm}`.")


def get_module(module: str) -> nnx.Module:
    match module:
        case "attention":
            return Attention
        case "linearattention":
            return LinearAttention
        case "mamba2mixer":
            return Mamba2Mixer
        case "mamba2visionmixer":
            return Mamba2VisionMixer
        case "mambavisionmixer":
            return MambaVisionMixer
        case "mlp":
            return Mlp
        case "swiglu":
            return SwiGLU
        case _:
            raise NotImplementedError(f"Unknown module. Got: `{module}`.")
