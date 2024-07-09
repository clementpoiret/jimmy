from typing import Optional

from flax import nnx

from jimmy.io import load
from jimmy.layers.mlp import Mlp
from jimmy.layers.swiglu import SwiGLU
from jimmy.models import vit

DINOV2_VITS14 = {
    "name": "dinov2_vits14",
    "class": "dinov2",
    "config": {
        "num_heads": 6,
        "embed_dim": 384,
        "mlp_ratio": 4,
        "patch_size": 14,
        "init_values": 1.0,
        "depth": 12,
        "img_size": 518,
        "reg_tokens": 0,
        "qkv_bias": True,
    }
}
DINOV2_VITS14_REG = {
    "name": "dinov2_vits14_reg",
    "class": "dinov2",
    "config": {
        "num_heads": 6,
        "embed_dim": 384,
        "mlp_ratio": 4,
        "patch_size": 14,
        "init_values": 1.0,
        "depth": 12,
        "img_size": 518,
        "reg_tokens": 4,
        "qkv_bias": True,
    }
}
DINOV2_VITB14 = {
    "name": "dinov2_vitb14",
    "class": "dinov2",
    "config": {
        "num_heads": 12,
        "embed_dim": 768,
        "mlp_ratio": 4,
        "patch_size": 14,
        "init_values": 1.0,
        "depth": 12,
        "img_size": 518,
        "reg_tokens": 0,
        "qkv_bias": True,
    }
}
DINOV2_VITB14_REG = {
    "name": "dinov2_vitb14_reg",
    "class": "dinov2",
    "config": {
        "num_heads": 12,
        "embed_dim": 768,
        "mlp_ratio": 4,
        "patch_size": 14,
        "init_values": 1.0,
        "depth": 12,
        "img_size": 518,
        "reg_tokens": 4,
        "qkv_bias": True,
    }
}
DINOV2_VITL14 = {
    "name": "dinov2_vitl14",
    "class": "dinov2",
    "config": {
        "num_heads": 16,
        "embed_dim": 1024,
        "mlp_ratio": 4,
        "patch_size": 14,
        "init_values": 1.0,
        "depth": 24,
        "img_size": 518,
        "reg_tokens": 0,
        "qkv_bias": True,
    }
}
DINOV2_VITL14_REG = {
    "name": "dinov2_vitl14_reg",
    "class": "dinov2",
    "config": {
        "num_heads": 16,
        "embed_dim": 1024,
        "mlp_ratio": 4,
        "patch_size": 14,
        "init_values": 1.0,
        "depth": 24,
        "img_size": 518,
        "reg_tokens": 4,
        "qkv_bias": True,
    }
}
DINOV2_VITG14 = {
    "name": "dinov2_vitg14",
    "class": "dinov2",
    "config": {
        "num_heads": 24,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "patch_size": 14,
        "init_values": 1.0,
        "depth": 40,
        "img_size": 518,
        "reg_tokens": 0,
        "qkv_bias": True,
        "ffn_layer": "swiglu",
    }
}
DINOV2_VITG14_REG = {
    "name": "dinov2_vitg14_reg",
    "class": "dinov2",
    "config": {
        "num_heads": 24,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "patch_size": 14,
        "init_values": 1.0,
        "depth": 40,
        "img_size": 518,
        "reg_tokens": 4,
        "qkv_bias": True,
        "ffn_layer": "swiglu",
    }
}


def parse_cfg(config: dict) -> dict:
    """Replace module names by the real modules"""
    for key, value in config.items():
        match value:
            case "mlp":
                config[key] = Mlp
            case "swiglu":
                config[key] = SwiGLU
            case _:
                continue
    return config


# TODO: pretrained arg
def load_model(specifications: dict,
               rngs: nnx.Rngs,
               url: Optional[str] = None,
               pretrained: bool = True,
               **kwargs) -> nnx.Module:
    cls = specifications["class"]
    config = specifications["config"] | kwargs
    config = parse_cfg(config)

    match cls:
        case "dinov2":
            model = vit.DinoV2(**config, rngs=rngs)
        case _:
            raise NotImplementedError(f"{cls} not implemented.")

    if pretrained:
        _, params, _ = nnx.split(model, nnx.Param, ...)

        raw = load(
            name=specifications["name"],
            url=url,
            params=params,
            specifications=specifications,
        )
        nnx.update(model, raw["params"])

    return model
