from typing import Optional

from flax import nnx

from jimmy.io import load
from jimmy.layers import Mlp, SwiGLU
from jimmy.models import vit


def parse_cfg(config: dict) -> dict:
    """
    Replace module names with their corresponding implementations.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: The updated configuration dictionary with module implementations.
    """
    for key, value in config.items():
        match value:
            case "mlp":
                config[key] = Mlp
            case "swiglu":
                config[key] = SwiGLU
            case _:
                continue
    return config


def load_model(specifications: dict,
               rngs: nnx.Rngs,
               url: Optional[str] = None,
               pretrained: bool = True,
               **kwargs) -> nnx.Module:
    """
    Load a model based on the given specifications.

    Args:
        specifications (dict): A dictionary containing model specifications.
        rngs (nnx.Rngs): Random number generators.
        url (Optional[str], optional): URL to download pretrained weights. Defaults to None.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        nnx.Module: The loaded model.

    Raises:
        NotImplementedError: If the specified model class is not implemented.
    """
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
