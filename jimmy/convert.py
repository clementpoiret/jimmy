import logging
import re
from typing import Set, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.tree_util import DictKey

DEFAULT_TRANSPOSE_WHITELIST = {
    "cls_token",
    "pos_embed",
    "mask_token",
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_params_from_torch_hub(
        jax_model: nnx.Module,
        torch_hub_cfg: Tuple[str],
        transpose_whitelist: Set[str] = DEFAULT_TRANSPOSE_WHITELIST,
        strict: bool = True) -> nnx.State:
    """
    Load weights from a torch hub model into a jax nnx.Module.

    Args:
        jax_model (nnx.Module): A preexisting Jax model corresponding to the checkpoint to download.
        torch_hub_cfg (Tuple[str]): Arguments passed to `torch.hub.load()`.
        transpose_whitelist (Set[str]): Parameters to exclude from format conversion.
        strict (bool): Whether to crash on missing parameters one of the models.

    Returns:
        nnx.State: 
    """
    try:
        import torch
    except:
        raise ImportError("`torch` not available")

    # Load the pytorch model from torch hub
    torch_model = torch.hub.load(*torch_hub_cfg)
    torch_params = {
        path: param for path, param in torch_model.named_parameters()
    }

    # Extract the parameters from the defined Jax model
    _, jax_params, _ = nnx.split(jax_model, nnx.Param, ...)
    jax_params_flat, jax_param_pytree = jax.tree_util.tree_flatten_with_path(
        jax_params)

    torch_params_flat = []
    for path, param in jax_params_flat:
        # Match the parameters' path of pytorch
        param_path = ".".join([p.key for p in path if type(p) is DictKey])
        param_path = re.sub(r"\.scale|.kernel", ".weight", param_path)
        shape = param.shape

        if param_path not in torch_params:
            _msg = f"{param_path} ({shape}) not found in PyTorch model."
            if strict:
                logger.error(_msg)
                raise AttributeError(_msg)

            logger.warning(f"{_msg} Appending `None` to flat param list.")
            torch_params_flat.append(None)
            continue

        logger.info(f"Converting {param_path}...")
        torch_param = torch_params[param_path]

        # To match format differences (eg NCHW vs NHWC) between PyTorch and Jax
        if param_path not in transpose_whitelist:
            if len(shape) == 4:
                torch_param = torch.permute(torch_param, (2, 3, 1, 0))
            else:
                torch_param = torch.permute(torch_param,
                                            tuple(reversed(range(len(shape)))))

        if shape != torch_param.shape:
            _msg = f"`{param_path}`: expected shape ({shape}) does not match its pytorch implementation ({torch_param.shape})."
            logger.error(_msg)
            raise ValueError(_msg)

        torch_params_flat.append(jnp.asarray(torch_param.detach().numpy()))
        _ = torch_params.pop(param_path)

    loaded_params = jax.tree_util.tree_unflatten(jax_param_pytree,
                                                 torch_params_flat)

    for path, param in torch_params.items():
        logger.warning(
            f"PyTorch parameters `{path}` ({param.shape}) were not converted.")
        if strict and path not in transpose_whitelist:
            _msg = f"The PyTorch model contains parameters ({path}) that do not have a Jax counterpart."
            logger.error(_msg)
            raise AttributeError(_msg)

    return loaded_params
