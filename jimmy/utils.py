from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import dtypes, random


def window_partition(x: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """Partition the input tensor into non-overlapping windows.

    Args:
        x: Input tensor of shape (B, H, W, C) in NHWC format.
        window_size: Size of each square window.

    Returns:
        Tensor of shape (num_windows*B, window_size*window_size, C) containing local window features.

    Note:
        This function assumes that both H and W are divisible by window_size.
    """
    return rearrange(
        x, "b (h w1) (w w2) c -> (b h w) (w1 w2) c", w1=window_size, w2=window_size
    )


def window_reverse(
    windows: jnp.ndarray, window_size: int, H: int, W: int
) -> jnp.ndarray:
    """Reverse the window partitioning process.

    Args:
        windows: Tensor of shape (num_windows*B, window_size*window_size, C) containing local window features.
        window_size: Size of each square window.
        H: Height of the original image.
        W: Width of the original image.

    Returns:
        Tensor of shape (B, H, W, C) in NHWC format.

    Note:
        This function assumes that both H and W are divisible by window_size.
    """
    return rearrange(
        windows,
        "(b h w) (w1 w2) c -> b (h w1) (w w2) c",
        h=H // window_size,
        w=W // window_size,
        w1=window_size,
        w2=window_size,
    )


def pad_or_truncate_to_length(x: jnp.ndarray, target_length: int):
    """Pad or truncate the last dimension of a tensor to a target length.

    Args:
        x: input tensor of shape [batch, ..., length]
        target_length: target length of the last dimension

    Returns:
        padded or truncated tensor
    """
    current_length = x.shape[-1]
    if current_length < target_length:
        # Pad
        pad_width = target_length - current_length
        return jnp.pad(x, ((0, 0), (0, 0), (pad_width, 0)))
    elif current_length > target_length:
        # Truncate
        return x[:, :, -target_length:]

    return x


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


def test(
    jax_fn: Callable,
    torch_fn: Callable,
    shape: Tuple[int] = (1, 3, 518, 518),
    jax_transpose: Tuple[int] = (0, 2, 3, 1),
    eps: float = 1e-4,
) -> bool:
    """
    Simple function to check if two similar implementations have the same output
    in Jax and PyTorch through cosine similarity.

    Args:
        jax_fn (Callable): the jax model or function to test.
        torch_fn (Callable): the torch model or function to test.
        shape (Tuple[int]): the shape of the random input array to generate.
        jax_transpose (Tuple[int]): the transpose operation to match jax's fmt.
        eps (float): Similarity tolerance.

    Returns:
        bool: True if outputs are similar.
    """
    try:
        import torch
    except:
        raise ImportError("`torch` not available")

    arr = np.random.normal(1, size=shape)

    jax_arr = jnp.asarray(arr).transpose(*jax_transpose)
    torch_arr = torch.tensor(arr).float()

    a = np.array(jax_fn(jax_arr))
    b = torch_fn(torch_arr).detach().numpy()

    dist = np.sum(a * b) / np.linalg.norm(a) / np.linalg.norm(b)

    return dist > (1 - eps)
