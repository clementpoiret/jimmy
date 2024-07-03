from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np


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

    dist = (np.sum(a * b) / np.linalg.norm(a) / np.linalg.norm(b))

    return dist > (1 - eps)
