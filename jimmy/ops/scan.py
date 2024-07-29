# Based on the original torch pscan impl. from radarFudan:
# https://github.com/radarFudan/mamba-minimal-jax/blob/b76334404f7f1d87e47ffc1158b1bd151098d1c2/model.py

import jax.numpy as jnp
from einops import einsum, rearrange
from flax import nnx
from jax import lax


def selective_scan(
    u: jnp.ndarray,
    delta: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    D: jnp.ndarray,
    delta_bias: jnp.ndarray | None = None,
    delta_softplus: bool = False,
):
    """
    Performs the selective scan algorithm as described in the Mamba paper.

    This function implements the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    where B and C (and the step size delta, which is used for discretization)
    are dependent on the input x(t).

    Args:
        u (jnp.ndarray): Input tensor of shape (b, d, l).
        delta (jnp.ndarray): Step size tensor of shape (b, d, l).
        A (jnp.ndarray): State transition matrix of shape (d, n).
        B (jnp.ndarray): Input matrix of shape (b, n, l).
        C (jnp.ndarray): Output matrix of shape (b, n, l).
        D (jnp.ndarray): Direct feedthrough matrix of shape (d,).
        delta_bias (jnp.ndarray | None, optional): Bias for delta. Defaults to None.
        delta_softplus (bool, optional): Whether to apply softplus to delta. Defaults to False.

    Returns:
        jnp.ndarray: Output tensor of shape (b, d, l).

    References:
        [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
        [2] The Annotated S4: run_SSM(A, B, C, u)

    Notes:
        - b: batch size
        - l: sequence length
        - d: hidden dimension
        - n: latent space dimension

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
    """
    b, d_in, l = u.shape

    if delta_bias is not None:
        delta = delta + jnp.expand_dims(delta_bias, axis=-1)
    if delta_softplus:
        delta = nnx.softplus(delta)

    # Discretize continuous parameters (A, B)
    deltaA = jnp.exp(einsum(delta, A, "b d l, d n -> b l d n"))
    deltaB_u = einsum(delta, B, u, "b d l, b n l, b d l -> b l d n")

    # Define the scan function
    def scan_fn(carry, x):
        x_prev, _ = carry
        x_next = deltaA[:, x] * x_prev + deltaB_u[:, x]
        y = einsum(x_next, C[:, :, x], "b d n, b n -> b d")
        return (x_next, None), y

    x_init = jnp.zeros((b, d_in, A.shape[1]))
    carry_init = (x_init, None)
    _, ys = lax.scan(scan_fn, carry_init, jnp.arange(l))
    ys = rearrange(ys, "l b d -> b d l")

    y = ys + u * jnp.expand_dims(D, axis=-1)

    return y
