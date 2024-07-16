# Based on the original torch pscan impl. from radarFudan:
# https://github.com/radarFudan/mamba-minimal-jax/blob/b76334404f7f1d87e47ffc1158b1bd151098d1c2/model.py

import jax.numpy as jnp
from einops import einsum
from flax import nnx


def selective_scan(u: jnp.ndarray,
                   delta: jnp.ndarray,
                   A: jnp.ndarray,
                   B: jnp.ndarray,
                   C: jnp.ndarray,
                   D: jnp.ndarray,
                   delta_bias: jnp.ndarray | None = None,
                   delta_softplus: bool = False):
    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
    """
    b, d_in, l = u.shape
    n = A.shape[1]

    if delta_bias is not None:
        delta = delta + jnp.expand_dims(delta_bias, axis=-1)
    if delta_softplus:
        delta = nnx.softplus(delta)

    # Discretize continuous parameters (A, B)
    deltaA = jnp.exp(einsum(delta, A, "b d l, d n -> b l d n"))
    deltaB_u = einsum(delta, B, u, 'b d l, b n l, b d l -> b l d n')

    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    x = jnp.zeros((b, d_in, n))
    ys = []
    for i in range(l):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = einsum(x, C[:, :, i], "b d n, b n -> b d")
        ys.append(y)
    y = jnp.stack(ys, axis=2)  # shape (b, l, d_in)

    y = y + u * jnp.expand_dims(D, axis=-1)

    return y
