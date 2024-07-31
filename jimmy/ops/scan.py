# Based on the original torch pscan impl. from radarFudan:
# https://github.com/radarFudan/mamba-minimal-jax/blob/b76334404f7f1d87e47ffc1158b1bd151098d1c2/model.py
# Mamba2-related code from https://github.com/walln/scratch/blob/ab0b6b891830375b7aa64c8e46e77783b843f5ca/src/scratch/language_modeling/mamba/mamba.py#L462

import jax.numpy as jnp
from einops import einsum, rearrange, repeat
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
    """Performs the selective scan algorithm as described in the Mamba paper.

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


def segsum(x: jnp.ndarray):
    """Stable segment sum calculation.

    Produces a 1-semiseperable matrix which is equivalent to a scalar SSM.

    Args:
        x (batch_size, seq_len, n_heads): input tensor

    Returns:
        output tensor of shape (batch_size, seq_len, n_heads)
    """
    T = x.shape[-1]
    x = repeat(x, "... d -> ... d e", e=T)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool), -1)
    x = jnp.where(mask, x, 0)
    x_segsum = jnp.cumsum(x, axis=-2)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool), 0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def ssd(
    x: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    chunk_size: int,
    initial_states: jnp.ndarray | None = None,
):
    """Structured State Space Duality (SSD).

    This function implements the SSD algorithm for computing the SSM states. It
    takes in the input tensor, A, B, and C, and returns the output tensor, y, and
    the updated SSM states. The SSD algorithm is a generalization of the SSM
    algorithm to the case where the SSM states are not scalars. It is a
    structured matrix multiplication that is equivalent to a scalar SSM.

    Args:
        u (jnp.ndarray): Input tensor of shape (b, l, n, d_head).
        A (jnp.ndarray): State transition matrix of shape (d, l, n).
        B (jnp.ndarray): Input matrix of shape (b, l, n, d_state).
        C (jnp.ndarray): Output matrix of shape (b, l, n, d_state).
        chunk_size: matrix partition size.
        initial_states: (b, 1, n, d_state)

    Returns:
        y (jnp.ndarray): Output tensor of shape (b, l, n, d_head).
        state (jnp.ndarray): Output tensor of shape (b, l, n, d_state).

    Notes:
        - b: batch size
        - l: sequence length
        - d: hidden dimension
        - n: number of heads

    Implementation taken from:
        https://github.com/walln/scratch/blob/ab0b6b891830375b7aa64c8e46e77783b843f5ca/src/scratch/language_modeling/mamba/mamba.py#L537
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    x, A, B, C = (rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size)
                  for m in (x, A, B, C))

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = jnp.cumsum(A, axis=-1)

    # Compute intra-chunk state (diagonal blocks)
    L = jnp.exp(segsum(A))
    Y_diag = jnp.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # Compute intra-chunk state - the right term of low rank factorization of the
    # off diagonal blocks; B terms
    decay_states = jnp.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = jnp.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # Compute the inter-chunk SSM recurrence. Producing the correct SSM states at chunk
    # boundaries. This is the middle term of off diagonal blocks; A terms.
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1])

    states = jnp.concat([initial_states, states], axis=1)
    decay_chunk = jnp.exp(
        segsum(jnp.pad(A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0)))))
    new_states = jnp.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # Compute state and output conversion per chunk
    # the left term of low rank factorization of the off diagonal blocks; C terms
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum("bclhn, bchpn, bhcl -> bclhp", C, states,
                       state_decay_out)

    # Add the output of intra-chunk and inter-chunk states
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state
