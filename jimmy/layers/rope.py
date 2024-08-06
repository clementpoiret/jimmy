import jax
import jax.numpy as jnp
from flax import nnx


class RoPE(nnx.Module):
    """Rotate tokens based on their position in the sequence.
    Rotation is applied to every dimensions in an non-interleaved manner.

    Paper: Roformer - https://arxiv.org/abs/2104.09864
    """

    def __init__(self, shape: tuple, base: int = 10000):
        super().__init__()
        self.shape = shape
        self.base = base

        channel_dims, feature_dim = self.shape[:-1], self.shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        if feature_dim % k_max != 0:
            raise ValueError("`feature_dim` is not divisible by `k_max`.")

        # angles
        theta_ks = jnp.power(self.base, -jnp.arange(k_max) / k_max)
        angles = jnp.concatenate(
            [
                t[..., None] * theta_ks
                for t in jnp.meshgrid(*[jnp.arange(d) for d in channel_dims],
                                      indexing="ij")
            ],
            axis=-1,
        )

        # rotations
        rotations_re = jnp.cos(angles)
        rotations_im = jnp.sin(angles)
        self.rotations = nnx.Param(
            jnp.stack([rotations_re, rotations_im], axis=-1))

    def __call__(self, x: jnp.ndarray):
        dtype = x.dtype
        x = x.astype(jnp.float32)

        # Reshape x to separate real and imaginary parts
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = x_reshaped[..., 0] + 1j * x_reshaped[..., 1]

        # Apply rotation
        rotations_complex = self.rotations[...,
                                           0] + 1j * self.rotations[..., 1]
        pe_x = rotations_complex * x_complex

        # Convert back to real representation
        pe_x_real = jnp.stack([pe_x.real, pe_x.imag], axis=-1)

        return pe_x_real.reshape(*x.shape).astype(dtype)
