import jax.numpy as jnp
from flax import nnx


class Identity(nnx.Module):
    """An identity module that returns the input unchanged."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the identity operation.

        Args:
            x (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: The input array unchanged.
        """
        return x


class Downsample(nnx.Module):
    """Downsampling block for reducing spatial dimensions of feature maps."""

    keep_dim: bool = False

    def __init__(self, dim: int, *, rngs: nnx.Rngs, **kwargs):
        """Initialize the Downsample block.

        Args:
            dim (int): Number of input channels.
            keep_dim (bool): If True, maintain the number of channels in the output.
                             If False, double the number of channels.
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        self.__dict__.update(**kwargs)

        dim_out = dim if self.keep_dim else 2 * dim
        self.reduction = nnx.Conv(
            in_features=dim,
            out_features=dim_out,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        return self.reduction(x)
