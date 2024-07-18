from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nnx.module import first_from

from .attention import Attention
from .mlp import Mlp


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


class LayerScale(nnx.Module):
    """Layer scale module for scaling the output of a layer."""

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        """Initialize the LayerScale module.

        Args:
            dim (int): The dimension of the input.
            init_values (float, optional): Initial value for scaling. Defaults to 1e-5.
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        self.gamma = nnx.Param(
            init_values * nnx.initializers.ones(rngs.params(), [dim]), )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply layer scaling to the input.

        Args:
            x (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Scaled output.
        """
        return x * self.gamma


class DropPath(nnx.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self,
                 drop_prob: float = 0.,
                 scale_by_keep: bool = True,
                 deterministic: bool = False,
                 rng_collection: str = "dropout",
                 rngs: nnx.Rngs = None):
        """Initialize the DropPath module.

        Args:
            drop_prob (float, optional): Probability of dropping a path. Defaults to 0.
            scale_by_keep (bool, optional): Whether to scale the kept values. Defaults to True.
            deterministic (bool, optional): Whether to use deterministic behavior. Defaults to False.
            rng_collection (str, optional): Name of the RNG collection. Defaults to "dropout".
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.deterministic = deterministic
        self.rng_collection = rng_collection
        self.rngs = rngs

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> jnp.ndarray:
        """Apply DropPath to the input.

        Args:
            x (jnp.ndarray): Input array.
            deterministic (bool | None, optional): Override for deterministic behavior. Defaults to None.
            rngs (nnx.Rngs | None, optional): Override for random number generator state. Defaults to None.

        Returns:
            jnp.ndarray: Output after applying DropPath.
        """
        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to Dropout as
            either a __call__ argument or class attribute""",
        )

        if (self.drop_prob == 0.0) or deterministic:
            return x

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.drop_prob == 1.0:
            return jnp.zeros_like(x)

        rngs = first_from(
            rngs,
            self.rngs,
            error_msg=
            """`deterministic` is False, but no `rngs` argument was provided to
            Dropout as either a __call__ argument or class attribute.""",
        )
        rng = rngs[self.rng_collection]()

        keep_prob = 1.0 - self.drop_prob

        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = jax.random.bernoulli(rng, p=keep_prob, shape=shape)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor /= keep_prob

        return x * random_tensor
        # return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class Block(nnx.Module):
    """Generic block for Vision Transformers and MambaVision."""

    def __init__(
        self,
        dim: int,
        block_type: str = "attention",
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float | list = 0.,
        attention: nnx.Module = Attention,
        act_layer: Callable = nnx.gelu,
        norm_layer: nnx.Module = nnx.LayerNorm,
        ffn_layer: nnx.Module = Mlp,
        rngs: nnx.Rngs = None,
    ):
        """Initialize the Block.

        Args:
            dim (int): Input dimension.
            block_type (str, optional): Type of block to use. Defaults to "attention".
            num_heads (int): Number of attention heads. Defaults to 12.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_norm (bool, optional): If True, normalize the query and key. Defaults to False.
            ffn_bias (bool, optional): If True, use bias in the feed-forward network. Defaults to True.
            proj_bias (bool, optional): If True, use bias in projections. Defaults to True.
            proj_drop (float, optional): Dropout rate of the projection. Defaults to 0.
            attn_drop (float, optional): Dropout rate of the attention. Defaults to 0.
            init_values (Optional[float], optional): Initial value for LayerScale. Defaults to None.
            drop_path (float | list, optional): Stochastic depth rate. Defaults to 0.
            attention (nnx.Module, optional): Attention module. Defaults to Attention.
            act_layer (Callable, optional): Activation function. Defaults to nnx.gelu.
            norm_layer (nnx.Module, optional): Normalization layer. Defaults to nnx.LayerNorm.
            ffn_layer (nnx.Module, optional): Feed-forward network module. Defaults to Mlp.
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        self.norm1 = norm_layer(num_features=dim, rngs=rngs)

        match block_type:
            case "attention":
                self.attn = attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    rngs=rngs,
                )
            case "mambavisionmixer":
                self.attn = attention(
                    d_model=dim,
                    d_state=8,
                    d_conv=3,
                    expand=1,
                    rngs=rngs,
                )
            case _:
                raise NotImplementedError(
                    f"block_type `{block_type}` undefined. Should be one of [`attention`, `mambavisionmixer`]"
                )

        if isinstance(drop_path, list):
            if len(drop_path) != 2:
                raise AssertionError(
                    f"`drop_path` needs to have 2 elements, got {len(drop_path)}."
                )
            dr1, dr2 = drop_path
        else:
            dr1 = dr2 = drop_path

        self.ls1 = LayerScale(dim, init_values,
                              rngs=rngs) if init_values else Identity()
        self.drop_path1 = DropPath(dr1, rngs=rngs) if dr1 > 0. else Identity()

        self.norm2 = norm_layer(num_features=dim, rngs=rngs)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            dropout_rate=proj_drop,
            bias=ffn_bias,
            rngs=rngs,
        )
        self.ls2 = LayerScale(dim, init_values,
                              rngs=rngs) if init_values else Identity()
        self.drop_path2 = DropPath(dr2, rngs=rngs) if dr2 > 0. else Identity()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the block to the input.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output after applying the block.
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class ConvBlock(nnx.Module):
    """Convolutional block with normalization, activation, and residual connection."""

    def __init__(
        self,
        dim: int,
        kernel_size=(3, 3),
        act_layer: Callable = nnx.gelu,
        norm_layer: Callable = nnx.BatchNorm,
        norm_params: dict = {"epsilon": 1e-5},
        drop_path: float = 0.,
        init_values: Optional[float] = None,
        rngs: nnx.Rngs = None,
    ):
        """Initialize the ConvBlock.

        Args:
            dim (int): Number of input and output channels.
            kernel_size (tuple, optional): Size of the convolutional kernel. Defaults to (3, 3).
            act_layer (Callable, optional): Activation function. Defaults to nnx.gelu.
            norm_layer (Callable, optional): Normalization layer. Defaults to nnx.BatchNorm.
            norm_params (dict, optional): Parameters for normalization layer. Defaults to {"epsilon": 1e-5}.
            drop_path (float, optional): Drop path rate. Defaults to 0.
            init_values (Optional[float], optional): Initial value for LayerScale. Defaults to None.
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        self.conv1 = nnx.Conv(in_features=dim,
                              out_features=dim,
                              kernel_size=kernel_size,
                              strides=1,
                              padding="SAME",
                              use_bias=True,
                              rngs=rngs)
        self.norm1 = norm_layer(num_features=dim, rngs=rngs, **norm_params)
        self.act = act_layer
        self.conv2 = nnx.Conv(in_features=dim,
                              out_features=dim,
                              kernel_size=kernel_size,
                              strides=1,
                              padding="SAME",
                              use_bias=True,
                              rngs=rngs)
        self.norm2 = norm_layer(num_features=dim, rngs=rngs, **norm_params)
        self.ls1 = LayerScale(dim, init_values,
                              rngs=rngs) if init_values else Identity()
        self.drop_path1 = DropPath(
            drop_path, rngs=rngs) if drop_path > 0. else Identity()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the ConvBlock to the input.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output after applying the ConvBlock.
        """
        x2 = self.act(self.norm1(self.conv1(x)))
        x2 = self.ls1(self.norm2(self.conv2(x2)))
        x = x + self.drop_path1(x2)

        return x
