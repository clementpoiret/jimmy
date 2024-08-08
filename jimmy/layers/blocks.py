import jax.numpy as jnp
from einops import rearrange
from flax import nnx

from jimmy.utils import get_defaults, window_partition, window_reverse

from .builders import get_act, get_module, get_norm
from .configs import AttentionConfig, ConvBlockConfig, MambaConfig, ViTBlockConfig
from .dropout import DropPath
from .misc import Identity
from .norm import LayerScale


class ViTBlock(nnx.Module):
    """Generic block for Vision Transformers and MambaVision."""

    attention_config = get_defaults(AttentionConfig)
    mamba_config = get_defaults(MambaConfig)

    def __init__(
        self,
        dim: int,
        config: ViTBlockConfig,
        *,
        rngs: nnx.Rngs,
        attention_kwargs: dict = {},
        mamba_kwargs: dict = {},
        **kwargs,
    ):
        """Initialize the Block.

        Args:
            dim (int): Input dimension.
            block_type (str, optional): Type of block to use. Defaults to "Attention".
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
        self.__dict__.update(**kwargs)

        self.config = config
        self.attention_config.update(**attention_kwargs)
        self.mamba_config.update(**mamba_kwargs)

        norm_layer = get_norm(config.norm_layer)
        self.norm1 = norm_layer(num_features=dim, rngs=rngs)

        is_mamba = "mamba" in config.attention
        self.use_windowed_attention = not is_mamba and config.msa_window_size != -1

        attention = get_module(config.attention)
        attn_cfg = (
            MambaConfig(d_model=dim, **self.mamba_config)
            if is_mamba
            else AttentionConfig(dim, **self.attention_config)
        )
        self.attn = attention(config=attn_cfg, rngs=rngs)

        self.ls1 = (
            LayerScale(dim, init_values=config.init_values, rngs=rngs)
            if config.init_values
            else Identity()
        )
        self.drop_path1 = (
            DropPath(config.dr1, rngs=rngs) if config.dr1 > 0.0 else Identity()
        )

        self.norm2 = norm_layer(num_features=dim, rngs=rngs)
        self.mlp = get_module(config.ffn_layer)(
            in_features=dim,
            hidden_features=int(dim * config.mlp_ratio),
            act_layer=get_act(config.act_layer),
            dropout_rate=config.proj_drop,
            bias=config.ffn_bias,
            rngs=rngs,
        )
        self.ls2 = (
            LayerScale(dim, init_values=config.init_values, rngs=rngs)
            if config.init_values
            else Identity()
        )
        self.drop_path2 = (
            DropPath(config.dr2, rngs=rngs) if config.dr2 > 0.0 else Identity()
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the block to the input.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output after applying the block.
        """
        if self.use_windowed_attention:
            # Apply windowed attention on Multi-Head Self Attention
            _, L, _ = x.shape
            H = W = int(L**0.5)
            x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)
            ws = self.config.msa_window_size

            pad_b = (ws - H % ws) % ws
            pad_r = (ws - W % ws) % ws
            if pad_r > 0 or pad_b > 0:
                x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
                _, Hp, Wp, _ = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, ws)

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if self.use_windowed_attention:
            x = window_reverse(x, ws, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]
            x = rearrange(x, "b h w c -> b (h w) c")

        return x


class ConvBlock(nnx.Module):
    """Convolutional block with normalization, activation, and residual connection."""

    def __init__(
        self,
        dim: int,
        config: ConvBlockConfig,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        """Initialize the ConvBlock.

        Args:
            dim (int): Number of input and output channels.
            kernel_size (tuple, optional): Size of the convolutional kernel. Defaults to (3, 3).
            act_layer (Callable, optional): Activation function. Defaults to nnx.gelu.
            norm_layer (Callable, optional): Normalization layer. Defaults to nnx.BatchNorm.
            drop_path (float, optional): Drop path rate. Defaults to 0.
            init_values (Optional[float], optional): Initial value for LayerScale. Defaults to None.
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        self.__dict__.update(**kwargs)

        norm_layer = get_norm(config.norm_layer)
        act_layer = get_act(config.act_layer)

        self.conv1 = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=config.kernel_size,
            strides=1,
            padding="SAME",
            use_bias=True,
            rngs=rngs,
        )
        self.norm1 = norm_layer(num_features=dim, rngs=rngs)
        self.act = act_layer
        self.conv2 = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=config.kernel_size,
            strides=1,
            padding="SAME",
            use_bias=True,
            rngs=rngs,
        )
        self.norm2 = norm_layer(num_features=dim, rngs=rngs)
        self.ls1 = (
            LayerScale(dim, init_values=config.init_values, rngs=rngs)
            if config.init_values
            else Identity()
        )
        self.drop_path1 = (
            DropPath(float(config.drop_path), rngs=rngs)
            if config.drop_path > 0.0
            else Identity()
        )

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


class MllaBlock(nnx.Module):

    attention_config = get_defaults(AttentionConfig)
    attention_config["num_heads"] = 12  # to match the original impl
    mamba_config = get_defaults(MambaConfig)
    use_dwc: bool = True  # For Mlla but not for VMamba-2

    def __init__(
        self,
        dim: int,
        config: ViTBlockConfig,
        *,
        rngs: nnx.Rngs,
        attention_kwargs: dict = {},
        mamba_kwargs: dict = {},
        **kwargs,
    ):
        """Initialize the Block.

        Args:
            dim (int): Input dimension.
            block_type (str, optional): Type of block to use. Defaults to "Attention".
            num_heads (int): Number of attention heads. Defaults to 12.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_norm (bool, optional): If True, normalize the query and key. Defaults to False.
            ffn_bias (bool, optional): If True, use bias in the feed-forward network. Defaults to True.
            proj_bias (bool, optional): If True, use bias in projections. Defaults to True.
            proj_drop (float, optional): Dropout rate of the projection. Defaults to 0.
            attn_drop (float, optional): Dropout rate of the attention. Defaults to 0.
            drop_path (float): Stochastic depth rate. Defaults to 0.
            attention (nnx.Module, optional): Attention module. Defaults to Attention.
            act_layer (Callable, optional): Activation function. Defaults to nnx.gelu.
            norm_layer (nnx.Module, optional): Normalization layer. Defaults to nnx.LayerNorm.
            ffn_layer (nnx.Module, optional): Feed-forward network module. Defaults to Mlp.
            init_values (Optional[float], optional): Initial value for LayerScale. Defaults to None.
            rngs (nnx.Rngs, optional): Random number generator state. Defaults to None.
        """
        self.__dict__.update(**kwargs)

        self.config = config
        self.attention_config.update(**attention_kwargs)
        self.mamba_config.update(**mamba_kwargs)

        norm_layer = get_norm(config.norm_layer)
        self.act = get_act(config.act_layer)

        self.cpe1 = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=3,
            strides=1,
            padding=1,
            feature_group_count=dim,
            use_bias=True,
            rngs=rngs,
        )
        self.norm1 = norm_layer(num_features=dim, rngs=rngs)

        if self.use_dwc:
            self.in_proj = nnx.Linear(dim, dim, rngs=rngs)
            self.act_proj = nnx.Linear(dim, dim, rngs=rngs)
            self.dwc = nnx.Conv(
                in_features=dim,
                out_features=dim,
                kernel_size=3,
                strides=1,
                padding=1,
                feature_group_count=dim,
                use_bias=True,
                rngs=rngs,
            )
            self.out_proj = nnx.Linear(dim, dim, rngs=rngs)

        is_mamba = "mamba" in config.attention

        attention = get_module(config.attention)
        attn_cfg = (
            MambaConfig(d_model=dim, **self.mamba_config)
            if is_mamba
            else AttentionConfig(dim, **self.attention_config)
        )
        self.attn = attention(config=attn_cfg, rngs=rngs)

        self.drop_path1 = (
            DropPath(config.dr1, rngs=rngs) if config.dr1 > 0.0 else Identity()
        )

        self.cpe2 = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=3,
            strides=1,
            padding=1,
            feature_group_count=dim,
            use_bias=True,
            rngs=rngs,
        )

        self.norm2 = norm_layer(num_features=dim, rngs=rngs)
        self.mlp = get_module(config.ffn_layer)(
            in_features=dim,
            hidden_features=int(dim * config.mlp_ratio),
            act_layer=self.act,
            dropout_rate=config.proj_drop,
            bias=config.ffn_bias,
            rngs=rngs,
        )

        self.drop_path2 = (
            DropPath(config.dr2, rngs=rngs) if config.dr2 > 0.0 else Identity()
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the block to the input.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output after applying the block.
        """
        _, l, _ = x.shape
        # Let's assume a squared initial shape
        h = w = int(l**0.5)

        x1 = x + rearrange(
            self.cpe1(rearrange(x, "b (h w) c -> b h w c", h=h, w=w)),
            "b h w c -> b (h w) c",
        )
        x1 = self.norm1(x1)

        if self.use_dwc:
            act_res = self.act(self.act_proj(x1))

            x1 = rearrange(self.in_proj(x1), "b (h w) c -> b h w c", h=h, w=w)
            x1 = self.act(rearrange(self.dwc(x1), "b h w c -> b (h w) c"))

        x1 = self.attn(x1)

        if self.use_dwc:
            x1 = self.out_proj(x * act_res)

        x = x + self.drop_path1(x1)

        x += rearrange(
            self.cpe2(rearrange(x, "b (h w) c -> b h w c", h=h, w=w)),
            "b h w c -> b (h w) c",
        )

        x += self.drop_path2(self.mlp(self.norm2(x)))

        return x
