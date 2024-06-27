import jax.numpy as jnp
from flax import nnx


class Attention(nnx.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nnx.Module = nnx.LayerNorm,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.proj_drop = nn.Dropout(proj_drop, rngs=Rngs)

        if qk_norm:
            self.q_norm = norm_layer(self.head_dim, rngs=rngs)
            self.k_norm = norm_layer(self.head_dim, rngs=rngs)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(self, x: jnp.ndarray):
        B, N, C = x.shape
        assert (
            C == self.dim
        ), f"Input embedding dimension ({C}) should match layer embedding dimension ({self.dim})."

        qkv = self.qkv(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)
        if qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # TODO: fused attention
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        attn = nnx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
