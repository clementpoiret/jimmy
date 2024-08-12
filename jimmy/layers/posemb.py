import jax.numpy as jnp
from einops import rearrange
from flax import nnx


class PosEmbMLPSwinv1D(nnx.Module):
    """From FasterViT"""

    rank: int = 2
    seq_len: int = 4
    conv: bool = False

    def __init__(self, dim, *, rngs: nnx.Rngs, **kwargs):
        self.__dict__.update(**kwargs)

        if self.conv:
            self.cpb_mlp = nnx.Sequential(
                nnx.Conv(
                    in_features=self.rank,
                    out_features=512,
                    kernel_size=(1),
                    strides=(1),
                    use_bias=True,
                    rngs=rngs,
                ),
                nnx.relu,
                nnx.Conv(
                    in_features=512,
                    out_features=dim,
                    kernel_size=(1),
                    strides=(1),
                    use_bias=False,
                    rngs=rngs,
                ),
            )
        else:
            self.cpb_mlp = nnx.Sequential(
                nnx.Linear(self.rank, 512, use_bias=True, rngs=rngs),
                nnx.relu,
                nnx.Linear(512, dim, use_bias=False, rngs=rngs),
            )

        self.grid_exists = False
        self.pos_emb = None
        self.deployed = False
        relative_bias = jnp.zeros((1, self.seq_len, dim))

        self.relative_bias = nnx.Variable(relative_bias)

    def deploy(self):
        self.deployed = True

    def __call__(self, x: jnp.ndarray):
        seq_len = x.shape[1] if not self.conv else x.shape[2]

        if self.deployed:
            return x + self.relative_bias
        else:
            self.grid_exists = False

        if not self.grid_exists:
            self.grid_exists = True

            if self.rank == 1:
                relative_coords_h = jnp.arange(0, seq_len, dtype=x.dtype)
                relative_coords_h -= seq_len // 2
                relative_coords_h /= seq_len // 2
                relative_coords_table = relative_coords_h[jnp.newaxis, :, jnp.newaxis]

                self.pos_emb = self.cpb_mlp(relative_coords_table)
                self.relative_bias = self.pos_emb
            else:
                seq_len = int(seq_len**0.5)
                relative_coords_h = jnp.arange(0, seq_len, dtype=x.dtype)
                relative_coords_w = jnp.arange(0, seq_len, dtype=x.dtype)
                relative_coords_table = jnp.stack(
                    jnp.meshgrid(relative_coords_h, relative_coords_w)
                )
                relative_coords_table -= seq_len // 2
                relative_coords_table /= seq_len // 2

                flattened_table = rearrange(relative_coords_table, "c h w -> 1 (h w) c")

                self.pos_emb = self.cpb_mlp(flattened_table)

        return x + self.pos_emb


class PosEmbMLPSwinv2D(nnx.Module):
    """
    From FasterViT, for Windowed Attention
    """

    ct_correct: bool = False
    no_log: bool = False

    def __init__(
        self,
        window_size: tuple,
        pretrained_window_size: int,
        num_heads: int,
        seq_len: int,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.__dict__.update(**kwargs)

        self.cpb_mlp = nnx.Sequential(
            nnx.Linear(2, 512, use_bias=True, rngs=rngs),
            nnx.relu,
            nnx.Linear(512, num_heads, use_bias=False, rngs=rngs),
        )

        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.seq_len = seq_len

        relative_coords_h = jnp.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=jnp.float32
        )
        relative_coords_w = jnp.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=jnp.float32
        )
        relative_coords_table = jnp.stack(
            jnp.meshgrid(relative_coords_h, relative_coords_w)
        )
        relative_coords_table = rearrange(
            relative_coords_table,
            "c h w -> 1 h w c",
        )

        if self.pretrained_window_size[0] > 0:
            relative_coords_table = relative_coords_table.at[:, :, :, 0].set(
                relative_coords_table[:, :, :, 0] / pretrained_window_size[0] - 1
            )
            relative_coords_table = relative_coords_table.at[:, :, :, 1].set(
                relative_coords_table[:, :, :, 1] / pretrained_window_size[1] - 1
            )
        else:
            relative_coords_table = relative_coords_table.at[:, :, :, 0].set(
                relative_coords_table[:, :, :, 0] / window_size[0] - 1
            )
            relative_coords_table = relative_coords_table.at[:, :, :, 1].set(
                relative_coords_table[:, :, :, 1] / window_size[1] - 1
            )

        if not self.no_log:
            relative_coords_table = relative_coords_table * 8
            relative_coords_table = (
                jnp.sign(relative_coords_table)
                * jnp.log2(jnp.abs(relative_coords_table) + 1.0)
                / jnp.log2(8)
            )

        self.relative_coords_table = nnx.Variable(relative_coords_table)

        coords_h = jnp.arange(self.window_size[0])
        coords_w = jnp.arange(self.window_size[1])
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose(1, 2, 0)
        relative_coords = relative_coords + jnp.array(
            [self.window_size[0] - 1, self.window_size[1] - 1]
        )
        relative_coords = relative_coords.at[:, :, 0].set(
            relative_coords[:, :, 0] * (2 * self.window_size[1] - 1)
        )
        relative_position_index = jnp.sum(relative_coords, axis=-1)

        self.relative_position_index = nnx.Variable(relative_position_index)

        self.grid_exists = False
        self.pos_emb = None
        self.deployed = False
        relative_bias = jnp.zeros((1, self.num_heads, self.seq_len, self.seq_len))

        self.relative_bias = nnx.Variable(relative_bias)

    def deploy(self):
        self.deployed = True

    def __call__(self, x: jnp.ndarray, local_window_size: int):
        if self.deployed:
            return x + self.relative_bias
        else:
            self.grid_exists = False

        if not self.grid_exists:
            self.grid_exists = True

            relative_position_bias_table = self.cpb_mlp(
                self.relative_coords_table
            ).reshape(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.value.reshape(-1)
            ].reshape(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))
            relative_position_bias = 16 * nnx.sigmoid(relative_position_bias)

            n_global_feature = x.shape[2] - local_window_size
            if n_global_feature > 0 and self.ct_correct:
                step_for_ct = self.window_size[0] / (n_global_feature**0.5 + 1)
                seq_len = int(n_global_feature**0.5)
                indices = []

                # TODO: REMOVE THIS FOR LOOPS
                for i in range(seq_len):
                    for j in range(seq_len):
                        ind = (i + 1) * step_for_ct * self.window_size[0] + (
                            j + 1
                        ) * step_for_ct
                        indices.append(int(ind))

                top_part = relative_position_bias[:, indices, :]
                lefttop_part = relative_position_bias[:, indices, :][:, :, indices]
                left_part = relative_position_bias[:, :, indices]

            relative_position_bias = jnp.pad(
                relative_position_bias,
                ((0, 0), (n_global_feature, 0), (n_global_feature, 0)),
            )

            if n_global_feature > 0 and self.ct_correct:
                relative_position_bias = relative_position_bias * 0.0
                relative_position_bias = relative_position_bias.at[
                    :, :n_global_feature, :n_global_feature
                ].set(lefttop_part)
                relative_position_bias = relative_position_bias.at[
                    :, :n_global_feature, n_global_feature:
                ].set(top_part)
                relative_position_bias = relative_position_bias.at[
                    :, n_global_feature:, :n_global_feature
                ].set(left_part)

            self.pos_emb = relative_position_bias[jnp.newaxis, ...]
            self.relative_bias = self.pos_emb

        return x + self.pos_emb


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
                for t in jnp.meshgrid(
                    *[jnp.arange(d) for d in channel_dims], indexing="ij"
                )
            ],
            axis=-1,
        )

        # rotations
        rotations_re = jnp.cos(angles)
        rotations_im = jnp.sin(angles)
        self.rotations = nnx.Variable(jnp.stack([rotations_re, rotations_im], axis=-1))

    def __call__(self, x: jnp.ndarray):
        dtype = x.dtype
        x = x.astype(jnp.float32)

        # Reshape x to separate real and imaginary parts
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = x_reshaped[..., 0] + 1j * x_reshaped[..., 1]

        # Apply rotation
        rotations_complex = self.rotations[..., 0] + 1j * self.rotations[..., 1]
        pe_x = rotations_complex * x_complex

        # Convert back to real representation
        pe_x_real = jnp.stack([pe_x.real, pe_x.imag], axis=-1)

        return pe_x_real.reshape(*x.shape).astype(dtype)
