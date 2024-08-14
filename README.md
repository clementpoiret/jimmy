# Jimmy Vision

Jimmy Vision is a Jax-based library that provides implement computer vision models.
It's designed to be flexible, efficient, and easy to use for researchers and practitioners
in the field of deep learning.

> [!WARNING] Jimmy is not yet ready for production use.
> It is a work in progress, intended for experimentations.

## Features

- Implementation of DinoV2 (Vision Transformer) models
- Implementation of MambaVision models
- Support for loading pre-trained weights from PyTorch models (DinoV2 only)
- Flexible model configuration and customization
- Efficient Jax-based computations

## Installation

### Install using PyPI

```sh
# cpu
pip install jimmy-vision

# cuda12
pip install jimmy-vision[cuda12]
```

### Cloning the repo

You can either use [poetry](https://python-poetry.org/),
[nix](https://nixos.org/download/), or [devenv](devenv.sh):

```sh
git clone git@github.com:clementpoiret/jimmy.git
cd jimmy

# either
nix develop --impure

# or
poetry install -E cuda12
```

## Quick Start

Here's a quick example of how to use Jimmy to load a pre-trained DinoV2 model:

```python
import jax
import jax.numpy as jnp
from flax import nnx

from jimmy.models import DINOV2_VITS14, load_model

# Initialize random number generator
rngs = nnx.Rngs(42)

# Load the model
model = load_model(
    DINOV2_VITS14,
    rngs=rngs,
    pretrained=True,
    url=
    "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vits14.jim",
)

# Create a random input
key = rngs.params()
x = jax.random.normal(key, (1, 518, 518, 3))

# Run inference
output = model(x)
print(output.shape)  # 1, 1370, 384
```

## Models

Jimmy currently supports the following models:

- DinoV2 (various sizes: ViT-S/14, ViT-B/14, ViT-L/14, ViT-G/14)
- MambaVision (coming soon)

To load a specific model, you can use the `load_model` function with the appropriate
configuration:

```python
from jimmy.models import load_model, DINOV2_VITB14

model = load_model(DINOV2_VITB14, rngs=rngs)
```

## Custom Models

You can also create custom models by modifying the existing configurations:

```python
custom_config = {
    "name": "custom_dinov2",
    "class": "dinov2",
    "config": {
        "num_heads": 8,
        "embed_dim": 512,
        "mlp_ratio": 4,
        "patch_size": 16,
        "depth": 8,
        "img_size": 224,
        "qkv_bias": True,
    }
}

custom_model = load_model(custom_config, rngs=rngs)
```

## Advanced Training Example

Here is a toy example to train [Mlla](https://arxiv.org/abs/2405.16605)
using a [MESA](https://arxiv.org/abs/2205.14083) loss term based on an Exponential
Moving Average of the weights.

```python
from flax import nnx
from jimmy.models.mlla import Mlla
from jimmy.models.emamodel import EmaModel
from optax import adam
from optax.losses import softmax_cross_entropy

x, y = ...
criterion = softmax_cross_entropy

# Defines the model
model = Mlla(
    num_classes=1000,
    depths=[2, 4, 12, 4],
    patch_size=4,
    in_features=3,
    embed_dim=96,
    num_heads=[2, 4, 8, 16],
    layer_window_sizes=[-1, -1, -1, -1],
    rngs=rngs,
)
model.train()

# Defines the wrapper tracking the EMA of the weights
ema_model = EmaModel(model)
ema_model.model.eval()  # To disable dropouts

optimizer = nnx.Optimizer(model, adam(1e-3))


# Core training fn
@nnx.jit
def train(model, ema_model, optimizer, x, y)
    def loss_fn(model, ema_model):
        y_pred = model(x)
        ema_outputs = ema_model(x)

        # Actually you may want to setup a warmup phase, or start MESA after X epochs
        loss = criterion(y_pred, y) + 0.3 * criterion(y_pred, ema_outputs)

    loss, grads = nnx.value_and_grad(loss_fn)(model, ema_model)

    optimizer.update(grads)
    params = nnx.state(model, nnx.Param)

    # Updates the moving average
    ema_model.update(params)

    return loss

for i in range(8):
    loss = train(model, ema_model, optimizer, x, y)
    print(i, loss)
```

## Contributing

Contributions to Jimmy are welcome! Please feel free to submit a Pull Request.

## License

Jimmy is released under the MIT License. See the [LICENSE](LICENSE.md) file for more
details.

## References

This library drawed inspirations from:

- [DINOv2-JAX](https://github.com/kylestach/dinov2-jax/)
- [timm](https://github.com/huggingface/pytorch-image-models/)

## Citation

If you use Jimmy in your research, please cite it as follows:

```bibtex
@software{jimmy2024,
  author = {Clément POIRET},
  title = {Jimmy},
  year = {2024},
  url = {https://github.com/clementpoiret/jimmy},
}
```

For any questions or issues, please open an issue on the GitHub repository.
