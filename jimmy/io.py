import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlretrieve

import jax.numpy as jnp
import orbax
import py7zr
from flax import nnx
from flax.training import orbax_utils
from tqdm import tqdm

DEFAULT_MODEL_DIR = Path.home() / ".jimmy" / "hub" / "checkpoints"
FILTERS = [{"id": py7zr.FILTER_ZSTD, "level": 3}]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save(
    params: nnx.State,
    specifications: Dict,
    name: str,
    model_dir: Optional[str] = None,
    overwrite: bool = False,
    compress: bool = True,
) -> Path:
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    ckpt_dir = model_dir / name
    compressed_ckpt = model_dir / f"{name}.jim"

    if overwrite:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        if compressed_ckpt.exists():
            shutil.rmtree(compressed_ckpt)
    model_dir.mkdir(exist_ok=True, parents=True)

    ckpt = {
        "params": params,
        "specifications": specifications,
    }

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(ckpt_dir, ckpt, save_args=save_args)

    if compress:
        with py7zr.SevenZipFile(compressed_ckpt, "w",
                                filters=FILTERS) as archive:
            archive.writeall(ckpt_dir, arcname=name)
        shutil.rmtree(ckpt_dir)

        logger.info(f"Saved {name} to {compressed_ckpt}.")
        return compressed_ckpt

    logger.info(f"Saved {name} to {ckpt_dir}.")
    return ckpt_dir


# TODO: TEST ALL CASES
def load(name: str,
         params: nnx.State,
         specifications: Dict,
         url: Optional[str] = None,
         model_dir: Optional[str] = None):
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir = model_dir / name
    compressed_ckpt = model_dir / f"{name}.jim"

    if not ckpt_dir.exists() and not compressed_ckpt.exists() and url is None:
        raise ValueError(f"{name} not found in {model_dir}.")

    if (not ckpt_dir.exists() and not compressed_ckpt.exists()) and url:
        logger.info(f"Downloading {name} to {compressed_ckpt}...")
        urlretrieve(url, compressed_ckpt)

    if not ckpt_dir.exists() and compressed_ckpt.exists():
        archive = py7zr.SevenZipFile(compressed_ckpt, mode="r")
        archive.extractall(model_dir)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    target = {
        "params": params,
        "specifications": specifications,
    }
    raw = orbax_checkpointer.restore(ckpt_dir, item=target)

    return raw
