"""Small shared helpers; no installed package or old experiment is modified."""
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", LP_NUM_THREADS="4",
)

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import OUTPUT, SEED


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def convert(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=convert, allow_nan=False) + "\n")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def setup(seed=SEED):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def resources():
    import psutil
    import shutil
    probe = OUTPUT
    while not probe.exists():
        probe = probe.parent
    d = shutil.disk_usage(probe)
    return dict(unix_s=time.time(), ram_available_GiB=psutil.virtual_memory().available / 2**30,
                D_free_GiB=d.free / 2**30,
                gpu_allocated_GiB=torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else None,
                gpu_peak_GiB=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None)


def guard():
    r = resources()
    if r["D_free_GiB"] < 20:
        raise RuntimeError("Output filesystem free space is below the 20 GiB reserve")
    if r["ram_available_GiB"] < .4:
        raise RuntimeError("Available WSL RAM is below 0.4 GiB")
    return r

