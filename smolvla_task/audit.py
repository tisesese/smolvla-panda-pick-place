from importlib.metadata import version
import platform
import subprocess

import numpy as np
import torch

from .config import BASE, DATA, EVIDENCE, OLD, OUTPUT, SCENE
from .data import Demonstrations, split
from .runtime import read, setup, sha, write, resources


def run():
    setup()
    info = read(DATA / "meta/info.json")
    assert (info["total_episodes"], info["total_frames"], info["fps"]) == (100, 136589, 50)
    assert read(EVIDENCE / "full-native100-admission.json")["passed"]
    division = split()
    proof = read(BASE.parent / "verified.json")
    assert proof["passed"] and proof["optimizer_updates"] == 0
    base_sha = sha(BASE / "model.safetensors")
    assert base_sha == proof["model_sha256"]
    stats_sha = sha(EVIDENCE / "train-only-stats.json")
    assert stats_sha == proof["stats_sha256"]
    ds = Demonstrations([0])
    first, last = ds.sample(0, 0), ds.sample(0, ds.numeric[0]["n"] - 1)
    assert first["action"].shape == (50, 8) and first["observation.state"].shape == (8,)
    assert first["observation.images.fixed_rgb"].shape == (3, 480, 640)
    assert not first["action_is_pad"].any() and last["action_is_pad"].sum().item() == 49
    np.testing.assert_array_equal(first["action"][0], ds.numeric[0]["action"][0])
    assert torch.cuda.is_available()
    a = torch.randn(256, 256, device="cuda")
    assert torch.isfinite(a @ a).all()
    result = dict(passed=True, python=platform.python_version(),
                  versions={n: version(n) for n in ["torch", "torchvision", "lerobot", "mujoco", "gymnasium", "transformers"]},
                  gpu=torch.cuda.get_device_name(0), assets=OLD, output=OUTPUT,
                  dataset=dict(episodes=100, frames=136589, fps=50, split=division),
                  base=dict(directory=BASE, sha256=base_sha, optimizer_updates=0),
                  train_stats_sha256=stats_sha, scene_sha256=sha(SCENE),
                  verified_sample=dict(episode=0, image_shape=[3, 480, 640], state_shape=[8], action_shape=[50, 8], last_frame_padding=49),
                  resources=resources(), historical_pure_vla_success_confirmed=False)
    write(OUTPUT / "environment-audit.json", result)
    print(result, flush=True)

