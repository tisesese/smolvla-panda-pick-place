"""Read the original LeRobot v3 Parquet files without generating Arrow caches."""
import io
from collections import OrderedDict

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .config import CHUNK, DATA, EVIDENCE, IMAGE, TASK
from .runtime import read


class Demonstrations:
    def __init__(self, episodes, action_stride=1):
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        self.meta = LeRobotDatasetMetadata("local/panda_pick_place_100_lowdisk", root=DATA, token=False)
        self.episodes = list(episodes)
        self.action_stride=int(action_stride)
        assert 1<=self.action_stride<=10
        self.numeric = {}
        self.images = OrderedDict()
        for ep in self.episodes:
            path = DATA / self.meta.get_data_file_path(ep)
            table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"])
            rows = np.flatnonzero(np.asarray(table["episode_index"]) == ep)
            table = table.take(rows)
            n = len(table)
            assert n == self.meta.episodes[ep]["length"]
            np.testing.assert_array_equal(np.asarray(table["frame_index"]), np.arange(n))
            self.numeric[ep] = dict(path=path, rows=rows, n=n,
                                    state=np.asarray(table["observation.state"].to_pylist(), np.float32),
                                    action=np.asarray(table["action"].to_pylist(), np.float32))

    def image(self, ep, frame):
        d = self.numeric[ep]
        path = d["path"]
        if path not in self.images:
            # Two episode tables at most, rather than an unbounded HF Arrow cache.
            self.images[path] = pq.read_table(path, columns=[IMAGE])[IMAGE]
            if len(self.images) > 2:
                self.images.popitem(last=False)
        column = self.images.pop(path)
        self.images[path] = column
        raw = column[int(d["rows"][frame])].as_py()
        with Image.open(io.BytesIO(raw["bytes"])) as im:
            return np.array(im.convert("RGB"), dtype=np.uint8, copy=True)

    def sample(self, ep, frame, rgb=None):
        d = self.numeric[ep]
        assert 0 <= frame < d["n"]
        ids = np.minimum(np.arange(CHUNK)*self.action_stride + frame, d["n"] - 1)
        rgb = self.image(ep, frame) if rgb is None else rgb
        return {"observation.state": torch.from_numpy(d["state"][frame].copy()),
                IMAGE: torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255),
                "action": torch.from_numpy(d["action"][ids].copy()),
                "action_is_pad": torch.arange(CHUNK)*self.action_stride + frame >= d["n"], "task": TASK}


class AnchorPool:
    """Keep selected compressed image bytes, rather than full decoded images."""
    def __init__(self, demonstrations, frames_per_episode=256, seed=20261023, focus=False):
        self.ds = demonstrations
        self.anchors = []
        self.bytes = []
        rng = np.random.default_rng(seed)
        plans = read(EVIDENCE / "sampling-plans.json") if focus else {}
        for ep in self.ds.episodes:
            d = self.ds.numeric[ep]
            count = min(frames_per_episode, d["n"])
            # Uniform strata cover the whole episode, then all episodes and
            # times are globally shuffled by the training sampler.
            frames = [int(rng.integers(i*d["n"]//count, (i+1)*d["n"]//count)) for i in range(count)]
            if focus:
                key = focus if focus in ("lookahead", "completion") else "focused"
                frames = list(plans[key][str(ep)])
                if len(frames) != count:
                    raise ValueError("Archived focused plans require 256 anchors per episode")
                if any(not 0 <= frame < d["n"] for frame in frames):
                    raise ValueError("Sampling plan does not match the supplied dataset")
            assert len(frames)==count
            column = pq.read_table(d["path"], columns=[IMAGE])[IMAGE]
            for f in frames:
                self.anchors.append((ep, f))
                self.bytes.append(column[int(d["rows"][f])].as_py()["bytes"])
            del column
        self.compressed_bytes = sum(map(len, self.bytes))
        self.ds.images.clear()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        ep, frame = self.anchors[index]
        with Image.open(io.BytesIO(self.bytes[index])) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        return self.ds.sample(ep, frame, rgb)


def split():
    result = read(EVIDENCE / "split.json")
    assert len(result["train_episodes"]) == 80 and len(result["validation_episodes"]) == 20
    assert set(result["train_episodes"]).isdisjoint(result["validation_episodes"])
    return result

