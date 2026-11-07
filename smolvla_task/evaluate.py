"""Teacher-forced diagnostics, always separate from closed-loop task success."""
import time
from pathlib import Path

import numpy as np
import torch

from .config import BASE, OUTPUT
from .data import AnchorPool, Demonstrations, split
from .model import batch, load, processors
from .runtime import setup, sha, write


def offline(model, label, episode=None, batch_size=16):
    assert isinstance(batch_size, int) and batch_size > 0
    setup()
    model = BASE if str(model) == "base" else Path(model)
    out = OUTPUT/"evaluations"/label
    assert not out.exists(), "Evaluation artifacts must not be overwritten"
    out.mkdir(parents=True)
    episodes = [episode] if episode is not None else split()["validation_episodes"]
    from .runtime import read
    variant=read(model/"training-variant.json") if (model/"training-variant.json").exists() else {}
    ds = Demonstrations(episodes,action_stride=variant.get("action_stride",1))
    pool = AnchorPool(ds, frames_per_episode=32 if episode is not None else 8, seed=98762026)
    p = load(model)
    pre, post, _ = processors(p, model)
    p.eval()
    losses = []
    errors = []
    predictions = []
    timings = []
    for begin in range(0, len(pool), batch_size):
        samples = [pool[i] for i in range(begin, min(begin+batch_size, len(pool)))]
        x = batch(samples, pre)
        torch.manual_seed(44000+begin)
        torch.cuda.manual_seed_all(44000+begin)
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = p(x)
            pred = post(p.predict_action_chunk(x)).cpu()
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-t)
        assert torch.isfinite(loss) and torch.isfinite(pred).all()
        losses.append((float(loss), len(samples)))
        for local, sample in enumerate(samples):
            valid = ~sample["action_is_pad"]
            error = (pred[local]-sample["action"]).abs()[valid]
            ep, frame = pool.anchors[begin+local]
            errors.append(dict(episode=ep, frame=frame, valid_steps=int(valid.sum()),
                               joint_MAE_rad=float(error[:, :7].mean()), gripper_MAE=float(error[:, 7].mean())))
            predictions.append(pred[local].numpy())
    np.savez_compressed(out/"predictions.npz", actions=np.stack(predictions), anchors=np.asarray(pool.anchors))
    result = dict(teacher_forced_only=True, task_success_rate=None, model=model,
                  action_stride=variant.get("action_stride",1),
                  batch_size=batch_size, anchor_seed=98762026,
                  torch_noise_seed_rule="44000 + batch anchor offset; identical across compared models",
                  model_sha256=sha(model/"model.safetensors"), episodes=episodes, samples=len(pool),
                  scope="training-position diagnostic" if episode is not None else "fixed20 validation episodes",
                  mean_flow_loss=sum(l*n for l,n in losses)/sum(n for _,n in losses),
                  joint_MAE_rad=float(np.mean([r["joint_MAE_rad"] for r in errors])),
                  gripper_MAE=float(np.mean([r["gripper_MAE"] for r in errors])),
                  batch_wall_seconds=timings, per_anchor=errors)
    write(out/"metrics.json", result)
    print({k:v for k,v in result.items() if k not in ["per_anchor", "episodes"]}, flush=True)
    return result

