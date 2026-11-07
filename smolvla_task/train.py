"""Real forward/backward benchmark before selecting a training batch size."""
import time
import argparse
import gc
import math
import os
import random
import shutil
import signal
from pathlib import Path

import numpy as np

import torch

from .config import BASE, EVIDENCE, OUTPUT, SEED
from .data import AnchorPool, Demonstrations, split
from .model import batch, load, processors, train_mode
from .runtime import append, guard, read, setup, sha, write, resources


def benchmark(batch_size):
    setup()
    guard()
    policy = load()
    pre, post, stats = processors(policy)
    ds = Demonstrations([0])
    samples = [ds.sample(0, i * 25) for i in range(batch_size)]
    prepared = batch(samples, pre)
    for key in ["observation.state", "action"]:
        raw = torch.stack([s[key] for s in samples])
        expected = (raw - stats[key]["mean"]) / (stats[key]["std"] + 1e-8)
        torch.testing.assert_close(prepared[key].cpu(), expected, rtol=1e-5, atol=1e-6)
    params = [p for p in policy.parameters() if p.requires_grad]
    preset = policy.config.get_optimizer_preset()
    optimizer = torch.optim.AdamW(params, lr=preset.lr, betas=preset.betas,
                                 eps=preset.eps, weight_decay=preset.weight_decay, foreach=False)
    evidence = []
    before = policy.model.action_out_proj.weight.detach().cpu().clone()
    for step in range(3):
        begin = time.perf_counter()
        train_mode(policy)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(prepared)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in policy.model.vlm_with_expert.vlm.parameters())
        norm = torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        evidence.append(dict(step=step + 1, loss=float(loss.detach()), gradient_norm=float(norm), seconds=time.perf_counter() - begin))
        print(evidence[-1], flush=True)
    assert torch.count_nonzero(policy.model.action_out_proj.weight.detach().cpu() - before).item() > 0
    result = dict(passed=True, diagnostic_only=True, discarded_optimizer_updates=3, batch_size=batch_size,
                  trainable_parameters=sum(p.numel() for p in params), frozen_storage="bfloat16", trainable_storage="float32",
                  normalization_roundtrip=True, actual_parameter_update=True, steps=evidence, resources=resources())
    write(OUTPUT / f"benchmark-batch{batch_size}.json", result)
    print(result, flush=True)


def learning_rate(step, total, diagnostic):
    warmup = 25 if diagnostic else 500
    warmup = min(warmup, max(1, total//10))
    if step <= warmup:
        return 1e-4 * step / warmup
    if diagnostic:
        return 1e-4
    progress = (step-warmup) / max(1, total-warmup)
    return 2.5e-6 + .5*(1+math.cos(math.pi*progress))*(1e-4-2.5e-6)


def save_checkpoint(out, policy, optimizer, pre, post, step, protocol_hash, sampler_cursor):
    guard()
    destination = out / "checkpoints" / f"{step:06d}"
    writing = destination.with_name("." + destination.name + ".writing")
    assert not writing.exists() and not destination.exists()
    writing.mkdir(parents=True)
    policy.save_pretrained(writing / "pretrained_model")
    pre.save_pretrained(writing / "pretrained_model")
    post.save_pretrained(writing / "pretrained_model")
    if hasattr(policy, "_training_variant"):
        write(writing / "pretrained_model" / "training-variant.json", policy._training_variant)
    torch.save(dict(optimizer=optimizer.state_dict(), step=step, sampler_cursor=sampler_cursor,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), writing/"training_state.pt")
    weights = writing/"pretrained_model/model.safetensors"
    from safetensors import safe_open
    # Verify every saved tensor and key against the live model before commit.
    with safe_open(weights, framework="pt", device="cpu") as saved:
        state = policy.state_dict()
        assert set(saved.keys()) == set(state)
        for key in saved.keys():
            torch.testing.assert_close(saved.get_tensor(key), state[key].detach().cpu(), rtol=0, atol=0)
    write(writing/"manifest.json", dict(step=step, sampler_cursor=sampler_cursor,
          protocol_sha256=protocol_hash, weights_sha256=sha(weights),
          training_state_sha256=sha(writing/"training_state.pt"), all_saved_model_tensors_exact=True))
    os.rename(writing, destination)
    write(out/"latest.json", dict(step=step, checkpoint=destination))
    # Only this new experiment's two older recovery states can be rotated.
    completed = sorted(p for p in destination.parent.iterdir() if p.name.isdigit() and (p/"manifest.json").exists())
    for prior in completed[:-2]:
        assert prior.resolve().parent == (out/"checkpoints").resolve() and not prior.is_symlink()
        assert not any(p.is_symlink() for p in prior.rglob("*"))
        append(out/"checkpoint-rotation.jsonl", dict(removed=prior, new_verified_checkpoint=destination))
        shutil.rmtree(prior)
    print("CHECKPOINT", destination, flush=True)
    return destination


def verify_checkpoint(checkpoint):
    """Run in a separate process to test the exported standard policy file."""
    setup()
    checkpoint = Path(checkpoint)
    manifest = read(checkpoint/"manifest.json")
    assert sha(checkpoint/"pretrained_model/model.safetensors") == manifest["weights_sha256"]
    assert sha(checkpoint/"training_state.pt") == manifest["training_state_sha256"]
    policy = load(checkpoint/"pretrained_model", device="cpu", frozen_bf16=False)
    if hasattr(policy, "_training_variant"):
        from .refinement import configure
        params, optimizer = configure(policy)
    else:
        params = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params)
    state = torch.load(checkpoint/"training_state.pt", weights_only=False, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    assert {int(s["step"]) for s in optimizer.state.values()} == {manifest["step"]}
    assert state["sampler_cursor"] == manifest["sampler_cursor"]
    processors(policy, checkpoint/"pretrained_model")
    write(checkpoint/"reload-verification.json", dict(passed=True, strict_model_reload=True,
          optimizer_loaded=True, step=manifest["step"], sampler_cursor=state["sampler_cursor"],
          stats_verified=True, weights_sha256=manifest["weights_sha256"]))
    print("STRICT_RELOAD_OK", manifest["step"], flush=True)


def train_run(name, scope, steps, batch_size, until=None, episode=0, frames_per_episode=256):
    setup()
    guard()
    out = OUTPUT / "training" / name
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out/"run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    diagnostic = scope == "single-episode"
    episodes = [episode] if diagnostic else split()["train_episodes"]
    assert set(episodes).issubset(split()["train_episodes"])
    protocol = dict(name=name, scope=scope, total_optimizer_steps=steps, batch_size=batch_size,
                    train_episodes=episodes, validation_episodes=split()["validation_episodes"],
                    frames_per_episode=frames_per_episode, seed=SEED, initialization=str(BASE),
                    initialization_sha256=read(OUTPUT/"environment-audit.json")["base"]["sha256"],
                    train_stats_sha256=sha(EVIDENCE/"train-only-stats.json"),
                    peak_learning_rate=1e-4, gradient_clip=10., chunk_size=50,
                    objective="Unmodified official masked flow-matching objective",
                    frozen_VLM=True, frozen_storage="bfloat16", trainable_storage="float32",
                    sampling="Episode-time strata; globally shuffled anchors; no phase/time in model inputs",
                    counts_as_independent_generalization=False if diagnostic else None)
    if (out/"protocol.json").exists():
        assert read(out/"protocol.json") == protocol, "Resume must use the same immutable protocol"
    else:
        write(out/"protocol.json", protocol)
    protocol_hash = sha(out/"protocol.json")
    ds = Demonstrations(episodes)
    pool = AnchorPool(ds, frames_per_episode, SEED)
    write(out/"anchors.json", dict(anchors=pool.anchors, compressed_bytes=pool.compressed_bytes))
    guard()
    rng = np.random.default_rng(SEED+1)
    schedule = []
    while len(schedule) < steps*batch_size:
        schedule.extend(rng.permutation(len(pool)).tolist())
    schedule = np.asarray(schedule[:steps*batch_size], dtype=np.int32)
    schedule_path = out/"sampler.npy"
    if schedule_path.exists():
        np.testing.assert_array_equal(np.load(schedule_path), schedule)
    else:
        np.save(schedule_path, schedule)
    latest = read(out/"latest.json") if (out/"latest.json").exists() else None
    start = latest["step"] if latest else 0
    until = steps if until is None else until
    assert start <= until <= steps
    prior = [read_line for read_line in (out/"updates.jsonl").read_text().splitlines()] if (out/"updates.jsonl").exists() else []
    import json
    assert [json.loads(line)["step"] for line in prior] == list(range(1, start+1)), "Investigate any uncheckpointed journal before resuming"
    source = Path(latest["checkpoint"])/"pretrained_model" if latest else BASE
    policy = load(source)
    pre, post, stats = processors(policy, source)
    params = [p for p in policy.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) == 99880992
    preset = policy.config.get_optimizer_preset()
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=preset.betas, eps=preset.eps,
                                 weight_decay=preset.weight_decay, foreach=False)
    if latest:
        state = torch.load(Path(latest["checkpoint"])/"training_state.pt", weights_only=False, map_location="cpu")
        assert state["step"] == start and state["sampler_cursor"] == start*batch_size
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
        gc.collect()
    write(out/"launch.json", dict(pid=os.getpid(), started_unix=time.time(), target_steps=steps,
                                  starting_step=start, batch_size=batch_size, scope=scope,
                                  cached_anchors=len(pool), compressed_anchor_bytes=pool.compressed_bytes,
                                  resources=resources()))
    stop = [False]
    def request_stop(*_):
        stop[0] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    begin = time.perf_counter()
    completed = start
    recent = []
    for index in range(start, until):
        if index % 25 == 0:
            guard()
        ids = schedule[index*batch_size:(index+1)*batch_size]
        t = time.perf_counter()
        prepared = batch([pool[int(i)] for i in ids], pre)
        train_mode(policy)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(index+1, steps, diagnostic)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(prepared)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        completed = index+1
        row = dict(step=completed, loss=float(loss.detach()), lr=lr, gradient_norm=float(norm),
                   elapsed_seconds=time.perf_counter()-begin, update_seconds=time.perf_counter()-t,
                   anchor_indices=ids.tolist())
        append(out/"updates.jsonl", row)
        recent.append(row["loss"])
        write(out/"progress.json", dict(step=completed, target=steps, scope=scope, batch_size=batch_size,
                                        recent_mean_loss=float(np.mean(recent[-25:])), resources=resources()))
        if completed % 25 == 0 or completed == 1:
            print(dict(step=completed, loss=float(np.mean(recent[-25:])), seconds=time.perf_counter()-begin), flush=True)
        checkpoint_interval = 250 if diagnostic else 1000
        if completed % checkpoint_interval == 0 or completed == until or stop[0]:
            save_checkpoint(out, policy, optimizer, pre, post, completed, protocol_hash, completed*batch_size)
        if stop[0]:
            write(out/"stopped.json", dict(step=completed, clean_checkpoint=True))
            return
    write(out/f"through-{completed:06d}.json", dict(passed=True, step=completed, elapsed_seconds=time.perf_counter()-begin,
                                                   scope=scope, resources=resources()))
