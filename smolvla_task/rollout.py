import time
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import BASE, EVIDENCE, IMAGE, OUTPUT, TASK
from .data import Demonstrations
from .runtime import append, read, setup, write


class Video:
    def __init__(self, path):
        import av
        self.av = av
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=10)
        self.stream.width, self.stream.height = 640, 480
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "24", "preset": "veryfast"}

    def add(self, rgb):
        frame = self.av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def replay(episode=0):
    """Independent physics/label alignment diagnostic, not a learned result."""
    from .environment import PandaTask
    from PIL import Image
    setup()
    out = OUTPUT / "diagnostics" / f"teacher-replay-episode{episode:03d}"
    assert not (out / "result.json").exists(), "Retain completed diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    ds = Demonstrations([episode])
    plan = read(EVIDENCE / "plan.json")
    position = plan["positions"][episode]
    print("REPLAY_POSITION", position, flush=True)
    xyz = position.get("xyz_m", position.get("cube_xyz_m", position.get("position_m")))
    if xyz is None:
        raise KeyError(f"Unknown position schema: {position}")
    seed = position["seed"]
    movie = Video(out / "teacher-replay.mp4")
    max_state_error = 0.
    begin = time.perf_counter()
    with PandaTask() as env:
        obs, _ = env.reset(seed=seed, options={"cube_xyz_m": xyz})
        assert env.observation_space.contains(obs)
        Image.fromarray(obs[IMAGE]).save(out / "initial.png")
        for frame, action in enumerate(ds.numeric[episode]["action"]):
            state_error = float(np.max(np.abs(obs["observation.state"]-ds.numeric[episode]["state"][frame])))
            max_state_error = max(max_state_error, state_error)
            obs, reward, done, truncated, info = env.step(action, include_rgb=frame % 5 == 0)
            if frame % 5 == 0:
                movie.add(obs[IMAGE])
            if done or truncated:
                break
            if frame % 250 == 0:
                print("REPLAY_FRAME", frame, "state_error", max_state_error, flush=True)
        movie.close()
        Image.fromarray(env.render()).save(out / "terminal.png")
        result = dict(diagnostic="teacher_action_replay_only", pure_vla=False, episode=episode,
                      success=bool(info["success"]), commands=frame+1, sim_seconds=float(env.native.data.time),
                      stable_seconds=info["stable_seconds"], reason=info["reason"],
                      maximum_state_difference_rad_or_normalized_aperture=max_state_error,
                      wall_seconds=time.perf_counter()-begin)
    assert result["success"] and max_state_error < 1e-5, result
    write(out / "result.json", result)
    print(result, flush=True)


def policy_sample(obs):
    return {"observation.state": torch.from_numpy(obs["observation.state"].copy()),
            IMAGE: torch.from_numpy(obs[IMAGE].copy()).permute(2, 0, 1).float() / 255., "task": TASK}


def run(model, episode=0, horizon=25, label="development", noise_seed=20261023,
        position=None, bundle=None, scope=None, brightness=1., device="cuda", gripper_gain=1.,
        precision="bfloat16", num_steps=None, gripper_knee=1., gripper_rate=.06, action_samples=1, light_scale=1.):
    """Only RGB, measured joints/gripper and instruction enter the policy."""
    from .environment import PandaTask
    from .model import load, processors, batch
    from PIL import Image
    setup(noise_seed)
    if not isinstance(action_samples,int) or not 1<=action_samples<=16:
        raise ValueError("Action samples must be an integer within [1,16]")
    if device == "cpu":
        torch.set_num_threads(2)
    model = BASE if str(model) == "base" else Path(model)
    metadata=read(model/"training-variant.json") if (model/"training-variant.json").exists() else {}
    action_stride=int(metadata.get("action_stride",1))
    horizon=metadata.get("execution_window",25) if horizon is None else horizon
    if not 1<=action_stride<=10 or not 1<=horizon<=50 or (action_stride>1 and horizon>=50):
        raise ValueError("Invalid temporal action window or interpolation lookahead")
    trial = f"seed{position['seed']}" if position is not None else f"episode{episode:03d}-K{horizon}"
    out = OUTPUT / "rollouts" / label / trial
    assert not out.exists(), "Completed and partial rollouts are never silently overwritten"
    out.mkdir(parents=True)
    write(out/"launch.json",dict(pid=os.getpid(),label=label,started_unix=time.time(),model=model))
    write(out / "protocol.json", dict(model=model, position_source_episode=episode if position is None else None,
                                      position=position, execution_window=horizon,
                                      action_chunk=50, noise_seed=noise_seed, pure_vla=True,
                                      action_stride=action_stride,predicted_target_hz=50/action_stride,
                                      native_commands_per_window=horizon*action_stride,
                                      interpolation="None" if action_stride==1 else "Linear between adjacent policy targets at original50Hz",
                                      policy_inputs=[IMAGE, "observation.state", "task"], task_seconds=30,
                                      physics_Hz=500, control_Hz=50, sim_paused_during_inference=True,
                                      measured_tracking_window_rad=.08, max_target_increment_rad=.03,
                                      max_gripper_increment=gripper_rate, image_brightness_multiplier=brightness,
                                      scene_light_scale=light_scale,
                                      inference_device=device, compute_precision="float32" if device=="cpu" or precision=="float32" else "bfloat16 autocast",
                                      flow_integration_steps=num_steps,
                                      independent_action_samples=action_samples,
                                      action_aggregation="Single policy sample" if action_samples==1 else "Arithmetic mean of independent samples conditioned on identical permitted observations",
                                      image_preprocessing=metadata,
                                      gripper_command_gain=gripper_gain,
                                      gripper_calibration_knee=gripper_knee,
                                      gripper_calibration_formula="clip(g-(gain-1)*max(knee-g,0),0,1), then original increment limit"))
    if bundle is None:
        p = load(model, device=device, frozen_bf16=precision!="float32")
        pre, post, _ = processors(p, model)
    else:
        p, pre, post = bundle
    if num_steps is not None:
        if not 1<=num_steps<=100:raise ValueError("Invalid flow integration step count")
        p.config.num_steps=num_steps
    torch.manual_seed(noise_seed)
    torch.cuda.manual_seed_all(noise_seed)
    p.reset()
    if position is None:
        position = read(EVIDENCE / "plan.json")["positions"][episode]
    xyz = position.get("xyz_m", position.get("cube_xyz_m", position.get("position_m")))
    if xyz is None:
        raise KeyError(position)
    movie = Video(out / "pure-vla.mp4")
    latencies = []
    commands = 0
    lifted = bilateral = False
    min_distance = np.inf
    started = time.perf_counter()
    with PandaTask(gripper_gain=gripper_gain,gripper_knee=gripper_knee,gripper_rate=gripper_rate,light_scale=light_scale) as env:
        obs, _ = env.reset(seed=position["seed"], options={"cube_xyz_m": xyz})
        done = truncated = False
        info = dict(success=False, reason=None, stable_seconds=0.)
        try:
            while not (done or truncated):
                t = time.perf_counter()
                policy_obs = dict(obs)
                if brightness != 1.:
                    policy_obs[IMAGE] = np.clip(obs[IMAGE].astype(np.float32)*brightness, 0, 255).astype(np.uint8)
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" and precision!="float32" else nullcontext()
                with torch.inference_mode(), amp:
                    predicted = p.predict_action_chunk(batch([policy_sample(policy_obs) for _ in range(action_samples)], pre))
                sampled = post(predicted).detach().cpu().numpy()
                assert sampled.shape == (action_samples,50,8)
                raw = sampled[0] if action_samples==1 else sampled.mean(axis=0)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter()-t)
                assert raw.shape == (50, 8) and np.isfinite(raw).all()
                np.savez_compressed(out/f"chunk-{len(latencies):03d}.npz", actions=raw,
                                    individual_policy_samples=sampled,
                                    state=obs["observation.state"], sim_seconds=float(env.native.data.time))
                executed=raw[:horizon] if action_stride==1 else np.asarray([
                    raw[k]+(tick/action_stride)*(raw[k+1]-raw[k])
                    for k in range(horizon) for tick in range(action_stride)],dtype=np.float32)
                for slot, action in enumerate(executed):
                    state_before = obs["observation.state"].copy()
                    need_image = (commands + 1) % 5 == 0 or slot == len(executed)-1
                    obs, _, done, truncated, info = env.step(action, include_rgb=need_image)
                    commands += 1
                    m = env.native.last_info
                    bilateral |= m["bilateral"]
                    lifted |= env.native.ever_lifted
                    min_distance = min(min_distance, m["cube_grasp_distance_m"])
                    append(out / "commands.jsonl", dict(command=commands, sim_seconds=float(env.native.data.time),
                           raw=action, issued=info["action"]["issued"], state_before=state_before,
                           generated_action_slot=slot//action_stride,interpolation_fraction=(slot%action_stride)/action_stride,
                           state_after=obs["observation.state"], evaluation_only=m))
                    if commands % 5 == 0:
                        movie.add(obs[IMAGE])
                    if done or truncated:
                        break
                print(dict(commands=commands, sim_seconds=float(env.native.data.time),
                           closest_m=min_distance, bilateral=bilateral, lifted=lifted), flush=True)
        finally:
            movie.close()
            Image.fromarray(env.render()).save(out / "terminal.png")
        result = dict(pure_vla=True, success=bool(info["success"]), model=model, episode=episode,
                      execution_window=horizon,
                      action_stride=action_stride,predicted_target_hz=50/action_stride,
                      independent_action_samples=action_samples,
                      selection_scope=scope or ("training-position diagnostic" if episode in read(EVIDENCE/"split.json")["train_episodes"] else "validation position"),
                      commands=commands, sim_seconds=float(env.native.data.time), stable_seconds=info["stable_seconds"],
                      reason=info["reason"], ever_bilateral=bilateral, ever_lifted=lifted,
                      closest_cube_m=min_distance, inference_mean_seconds=float(np.mean(latencies)),
                      wall_seconds=time.perf_counter()-started, physical_peaks=env.native.peak)
        result.update(dropped=bool(env.native.last_info["cube_xyz_m"][2] < .70 or info["reason"] in ["lost_bilateral_payload_contact","payload_outside_gripper","cube_fallen_off_work_surface"]),
                      cube_below_work_surface=bool(env.native.last_info["cube_xyz_m"][2] < .70),
                      collision=bool(env.native.peak["robot_obstacle_force_N"] > 0 or "collision" in str(info["reason"])),
                      image_brightness_multiplier=brightness, seed=position["seed"], inference_device=device,
                      scene_light_scale=light_scale,inference_latencies_seconds=latencies,
                      gripper_command_gain=gripper_gain,gripper_calibration_knee=gripper_knee,
                      max_gripper_increment=gripper_rate,
                      compute_precision=precision if device=="cuda" else "float32",
                      flow_integration_steps=p.config.num_steps)
        write(out / "result.json", result)
        print(result, flush=True)
    return result

