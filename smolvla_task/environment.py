"""Gymnasium adapter preserving the data collection scene and native physics."""
import sys
import string

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import IMAGE, OLD, TASK


def float32_interior_limits(limits):
    """Representable endpoints contained in the native float64 control range."""
    limits = np.asarray(limits, dtype=np.float64)
    low = limits[:, 0].astype(np.float32)
    high = limits[:, 1].astype(np.float32)
    low = np.where(low.astype(np.float64) < limits[:, 0],
                   np.nextafter(low, np.float32(np.inf)), low)
    high = np.where(high.astype(np.float64) > limits[:, 1],
                    np.nextafter(high, np.float32(-np.inf)), high)
    return low, high


class PandaTask(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode="rgb_array", gripper_gain=1., gripper_knee=1., gripper_rate=.06, light_scale=1.):
        if not np.isfinite(light_scale) or not .85<=light_scale<=1.15:
            raise ValueError("Light scale must be finite and within [.85,1.15]")
        if not np.isfinite(gripper_gain) or not 1. <= gripper_gain <= 2.:
            raise ValueError("Gripper command calibration must be finite and within [1,2]")
        if not np.isfinite(gripper_knee) or not .5 <= gripper_knee <= 1.:
            raise ValueError("Gripper calibration knee must be finite and within [.5,1]")
        self.gripper_gain=float(gripper_gain)
        self.gripper_knee=float(gripper_knee)
        if not np.isfinite(gripper_rate) or not 0. < gripper_rate <= .06:
            raise ValueError("Gripper rate must be in (0,.06]")
        self.gripper_rate=float(gripper_rate)
        # Reuse the verified scene and task implementation read-only. Training
        # and inference never call a historical experiment's pipeline.
        if str(OLD) not in sys.path:
            sys.path.insert(0, str(OLD))
        from experiments.panda_pick_place.environment import PandaPickPlaceEnv

        class Native(PandaPickPlaceEnv):
            def hold(self):
                # A terminated simulation simply stops integrating physics.
                pass

        self.native = Native()
        self.light_scale=float(light_scale)
        for name in ["light_ambient","light_diffuse","light_specular"]:
            getattr(self.native.model,name)[:]*=self.light_scale
        self.render_mode = render_mode
        low, high = float32_interior_limits(self.native.limits)
        self.action_space = spaces.Box(low, high)
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            IMAGE: spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
            "task": spaces.Text(max_length=128, charset=string.ascii_letters + string.digits + " ."),
        })

    def observation(self, include_rgb=True):
        obs = self.native.observe(include_rgb=include_rgb)
        result = {"observation.state": np.asarray(obs["observation.state"], np.float32), "task": TASK}
        if include_rgb:
            result[IMAGE] = np.asarray(obs[IMAGE], np.uint8)
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        if "cube_xyz_m" not in options:
            options["cube_xyz_m"] = [float(self.np_random.uniform(.105, .155)),
                                     float(self.np_random.uniform(-.215, -.165)), .782]
        self.native.reset(seed=seed or 0, options=options, include_rgb=False)
        for _ in range(50):
            _, _, done, truncated, info = self.native.step(self.native.initial_ctrl, include_rgb=False)
            if done or truncated:
                raise RuntimeError("Environment terminated during the dataset's neutral reset interval")
        return self.observation(), {"reset_position": options["cube_xyz_m"], "sim_time_s": float(self.native.data.time)}

    def limit_action(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (8,) or not np.isfinite(raw).all():
            raise ValueError("Model output must be a finite 8D absolute target")
        low, high = float32_interior_limits(self.native.limits)
        bounded = np.clip(raw, low.astype(np.float64), high.astype(np.float64))
        # Generic feedback constraint based only on measured robot joints.
        # Prevent accumulating position targets faster than servos can track.
        measured = self.native.observer.state()[:7]
        tracked = bounded.copy()
        tracked[:7] = np.clip(tracked[:7], measured-.08, measured+.08)
        tracked[7] = np.clip(bounded[7]-(self.gripper_gain-1.)*max(self.gripper_knee-bounded[7],0.),0.,1.)
        increments = np.r_[np.full(7, .03), self.gripper_rate]
        issued = self.native.previous + np.clip(tracked - self.native.previous, -increments, increments)
        command = np.clip(issued.astype(np.float32), low, high)
        assert np.all(command.astype(np.float64) >= self.native.limits[:, 0])
        assert np.all(command.astype(np.float64) <= self.native.limits[:, 1])
        return command, dict(raw=raw, issued=command.astype(np.float64), endpoint_projection=np.abs(raw-bounded),
                                              measured_tracking_projection=np.r_[np.abs(bounded[:7]-tracked[:7]),0.],
                                              gripper_calibration_delta=float(tracked[7]-bounded[7]),
                                              float32_quantization=np.abs(command.astype(np.float64)-issued),
                                              rate_projection=np.abs(tracked-issued))

    def step(self, action, *, include_rgb=True):
        issued, action_info = self.limit_action(action)
        _, reward, terminated, truncated, info = self.native.step(issued, include_rgb=False)
        info = dict(info, action=action_info)
        return self.observation(include_rgb), reward, terminated, truncated, info

    def render(self):
        return self.native.image()

    def close(self):
        self.native.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

