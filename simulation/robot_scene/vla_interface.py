"""Explicit measured robot state and policy-to-actuator mapping for Panda.

This module does not load a learned policy.  In particular, the three vectors
below have different meanings and must not be used interchangeably:

* robot state: measured joint positions and measured normalized finger aperture;
* policy action: requested joint targets and requested normalized finger opening;
* MuJoCo ctrl: actuator targets, with the gripper expressed in its native units.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .panda_scene import ARM_JOINTS, CAMERA_NAME, PandaTableScene

DEFAULT_TASK = "Pick up the red cube."
MAX_FINGER_OPENING_M = 0.08
STATE_KEY = "observation.state"
IMAGE_KEY = f"observation.images.{CAMERA_NAME}"

STATE_SCHEMA = tuple(
    [{"index": i, "name": name, "meaning": "measured arm joint position",
      "unit": "rad"} for i, name in enumerate(ARM_JOINTS)]
    + [{"index": 7, "name": "gripper_opening_normalized",
        "meaning": "measured total finger aperture / 0.08 m, clipped to [0, 1]",
        "unit": "dimensionless", "closed": 0.0, "open": 1.0}]
)
ACTION_SCHEMA = tuple(
    [{"index": i, "name": name, "meaning": "absolute arm joint position target",
      "unit": "rad"} for i, name in enumerate(ARM_JOINTS)]
    + [{"index": 7, "name": "gripper_command_normalized",
        "meaning": "requested gripper opening, clipped to [0, 1]",
        "unit": "dimensionless", "closed": 0.0, "open": 1.0}]
)


class ActionAdapter:
    """Pure conversion from an 8D policy action to native MuJoCo ctrl.

    Construction copies the model ranges. Conversion neither changes the model
    nor reads/writes MjData, and never advances the simulation. Arm targets are
    clipped to their actuator ranges; normalized gripper commands are clipped
    before mapping 0 -> 0 and 1 -> 255. The returned array owns its data.
    """

    def __init__(self, model: Any):
        if int(model.nu) != 8:
            raise ValueError("This Panda action adapter requires exactly 8 actuators")
        ranges = np.array(model.actuator_ctrlrange, dtype=np.float64, copy=True)
        if (ranges.shape != (8, 2) or not np.isfinite(ranges).all()
                or np.any(ranges[:, 0] >= ranges[:, 1])):
            raise ValueError("Model actuator control ranges must be finite and increasing")
        if not np.allclose(ranges[7], [0.0, 255.0], rtol=0, atol=1e-12):
            raise ValueError("Expected the upstream Panda gripper control range [0, 255]")
        # Refuse accidental use with a differently ordered robot actuator model.
        names = [model.actuator(i).name for i in range(8)]
        if names != [f"actuator{i}" for i in range(1, 9)]:
            raise ValueError("Unexpected Panda actuator order")
        ranges.setflags(write=False)
        self._ctrl_ranges = ranges

    @property
    def ctrl_ranges(self) -> np.ndarray:
        """A detached copy of the native actuator limits."""
        return self._ctrl_ranges.copy()

    def to_ctrl(self, policy_action: Any) -> np.ndarray:
        """Validate, clip and convert a policy action without stepping MuJoCo."""
        try:
            action = np.asarray(policy_action, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("policy action must be a finite numeric vector of shape (8,)") from exc
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("policy action must be a finite numeric vector of shape (8,)")
        ctrl = np.empty(8, dtype=np.float64)
        ctrl[:7] = np.clip(action[:7], self._ctrl_ranges[:7, 0], self._ctrl_ranges[:7, 1])
        opening = float(np.clip(action[7], 0.0, 1.0))
        lo, hi = self._ctrl_ranges[7]
        ctrl[7] = lo + opening * (hi - lo)
        return ctrl


class VLAInterface:
    """A small observation/action layer over the existing native scene.

    Full observations have three flat keys: observation.state, the fixed RGB
    image, and task. Set include_rgb=False only for simulation/debugging loops
    that do not need an image; that returns a partial observation deliberately.
    State is reconstructed from measured qpos on every call, never from ctrl.
    No learned-model package, weights, wrapper, or inference is involved.
    """

    def __init__(self, scene: PandaTableScene, task: str = DEFAULT_TASK):
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        self.scene = scene
        self.task = task
        self.action_adapter = ActionAdapter(scene.model)

    def robot_state(self) -> np.ndarray:
        """Return measured 7D arm qpos (rad) + normalized measured aperture."""
        measured = self.scene.observe()
        positions = np.asarray(measured["joint_positions"], dtype=np.float64)
        width = float(measured["gripper_width"])
        if positions.shape != (7,) or not np.isfinite(positions).all() or not np.isfinite(width):
            raise RuntimeError("Scene reported an invalid measured robot state")
        state = np.empty(8, dtype=np.float32)
        state[:7] = positions
        state[7] = np.clip(width / MAX_FINGER_OPENING_M, 0.0, 1.0)
        return state

    def observe(self, include_rgb: bool = True) -> dict[str, Any]:
        """Return detached measured state, task text and optionally fixed RGB."""
        observation = {STATE_KEY: self.robot_state(), "task": self.task}
        if include_rgb:
            image = self.scene.render_rgb()
            if (not isinstance(image, np.ndarray) or image.shape != (480, 640, 3)
                    or image.dtype != np.uint8):
                raise RuntimeError("Fixed camera must return uint8 RGB of shape (480, 640, 3)")
            observation[IMAGE_KEY] = image.copy()
        return observation

    def step_policy(self, policy_action: Any, n_steps: int = 10,
                    include_rgb: bool = True) -> dict[str, Any]:
        """Apply a normalized policy target through the independent adapter."""
        ctrl = self.action_adapter.to_ctrl(policy_action)
        self.scene.step(ctrl, n_steps=n_steps)
        return self.observe(include_rgb=include_rgb)

    def schema(self) -> dict[str, Any]:
        """Serializable semantics, limits and dimensions for future integration."""
        return {
            STATE_KEY: {"shape": [8], "dtype": "float32",
                        "elements": deepcopy(list(STATE_SCHEMA))},
            IMAGE_KEY: {"shape": [480, 640, 3], "dtype": "uint8", "color_space": "RGB"},
            "task": {"type": "str", "default": DEFAULT_TASK},
            "policy_action": {"shape": [8], "unit": "rad for arm; dimensionless gripper",
                              "elements": deepcopy(list(ACTION_SCHEMA))},
            "mujoco_actuator_ctrl": {
                "shape": [8], "dtype": "float64", "arm_unit": "rad",
                "gripper_unit": "native actuator command", "ranges": self.action_adapter.ctrl_ranges.tolist(),
                "mapping": "clip arm per actuator range; ctrl[7] = 255 * clip(action[7], 0, 1)",
            },
            "robot_state_is_measured_not_action_or_ctrl": True,
        }
