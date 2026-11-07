"""Measured observations and an independent action converter for Panda lessons.

No policy, inference, grasp planner or learned-data normalization is involved.
The existing, tested ActionAdapter is re-exported rather than duplicated.
"""
#从 MuJoCo 仿真环境中读取 Panda 机械臂的实际状态和相机图像，
#并整理成未来 VLA 模型可以使用的数据结构。
from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np

from .panda_scene import ARM_JOINTS, CAMERA_NAME, FINGER_JOINTS, PandaTableScene
from .vla_interface import ActionAdapter

__all__ = ["ActionAdapter", "ObservationAdapter"]
STATE_KEY = "observation.state"
IMAGE_KEY = f"observation.images.{CAMERA_NAME}"
DEFAULT_TASK = "Move joint1 slightly, then open, close and reopen the gripper."


class ObservationAdapter:
    """Read actual qpos through named joint IDs and their qpos addresses.

    An arm joint ID is NOT a qpos array index in general: free/ball joints use
    multiple qpos coordinates. Resolve jnt_qposadr even when IDs happen to match
    qpos addresses in this specific scene. Reading never advances the physics.
    """

    def __init__(self, scene: PandaTableScene, task: str = DEFAULT_TASK):
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        self.scene = scene
        self.task = task
        self._arm = self._resolve(ARM_JOINTS, mujoco.mjtJoint.mjJNT_HINGE)
        self._fingers = self._resolve(FINGER_JOINTS, mujoco.mjtJoint.mjJNT_SLIDE)
        self._arm_addresses = np.array([item["qpos_address"] for item in self._arm])
        self._finger_addresses = np.array([item["qpos_address"] for item in self._fingers])
        # Each upstream finger slides 0..0.04 m; summed nominal aperture is 0..0.08 m.
        self.closed_width_m = sum(item["joint_range"][0] for item in self._fingers)
        self.open_width_m = sum(item["joint_range"][1] for item in self._fingers)
        if self.open_width_m <= self.closed_width_m:
            raise ValueError("Finger limits must define a positive opening span")

    def _resolve(self, names, expected_type):
        bindings = []
        for name in names:
            joint_id = int(self.scene.model.joint(name).id)
            if self.scene.model.jnt_type[joint_id] != expected_type:
                raise ValueError(f"Unexpected joint type for {name}")
            if not self.scene.model.jnt_limited[joint_id]:
                raise ValueError(f"Expected finite joint limits for {name}")
            bindings.append({
                "joint_name": name,
                "joint_id": joint_id,
                "qpos_address": int(self.scene.model.jnt_qposadr[joint_id]),
                "joint_range": self.scene.model.jnt_range[joint_id].tolist(),
                "unit": "rad" if expected_type == mujoco.mjtJoint.mjJNT_HINGE else "m",
            })
        return tuple(bindings)

    def finger_width_m(self):
        """Actual sum of finger slide qpos, in metres; not an actuator value.

        This is nominal kinematic jaw aperture, not a mesh-to-mesh distance.
        """
        width = float(self.scene.data.qpos[self._finger_addresses].sum())
        if not np.isfinite(width):
            raise RuntimeError("Nonfinite measured finger width")
        return width

    def state(self):
        """Return a detached float32 (8,) vector of actual measured state."""
        arm_positions = self.scene.data.qpos[self._arm_addresses]
        width = self.finger_width_m()
        if not np.isfinite(arm_positions).all():
            raise RuntimeError("Nonfinite measured arm joint position")
        state = np.empty(8, dtype=np.float32)
        state[:7] = arm_positions
        state[7] = np.clip(
            (width - self.closed_width_m) / (self.open_width_m - self.closed_width_m),
            0.0, 1.0,
        )
        return state

    def observe(self, include_rgb=True):
        """Full observation by default; skip RGB only in logging/control loops."""
        observation = {STATE_KEY: self.state(), "task": self.task}
        if include_rgb:
            image = self.scene.render_rgb()
            if (not isinstance(image, np.ndarray) or image.shape != (480, 640, 3)
                    or image.dtype != np.uint8 or not np.isfinite(image).all()):
                raise RuntimeError("Expected finite uint8 RGB image of shape (480, 640, 3)")
            observation[IMAGE_KEY] = image.copy()
        return observation

    def schema(self):
        """Return names, explicit address bindings, units and normalization."""
        return {
            STATE_KEY: {
                "shape": [8], "dtype": "float32",
                "arm_elements_0_to_6": deepcopy(list(self._arm)),
                "gripper_element_7": {
                    "meaning": "actual nominal finger aperture normalized to [0,1]",
                    "unit": "dimensionless", "closed": 0.0, "open": 1.0,
                    "closed_width_m": self.closed_width_m,
                    "open_width_m": self.open_width_m,
                    "finger_bindings": deepcopy(list(self._fingers)),
                },
            },
            IMAGE_KEY: {"shape": [480, 640, 3], "dtype": "uint8", "color_space": "RGB"},
            "task": {"type": "str", "value": self.task},
            "state_is_measured_not_action_or_ctrl": True,
            "learned_data_normalization": False,
        }
