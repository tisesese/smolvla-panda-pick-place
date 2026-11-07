"""Small actual-pose DLS target correction; no force or live qpos writes."""
from pathlib import Path
import json
import numpy as np
import mujoco

PARAMETERS_PATH = Path(__file__).with_name('parameters.json')
PARAMETERS = json.loads(PARAMETERS_PATH.read_text())
VERSION = PARAMETERS['controller_version']


class BoundedDescentCorrection:
    def __init__(self, scene, site_id, nominal_target):
        self.site_id = site_id
        self.anchor = np.asarray(nominal_target, np.float64).copy()
        self.dofs = scene.arm_dofs.copy()
        self.ranges = scene.model.actuator_ctrlrange[:7].copy()
        self.jacp = np.zeros((3, scene.model.nv))
        self.jacr = np.zeros((3, scene.model.nv))
        self.records = []

    def propose(self, scene, position, quaternion, previous_target):
        p = PARAMETERS
        previous = np.asarray(previous_target, np.float64)
        site = scene.data.site(self.site_id)
        actual_quat = np.empty(4)
        mujoco.mju_mat2Quat(actual_quat, site.xmat)
        local_error = np.empty(3)
        mujoco.mju_subQuat(local_error, quaternion, actual_quat)
        world_error = site.xmat.reshape(3, 3) @ local_error
        position_error = np.asarray(position) - site.xpos
        weight = p['rotation_weight_m_per_rad']
        error = np.r_[position_error, weight * world_error]
        mujoco.mj_jacSite(scene.model, scene.data, self.jacp, self.jacr, self.site_id)
        jac = np.vstack((self.jacp[:, self.dofs], weight * self.jacr[:, self.dofs]))
        delta = p['feedback_gain'] * jac.T @ np.linalg.solve(
            jac @ jac.T + p['damping'] ** 2 * np.eye(6), error)
        maximum = p['maximum_joint_target_increment_rad']
        if np.abs(delta).max() > maximum:
            delta *= maximum / np.abs(delta).max()
        cap = p['maximum_joint_target_offset_from_nominal_rad']
        lower = np.maximum(self.ranges[:, 0], self.anchor - cap)
        upper = np.minimum(self.ranges[:, 1], self.anchor + cap)
        target = np.clip(previous + delta, lower, upper).astype(np.float32).astype(np.float64)
        if not np.isfinite(target).all(): raise RuntimeError('nonfinite_descent_correction')
        # The runner transmits float32 actions; allow only its rounding budget.
        assert np.abs(target - previous).max() <= maximum + 4e-7
        assert np.abs(target - self.anchor).max() <= cap + 4e-7
        record = {'version': VERSION, 'correction_index': len(self.records),
                  'actual_position_error_m': float(np.linalg.norm(position_error)),
                  'actual_orientation_error_rad': float(np.linalg.norm(world_error)),
                  'actual_position_m': site.xpos.copy(), 'actual_quaternion_wxyz': actual_quat,
                  'world_orientation_error': world_error, 'nominal_anchor_q_rad': self.anchor.copy(),
                  'previous_target_q_rad': previous.copy(), 'next_target_q_rad': target.copy(),
                  'increment_rad': target - previous, 'offset_rad': target - self.anchor,
                  'max_increment_rad': float(np.abs(target - previous).max()),
                  'max_offset_rad': float(np.abs(target - self.anchor).max()),
                  'no_live_qpos_or_external_force_writes': True}
        self.records.append(record)
        return target, record
