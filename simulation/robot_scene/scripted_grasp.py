"""Actuator-driven grasp using native MuJoCo Jacobians and NumPy only."""
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .panda_scene import GRASP_SITE_NAME, PandaTableScene, ROOT, build_scene
from .vla_interface import VLAInterface

OUT = ROOT / "outputs" / "scripted_grasp"
THRESHOLDS = {
    "minimum_lift_from_initial_m": 0.10,
    "minimum_cube_bottom_clearance_m": 0.08,
    "maximum_cube_to_grasp_center_distance_m": 0.04,
    "minimum_hold_steps": 1000,
    "maximum_height_drift_during_hold_m": 0.02,
    "minimum_bilateral_contact_fraction": 0.95,
    "minimum_finger_normal_force_n": 0.05,
    "cube_table_contact_allowed": False,
}


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


class NativeJacobianIK:
    """6D damped least-squares IK on independent scratch MjData.

    Kinematic trial qpos are never written into the live scene. Position error
    and rotational Jacobian are both in world coordinates. mju_subQuat returns
    a local tangent error, so it is rotated to world before solving.
    """

    def __init__(self, scene):
        self.model = scene.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = scene.data.qpos
        self.qpos_indices = scene.arm_qpos.copy()
        self.dof_indices = scene.arm_dofs.copy()
        self.site_id = self.model.site(GRASP_SITE_NAME).id
        self.limits = self.model.actuator_ctrlrange[:7].copy()
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.rotation_weight = 0.20
        self.damping = 0.01

    def error(self, q, position, quaternion):
        self.data.qpos[self.qpos_indices] = q
        self.data.qvel[:] = 0
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        site = self.data.site(self.site_id)
        current_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(current_quaternion, site.xmat)
        local_error = np.empty(3)
        mujoco.mju_subQuat(local_error, quaternion, current_quaternion)
        world_error = site.xmat.reshape(3, 3) @ local_error
        pos_error = position - site.xpos
        weighted = np.r_[pos_error, self.rotation_weight * world_error]
        return weighted, float(np.linalg.norm(pos_error)), float(np.linalg.norm(world_error))

    def solve(self, position, quaternion, seed):
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        q = np.clip(np.asarray(seed, dtype=np.float64), self.limits[:, 0] + 0.01,
                    self.limits[:, 1] - 0.01).copy()
        for iteration in range(200):
            error, pos_error, rot_error = self.error(q, position, quaternion)
            if pos_error < 0.0002 and rot_error < 0.002:
                return q, {"iterations": iteration, "position_error_m": pos_error,
                           "orientation_error_rad": rot_error}
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
            jac = np.vstack([self.jacp[:, self.dof_indices],
                             self.rotation_weight * self.jacr[:, self.dof_indices]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping**2 * np.eye(6), error)
            norm = np.linalg.norm(dq)
            if norm > 0.10:
                dq *= 0.10 / norm
            accepted = False
            for scale in [1.0, 0.5, 0.25, 0.1]:
                candidate = np.clip(q + scale * dq, self.limits[:, 0] + 0.01,
                                    self.limits[:, 1] - 0.01)
                candidate_error, _, _ = self.error(candidate, position, quaternion)
                if np.linalg.norm(candidate_error) < np.linalg.norm(error):
                    q = candidate
                    accepted = True
                    break
            if not accepted:
                break
        _, pos_error, rot_error = self.error(q, position, quaternion)
        raise RuntimeError(f"IK did not converge: position={pos_error:.6f} m, "
                           f"orientation={rot_error:.6f} rad")


def contact_metrics(scene):
    model, data = scene.model, scene.data
    cube_body = model.body("red_cube").id
    finger_ids = {model.body("left_finger").id: "left", model.body("right_finger").id: "right"}
    forces = {"left": 0.0, "right": 0.0}
    table_contact = False
    contacts = []
    for i in range(data.ncon):
        con = data.contact[i]
        body1 = int(model.geom_bodyid[con.geom1])
        body2 = int(model.geom_bodyid[con.geom2])
        if cube_body not in (body1, body2):
            continue
        other_body = body2 if body1 == cube_body else body1
        other_geom = int(con.geom2 if body1 == cube_body else con.geom1)
        wrench = np.empty(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        if other_body in finger_ids:
            side = finger_ids[other_body]
            forces[side] += max(0.0, float(wrench[0]))
            contacts.append({"finger": side, "normal_force_n": float(wrench[0])})
        if model.geom(other_geom).name in ["table_top", "floor"]:
            table_contact = True
    return {"finger_normal_force_n": forces, "cube_table_contact": table_contact,
            "finger_cube_contacts": contacts}


def run(output_dir=OUT, *, before_step=None, action_dtype=np.float64, rebuild_scene=True,
        reset_initializer=None):
    """Run the same grasp; optionally observe each command BEFORE physics.

    before_step(scene, interface, phase, action, ctrl, physics_steps) is a
    recording hook, not a controller. It receives copies of the exact command.
    Defaults preserve the original CLI, output path and float64 controller.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rebuild_scene:
        build_scene()
    snapshots = {}
    ik_metrics = []
    samples = []
    with PandaTableScene() as scene, (out / "trajectory.jsonl").open("w") as trajectory:
        interface = VLAInterface(scene)
        adapter = interface.action_adapter
        ik = NativeJacobianIK(scene)
        # The nominal home orientation is already vertical, with fingers along world x.
        orientation = scene.get_world_poses()["grasp_center"]["quaternion_wxyz"].copy()
        action = np.r_[scene.model.key_ctrl[scene.home_key, :7], 1.0]
        total_steps = 0
        max_qvel = 0.0
        max_qacc = 0.0

        def tick(phase, policy_action):
            nonlocal total_steps, max_qvel, max_qacc, action
            # Dataset recording can explicitly dispatch float32 policy actions.
            # Convert back to float64 only for MuJoCo; the numeric values match.
            action = np.asarray(policy_action, dtype=action_dtype).astype(np.float64)
            ctrl = adapter.to_ctrl(action)
            if before_step is not None:
                before_step(scene, interface, phase, action.copy(), ctrl.copy(), total_steps)
            scene.step(ctrl, n_steps=10)
            total_steps += 10
            max_qvel = max(max_qvel, float(np.max(np.abs(scene.data.qvel))))
            max_qacc = max(max_qacc, float(np.max(np.abs(scene.data.qacc))))
            if max_qvel > 30 or not np.isfinite(scene.data.qacc).all():
                raise RuntimeError("Abnormal dynamics during scripted grasp")
            poses = scene.get_world_poses()
            record = {"phase": phase, "time_s": float(scene.data.time),
                      "robot_state": interface.robot_state(), "policy_action": action.copy(),
                      "mujoco_actuator_ctrl": ctrl.copy(), "poses": poses,
                      "contact": contact_metrics(scene)}
            trajectory.write(json.dumps(serializable(record)) + "\n")
            samples.append(serializable(record))
            return record

        def capture(name):
            observation = interface.observe()
            png = out / f"{name}_rgb.png"
            Image.fromarray(observation["observation.images.fixed_rgb"]).save(png)
            snapshots[name] = serializable({"image_path": str(png), "time_s": scene.data.time,
                "robot_state": observation["observation.state"], "policy_action": action.copy(),
                "mujoco_actuator_ctrl": scene.data.ctrl.copy(), "poses": scene.get_world_poses(),
                "contact": contact_metrics(scene)})
            print(name.upper() + "=" + json.dumps(snapshots[name]), flush=True)

        def hold(phase, ticks):
            for _ in range(ticks):
                tick(phase, action)

        def move_to(phase, position, seconds):
            start = scene.get_world_poses()["grasp_center"]["position_m"].copy()
            position = np.asarray(position, dtype=np.float64)
            seed = action[:7].copy()
            steps = int(round(seconds / 0.02))
            for t in range(1, steps + 1):
                alpha = t / steps
                blend = 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5
                target = start + blend * (position - start)
                seed, metric = ik.solve(target, orientation, seed)
                policy_action = np.r_[seed, action[7]]
                tick(phase, policy_action)
                metric.update({"phase": phase, "target_position_m": target.tolist()})
                ik_metrics.append(metric)
            hold(phase + "_settle", 30)

        scene.reset()
        # Optional initialization is invoked once at t=0, before any controller step.
        # Existing callers keep their original reset and action semantics.
        reset_metadata = reset_initializer(scene) if reset_initializer is not None else None
        if not np.isclose(scene.data.time, 0.0, rtol=0, atol=1e-12):
            raise ValueError("reset_initializer must not advance simulation time")
        hold("home_settle", 50)
        capture("01_initial")
        cube = scene.get_world_poses()["cube"]["position_m"].copy()
        initial_cube_z = float(cube[2])
        table_top_z = float(scene.data.geom("table_top").xpos[2] +
                            scene.model.geom("table_top").size[2])
        cube_half_size = scene.model.geom("red_cube_geom").size.copy()
        move_to("approach", cube + [0, 0, 0.16], 3.0)
        capture("02_approach")
        # Centre pads at the cube centre; the fingertips stay above the table.
        cube = scene.get_world_poses()["cube"]["position_m"].copy()
        move_to("descend", cube, 2.5)
        capture("03_descend")
        close_start = action.copy()
        for alpha in np.linspace(0.0, 1.0, 75):
            close_action = close_start.copy()
            close_action[7] = 1.0 - alpha
            tick("close", close_action)
        hold("close_settle", 50)
        capture("04_closed")
        grasp = scene.get_world_poses()["grasp_center"]["position_m"].copy()
        move_to("lift", grasp + [0, 0, 0.16], 3.0)
        hold_start = len(samples)
        hold("lift_hold", 100)
        capture("05_lift")

        window = samples[hold_start:]
        heights = np.array([x["poses"]["cube"]["position_m"][2] for x in window])
        distances = np.array([np.linalg.norm(
            np.array(x["poses"]["cube"]["position_m"]) -
            np.array(x["poses"]["grasp_center"]["position_m"])) for x in window])
        bottom_clearances = []
        for sample in window:
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, np.asarray(
                sample["poses"]["cube"]["quaternion_wxyz"], dtype=np.float64))
            vertical_extent = float(np.abs(rotation.reshape(3, 3)[2]) @ cube_half_size)
            bottom_clearances.append(sample["poses"]["cube"]["position_m"][2] -
                                     vertical_extent - table_top_z)
        bottom_clearances = np.asarray(bottom_clearances)
        bilateral = np.array([all(x["contact"]["finger_normal_force_n"][side] > 0.05
                                  for side in ["left", "right"]) for x in window])
        table_contacts = [x["contact"]["cube_table_contact"] for x in window]
        checks = {
            "cube_lift_height": float(heights.min() - initial_cube_z) >= 0.10,
            "cube_bottom_clearance": float(bottom_clearances.min()) >= 0.08,
            "cube_gripper_proximity": float(distances.max()) <= 0.04,
            "sustained_hold": len(window) * 10 >= 1000,
            "no_immediate_drop": float(np.ptp(heights)) <= 0.02,
            "bilateral_finger_contact": float(bilateral.mean()) >= 0.95,
            "no_cube_table_contact": not any(table_contacts),
            "no_mujoco_warning": not any(w.number for w in scene.data.warning),
        }
        result = {
            "implementation": "native MuJoCo mj_jacSite + NumPy 6D damped least-squares IK; actuator-only motion",
            "observation_schema": interface.schema(), "thresholds": THRESHOLDS,
            "table_top_z_m": table_top_z, "initial_cube_z_m": initial_cube_z,
            "final_cube_z_m": float(heights[-1]), "actual_cube_lift_m": float(heights[-1] - initial_cube_z),
            "hold_metrics": {"simulation_steps": len(window) * 10, "duration_s": len(window) * 0.02,
                "minimum_lift_m": float(heights.min() - initial_cube_z),
                "minimum_bottom_clearance_m": float(bottom_clearances.min()),
                "bottom_clearance_includes_cube_orientation": True,
                "sample_period_s": 0.02, "hold_samples": len(window),
                "maximum_cube_grasp_distance_m": float(distances.max()),
                "cube_height_drift_m": float(np.ptp(heights)),
                "bilateral_contact_fraction": float(bilateral.mean()),
                "cube_table_contact_count": int(sum(table_contacts))},
            "snapshots": snapshots, "total_simulation_steps": total_steps,
            "final_time_s": float(scene.data.time), "max_abs_qvel": max_qvel,
            "max_abs_qacc": max_qacc, "ik_solve_count": len(ik_metrics),
            "maximum_kinematic_ik_position_error_m": max(x["position_error_m"] for x in ik_metrics),
            "maximum_kinematic_ik_orientation_error_rad": max(x["orientation_error_rad"] for x in ik_metrics),
            "reset_initialization": serializable(reset_metadata),
            "cube_qpos_initialized_at_reset": reset_initializer is not None,
            "cube_qpos_modified_during_execution": False,
            "real_cube_qpos_overwritten": False, "weld_or_external_attachment": False,
            "new_robotics_dependencies": [], "checks": checks, "success": all(checks.values()),
        }
        (out / "grasp-result.json").write_text(json.dumps(serializable(result), indent=2) + "\n")
        (out / "ik-log.json").write_text(json.dumps(ik_metrics, indent=2) + "\n")
        print("GRASP_RESULT=" + json.dumps({k: result[k] for k in
            ["initial_cube_z_m", "final_cube_z_m", "actual_cube_lift_m", "hold_metrics", "checks", "success"]}), flush=True)
        if not result["success"]:
            raise AssertionError("Scripted grasp did not satisfy numerical criteria")
        return result


if __name__ == "__main__":
    run()
