"""Observation/action lesson: a small joint1 motion and empty-gripper cycling.

Headless: MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m robot_scene.demo_vla_adapter
Live: env -u MUJOCO_GL -u PYOPENGL_PLATFORM python -m robot_scene.demo_vla_adapter --viewer
No grasp, IK, policy, model weights, training or new dependencies.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import inspect
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image

from .panda_scene import CAMERA_NAME, ROOT, PandaTableScene
from .vla_adapter import ActionAdapter, IMAGE_KEY, ObservationAdapter, STATE_KEY
from .view_panda_scene import managed_passive_viewer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def image_values(image):
    # Print representative actual RGB values; the full pixel array is saved as PNG.
    coordinates = ((0, 0), (240, 320), (479, 639))
    return {
        "shape": list(image.shape), "dtype": str(image.dtype),
        "min": int(image.min()), "max": int(image.max()),
        "sample_pixels_row_col": [
            {"row_col": [row, col], "rgb": image[row, col].tolist()}
            for row, col in coordinates
        ],
    }


def check_action_mapping(adapter, scene, home_action):
    """Probe clipping with pure conversions; never apply extreme controls."""
    ranges = adapter.ctrl_ranges
    before = (scene.data.time, scene.data.qpos.copy(), scene.data.ctrl.copy())
    lower = adapter.to_ctrl(np.r_[ranges[:7, 0] - 1.0, -1.0])
    upper = adapter.to_ctrl(np.r_[ranges[:7, 1] + 1.0, 2.0])
    middle = home_action.copy()
    middle[7] = 0.5
    middle_ctrl = adapter.to_ctrl(middle)
    np.testing.assert_allclose(lower, ranges[:, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(upper, ranges[:, 1], atol=1e-12, rtol=0)
    assert middle_ctrl[7] == 127.5
    for invalid in (np.zeros(7), np.full(8, np.nan), np.full(8, np.inf)):
        try:
            adapter.to_ctrl(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Adapter accepted an invalid action")
    assert scene.data.time == before[0]
    np.testing.assert_array_equal(scene.data.qpos, before[1])
    np.testing.assert_array_equal(scene.data.ctrl, before[2])
    return {
        "passed": True, "ctrl_ranges": ranges.tolist(),
        "clipped_lower_ctrl": lower.tolist(), "clipped_upper_ctrl": upper.tolist(),
        "half_open_ctrl": middle_ctrl.tolist(), "pure_conversion": True,
    }


def run_demo(output_dir, show_viewer=False):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, stages = [], {}
    with PandaTableScene() as scene:
        observer = ObservationAdapter(scene)
        action_adapter = ActionAdapter(scene.model)
        schema = observer.schema()
        initial = observer.observe()
        home = np.r_[initial[STATE_KEY][:7].astype(np.float64), 1.0]
        joint1 = schema[STATE_KEY]["arm_elements_0_to_6"][0]
        joint_limits = joint1["joint_range"]
        ctrl_limits = action_adapter.ctrl_ranges[0]
        safe_lo = max(joint_limits[0], ctrl_limits[0]) + 0.01
        safe_hi = min(joint_limits[1], ctrl_limits[1]) - 0.01
        joint1_target = float(np.clip(home[0] + 0.1, safe_lo, safe_hi))
        if joint1_target - home[0] < 0.05:
            raise RuntimeError("Initial joint1 pose leaves too little room for this small experiment")
        mapping = check_action_mapping(action_adapter, scene, home)
        default_steps = inspect.signature(PandaTableScene.step).parameters["n_steps"].default
        assert default_steps == 10, "Review timing if core step defaults change"
        n_steps = 10
        dt = float(scene.model.opt.timestep)
        action_period = n_steps * dt
        print(json.dumps({
            "observation.state": {"shape": list(initial[STATE_KEY].shape),
                                  "dtype": str(initial[STATE_KEY].dtype),
                                  "values": initial[STATE_KEY].tolist()},
            IMAGE_KEY: image_values(initial[IMAGE_KEY]),
            "task": {"type": type(initial["task"]).__name__, "shape": None,
                     "dtype": "not applicable", "value": initial["task"]},
            "schema": schema, "action_mapping": mapping,
            "timing": {"physics_timestep_s": dt, "physics_hz": 1 / dt,
                       "step_default_n_steps": default_steps,
                       "physics_steps_per_action": n_steps,
                       "action_period_simulation_s": action_period,
                       "action_hz_simulation": 1 / action_period},
        }, indent=2), flush=True)
        Image.fromarray(initial[IMAGE_KEY]).save(output_dir / "initial_rgb.png")
        write_json(output_dir / "observation-schema.json", schema)
        write_json(output_dir / "initial-observation.json", {
            STATE_KEY: initial[STATE_KEY].tolist(), "task": initial["task"],
            IMAGE_KEY: image_values(initial[IMAGE_KEY]),
        })
        if show_viewer:
            viewer_context = managed_passive_viewer(scene.model, scene.data)
        else:
            viewer_context = nullcontext(None)
        # Only joint1 changes; all other arm targets stay at the legal home pose.
        schedule = [
            ("home_settle", 0.6, home[0], 1.0, False),
            ("joint1_ramp_out", 1.0, joint1_target, 1.0, True),
            ("joint1_hold", 1.0, joint1_target, 1.0, False),
            ("joint1_return", 1.0, home[0], 1.0, True),
            ("joint1_home_hold", 0.4, home[0], 1.0, False),
            ("gripper_open", 0.8, home[0], 1.0, False),
            ("gripper_close", 1.2, home[0], 0.0, False),
            ("gripper_reopen", 1.2, home[0], 1.0, False),
        ]
        interrupted = False
        with viewer_context as viewer, (output_dir / "trajectory.jsonl").open("w") as trace:
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = scene.model.camera(CAMERA_NAME).id
                viewer.sync()
            previous_joint_target = home[0]
            for stage, seconds, joint_target, grip_target, ramp in schedule:
                count = int(round(seconds / action_period))
                if not np.isclose(count * action_period, seconds):
                    raise ValueError("Stage duration must be a multiple of the action period")
                for index in range(count):
                    frame_start = time.monotonic()
                    if viewer is not None and not viewer.is_running():
                        interrupted = True
                        break
                    action = home.copy()
                    fraction = (index + 1) / count
                    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                    action[0] = (previous_joint_target + smooth * (joint_target - previous_joint_target)
                                 if ramp else joint_target)
                    action[7] = grip_target
                    ctrl = action_adapter.to_ctrl(action)
                    with viewer.lock() if viewer is not None else nullcontext():
                        before_time = float(scene.data.time)
                        # First call deliberately checks the core's actual default.
                        if not rows:
                            scene.step(ctrl)
                        else:
                            scene.step(ctrl, n_steps=n_steps)
                        after_time = float(scene.data.time)
                        elapsed = after_time - before_time
                        assert np.isclose(elapsed, action_period, atol=1e-10, rtol=0)
                        measured = observer.state()
                        assert measured.shape == (8,) and measured.dtype == np.float32
                        assert np.isfinite(measured).all()
                        row = {
                            "stage": stage, "action_applied_at_s": before_time,
                            "state_measured_at_s": after_time,
                            "physics_steps": int(round(elapsed / dt)),
                            "step_elapsed_simulation_s": elapsed,
                            "policy_action": action.tolist(),
                            "mujoco_ctrl": scene.data.ctrl.copy().tolist(),
                            "observation.state": measured.tolist(),
                            "finger_width_m": observer.finger_width_m(),
                        }
                    rows.append(row)
                    trace.write(json.dumps(row) + "\n")
                    if viewer is not None:
                        viewer.sync()
                        remaining = action_period - (time.monotonic() - frame_start)
                        if remaining > 0:
                            time.sleep(remaining)
                if interrupted:
                    break
                previous_joint_target = joint_target
                stages[stage] = rows[-1]
                print(json.dumps({"stage_end": stage, **rows[-1]}), flush=True)
                if stage in ("joint1_hold", "gripper_close", "gripper_reopen"):
                    Image.fromarray(scene.render_rgb()).save(output_dir / f"{stage}_rgb.png")
        with (output_dir / "trajectory.csv").open("w", newline="") as csv_file:
            fields = ["stage", "action_applied_at_s", "state_measured_at_s", "physics_steps",
                      "step_elapsed_simulation_s", "finger_width_m"]
            fields += [f"{kind}_{i}" for kind in ("state", "action", "ctrl") for i in range(8)]
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flat = {key: row[key] for key in fields[:6]}
                for kind, source in (("state", "observation.state"), ("action", "policy_action"),
                                     ("ctrl", "mujoco_ctrl")):
                    flat.update({f"{kind}_{i}": row[source][i] for i in range(8)})
                writer.writerow(flat)
        if interrupted:
            write_json(output_dir / "result.json", {"completed": False, "reason": "viewer closed",
                                                      "logged_actions": len(rows)})
            print("Viewer closed; partial experiment logs saved and resources released.", flush=True)
            return
        held = stages["joint1_hold"]
        actual_delta = held[STATE_KEY][0] - stages["home_settle"][STATE_KEY][0]
        tracking_error = abs(held[STATE_KEY][0] - joint1_target)
        checks = {
            "finite_8d_state_all_samples": all(np.isfinite(row[STATE_KEY]).all() for row in rows),
            "rgb_valid": bool(initial[IMAGE_KEY].max() > initial[IMAGE_KEY].min()),
            "action_mapping": mapping["passed"],
            "joint1_actual_delta_gt_0_05_rad": actual_delta > 0.05,
            "joint1_hold_error_lt_0_02_rad": tracking_error < 0.02,
            "joint1_return_error_lt_0_02_rad": abs(stages["joint1_home_hold"][STATE_KEY][0] - float(home[0])) < 0.02,
            "gripper_open_width_ge_0_075_m": stages["gripper_open"]["finger_width_m"] >= 0.075,
            "gripper_closed_width_le_0_005_m": abs(stages["gripper_close"]["finger_width_m"]) <= 0.005,
            "gripper_reopened_width_ge_0_075_m": stages["gripper_reopen"]["finger_width_m"] >= 0.075,
            "all_steps_advance_exactly_10_physics_steps": all(row["physics_steps"] == 10 for row in rows),
            "no_mujoco_warnings": all(warning.number == 0 for warning in scene.data.warning),
        }
        result = {
            "completed": True, "passed": all(checks.values()), "checks": checks,
            "initial_state": initial[STATE_KEY].tolist(), "task": observer.task,
            "joint1_home_target_rad": float(home[0]), "joint1_motion_target_rad": joint1_target,
            "joint1_held_actual_rad": held[STATE_KEY][0],
            "joint1_actual_delta_rad": actual_delta, "joint1_hold_error_rad": tracking_error,
            "gripper_widths_m": {name: stages[name]["finger_width_m"] for name in
                                  ("gripper_open", "gripper_close", "gripper_reopen")},
            "physics_timestep_s": dt, "physics_frequency_simulation_hz": 1 / dt,
            "step_default_n_steps": default_steps, "default_step_measured_delta_s": rows[0]["step_elapsed_simulation_s"],
            "action_period_simulation_s": action_period, "action_frequency_simulation_hz": 1 / action_period,
            "action_count": len(rows), "total_physics_steps": sum(row["physics_steps"] for row in rows),
            "total_simulation_time_s": float(scene.data.time),
            "viewer_used": show_viewer, "wallclock_pacing": "best effort" if show_viewer else "unpaced",
            "mapping_checks": mapping, "stage_endpoints": stages,
            "scripted_grasp_used": False, "learned_policy_used": False,
        }
        write_json(output_dir / "result.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "stage_endpoints"}, indent=2), flush=True)
        if not result["passed"]:
            raise RuntimeError("Basic adapter/motion experiment failed acceptance; inspect result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="show a passive viewer during this finite experiment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/adapter_basics")
    args = parser.parse_args()
    try:
        run_demo(args.output_dir, show_viewer=args.viewer)
    except KeyboardInterrupt:
        print("Interrupted; viewer/renderer resources released. Any completed trace rows remain saved.", flush=True)


if __name__ == "__main__":
    main()
