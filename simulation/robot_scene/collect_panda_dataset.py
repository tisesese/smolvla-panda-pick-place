"""Collect local LeRobot v3 episodes using the existing successful controller.

Every row is (observation BEFORE step, exact action ABOUT TO BE applied).
50 Hz recording matches 50 Hz control; no frame skipping or action aggregation.
500 Hz physics advances ten steps after every recorded row. No Hub uploads.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .panda_scene import ARM_JOINTS, ROOT
from .scripted_grasp import run, serializable
from .vla_adapter import IMAGE_KEY, ObservationAdapter, STATE_KEY

REPO_ID = "local/panda_scripted_grasp"
TASK = "Pick up the red cube."
FPS = 50
DATASET_ROOT = ROOT / "outputs/lerobot_panda_dataset"
VALIDATION_ROOT = ROOT / "outputs/lerobot_panda_dataset_validation"


def features():
    return {
        STATE_KEY: {"dtype": "float32", "shape": (8,),
                    "names": list(ARM_JOINTS) + ["gripper_opening_normalized"]},
        IMAGE_KEY: {"dtype": "image", "shape": (3, 480, 640),
                    "names": ["channels", "height", "width"]},
        "action": {"dtype": "float32", "shape": (8,),
                   "names": [f"{name}_target" for name in ARM_JOINTS] + ["gripper_command_normalized"]},
        "sim.time": {"dtype": "float64", "shape": (1,), "names": None},
        "sim.physics_step": {"dtype": "int64", "shape": (1,), "names": None},
        "sim.control_tick": {"dtype": "int64", "shape": (1,), "names": None},
    }


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2) + "\n")


class EpisodeRecorder:
    """A passive pre-step recorder; never writes qpos, controls or attachments."""

    def __init__(self, dataset, audit_file, episode_index, progress_stream):
        self.dataset = dataset
        self.audit_file = audit_file
        self.episode_index = episode_index
        self.progress_stream = progress_stream
        self.observer = None
        self.count = 0

    def __call__(self, scene, interface, phase, action, ctrl, physics_steps):
        if self.observer is None:
            self.observer = ObservationAdapter(scene, task=TASK)
        assert np.isclose(scene.model.opt.timestep, 0.002, rtol=0, atol=1e-15)
        assert physics_steps == self.count * 10
        assert np.isclose(scene.data.time, self.count / FPS, rtol=0, atol=1e-9)
        observation = self.observer.observe()
        state = observation[STATE_KEY]
        image = observation[IMAGE_KEY]
        policy_action = np.asarray(action, dtype=np.float32).copy()
        # The controller dispatched float32 values, rather than rounding only the label.
        np.testing.assert_array_equal(policy_action.astype(np.float64), action)
        np.testing.assert_array_equal(state, interface.robot_state())
        np.testing.assert_array_equal(ctrl[:7], policy_action[:7].astype(np.float64))
        assert np.isclose(ctrl[7], float(policy_action[7]) * 255, rtol=0, atol=1e-12)
        self.dataset.add_frame({
            STATE_KEY: state.copy(), IMAGE_KEY: image,
            "action": policy_action, "task": TASK,
            "sim.time": np.array([scene.data.time], dtype=np.float64),
            "sim.physics_step": np.array([physics_steps], dtype=np.int64),
            "sim.control_tick": np.array([self.count], dtype=np.int64),
        })
        audit = {
            "episode_index": self.episode_index, "frame_index": self.count,
            "phase": phase, "simulation_time_s": float(scene.data.time),
            "physics_step": physics_steps, "state": state, "action": policy_action,
            "ctrl": ctrl, "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "cube_position_m": scene.get_world_poses()["cube"]["position_m"],
            "gripper_width_m": self.observer.finger_width_m(),
        }
        self.audit_file.write(json.dumps(serializable(audit)) + "\n")
        self.count += 1
        if self.count % 100 == 0:
            print(f"episode={self.episode_index} recorded={self.count} phase={phase}",
                  file=self.progress_stream, flush=True)


def collect(root=DATASET_ROOT, validation_root=VALIDATION_ROOT, episodes=1, resume=False):
    if metadata.version("lerobot") != "0.6.2":
        raise RuntimeError("This collector was verified against the installed LeRobot 0.6.2")
    root = Path(root).expanduser().resolve()
    validation_root = Path(validation_root).expanduser().resolve()
    validation_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "meta/collection.json"
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["repo_id"] != REPO_ID or manifest["dataset_fps"] != FPS:
            raise ValueError("Refusing to append to an unrelated dataset")
        dataset = LeRobotDataset.resume(REPO_ID, root=root, video_backend="pyav",
                                        image_writer_threads=2, token=False)
    else:
        if root.exists():
            raise FileExistsError(f"Refusing to overwrite {root}; use --resume to append")
        dataset = LeRobotDataset.create(
            REPO_ID, fps=FPS, features=features(), root=root,
            robot_type="franka_panda_mujoco", use_videos=False,
            video_backend="pyav", image_writer_threads=2, metadata_buffer_size=1,
        )
        manifest = {
            "repo_id": REPO_ID, "task": TASK,
            "physics_frequency_hz": 500, "control_frequency_hz": 50,
            "dataset_fps": FPS, "physics_steps_per_record": 10,
            "row_semantics": "state/RGB at t, action applied immediately after observation, then 10 physical steps",
            "timestamp_semantics": "episode-relative frame_index/fps; matches actual pre-step simulation time",
            "downsampling": False, "terminal_observation_without_action": False,
            "controller": "robot_scene.scripted_grasp.run; native MuJoCo Jacobian IK reused",
            "policy_action_dtype_dispatched": "float32",
            "state_units": ["rad"] * 7 + ["dimensionless measured aperture / 0.08m"],
            "action_units": ["rad absolute target"] * 7 + ["dimensionless commanded opening"],
            "raw_rgb": "HWC uint8 RGB, lossless PNG embedded in Parquet",
            "reader_rgb": "installed image reader returns CHW float32 RGB in [0,1]",
            "reset": "normal scene.reset between episodes; no in-episode cube qpos edits or welds",
            "purpose": "small deterministic pipeline validation, not a generalization training dataset",
            "versions": {name: metadata.version(name) for name in
                         ("lerobot", "torch", "torchvision", "transformers", "mujoco", "datasets", "pyarrow")},
            "successful_episodes": [],
        }
        write_json(manifest_path, manifest)
    print(f"COLLECT root={root} existing_episodes={dataset.meta.total_episodes} fps={FPS}", flush=True)
    progress_stream = sys.stdout
    try:
        for _ in range(episodes):
            episode_index = dataset.meta.total_episodes
            proof = validation_root / f"episode-{episode_index:03d}"
            proof.mkdir(exist_ok=False)
            try:
                with (proof / "pre-step-audit.jsonl").open("w") as audit, (proof / "controller.log").open("w") as log:
                    recorder = EpisodeRecorder(dataset, audit, episode_index, progress_stream)
                    with redirect_stdout(log):
                        result = run(output_dir=proof, before_step=recorder,
                                     action_dtype=np.float32, rebuild_scene=False)
                if not result["success"]:
                    raise RuntimeError("Unsuccessful grasp: episode will not be committed")
                assert recorder.count * 10 == result["total_simulation_steps"]
                dataset.save_episode(parallel_encoding=False)
                episode_summary = {
                    "episode_index": episode_index, "frames": recorder.count,
                    "success": result["success"], "actual_cube_lift_m": result["actual_cube_lift_m"],
                    "hold_metrics": result["hold_metrics"], "proof_directory": str(proof),
                }
                manifest["successful_episodes"].append(episode_summary)
                write_json(manifest_path, manifest)
                print("SAVED=" + json.dumps(episode_summary), flush=True)
            except Exception as exc:
                with (validation_root / "failed-episodes.jsonl").open("a") as failure_log:
                    failure_log.write(json.dumps({"episode_index": episode_index,
                                                  "error_type": type(exc).__name__, "error": str(exc)}) + "\n")
                raise
    finally:
        # This installed API requires finalize to flush Parquet footers and metadata.
        dataset.finalize()
    print(f"FINALIZED episodes={dataset.meta.total_episodes} frames={dataset.meta.total_frames}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--validation-root", type=Path, default=VALIDATION_ROOT)
    parser.add_argument("--episodes", type=int, default=1, help="number of NEW episodes to collect")
    parser.add_argument("--resume", action="store_true", help="append without overwriting prior episodes")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    collect(args.root, args.validation_root, args.episodes, args.resume)


if __name__ == "__main__":
    main()
