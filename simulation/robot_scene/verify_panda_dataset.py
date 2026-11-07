"""Read a local LeRobotDataset and audit every image, state, action and timestamp.

This is a read-only dataset check (apart from its own validation outputs).
It does not start a controller, download weights or change the simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .collect_panda_dataset import DATASET_ROOT, FPS, REPO_ID, TASK, VALIDATION_ROOT, write_json
from .vla_adapter import IMAGE_KEY, STATE_KEY


def tensor_rgb_to_uint8(image):
    # The installed image (not video) reader always applies PIL -> CHW float [0,1].
    assert image.shape == (3, 480, 640) and image.dtype == torch.float32
    assert torch.isfinite(image).all().item()
    assert image.min().item() >= 0 and image.max().item() <= 1
    return np.ascontiguousarray((image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def sample_output(item):
    rgb = tensor_rgb_to_uint8(item[IMAGE_KEY])
    result = {}
    for key, value in item.items():
        if key == IMAGE_KEY:
            result[key] = {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "range": [float(value.min()), float(value.max())], "color_space": "RGB",
                "uint8_hwc_shape": list(rgb.shape),
                "center_pixel_rgb_uint8": rgb[240, 320].tolist(),
            }
        elif isinstance(value, torch.Tensor):
            result[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                           "value": value.tolist()}
        else:
            result[key] = value
    return result


def verify(root=DATASET_ROOT, validation_root=VALIDATION_ROOT, report_name="verification.json"):
    torch.set_num_threads(1)  # Process-local CPU setting for small image operations.
    root = Path(root).expanduser().resolve()
    validation_root = Path(validation_root).expanduser().resolve()
    manifest = json.loads((root / "meta/collection.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    assert info["fps"] == FPS and info["codebase_version"] == "v3.0"
    dataset = LeRobotDataset(REPO_ID, root=root, video_backend="pyav", download_videos=False, token=False)
    summaries = manifest["successful_episodes"]
    assert dataset.meta.total_episodes == len(summaries)
    assert len(dataset) == sum(episode["frames"] for episode in summaries)
    checks, keyframes, action_hashes = [], {}, []
    global_index = 0
    for episode_index, summary in enumerate(summaries):
        assert summary["episode_index"] == episode_index
        proof = validation_root / f"episode-{episode_index:03d}"
        audit = [json.loads(line) for line in (proof / "pre-step-audit.jsonl").read_text().splitlines()]
        post = [json.loads(line) for line in (proof / "trajectory.jsonl").read_text().splitlines()]
        grasp = json.loads((proof / "grasp-result.json").read_text())
        assert grasp["success"] and all(grasp["checks"].values())
        assert not grasp["real_cube_qpos_overwritten"] and not grasp["weld_or_external_attachment"]
        assert len(audit) == len(post) == summary["frames"]
        assert len(audit) * 10 == grasp["total_simulation_steps"]
        limits = np.asarray(grasp["observation_schema"]["mujoco_actuator_ctrl"]["ranges"])
        phases = [row["phase"] for row in audit]
        selected = {0: "00_initial", len(audit) - 1: "04_lifted"}
        for phase, name in (("descend", "01_above_cube"), ("close", "02_at_cube"), ("lift", "03_closed")):
            selected[phases.index(phase)] = name
        times, action_rows = [], []
        image_count = 0
        for frame_index, raw in enumerate(audit):
            item = dataset[global_index]
            assert int(item["episode_index"].item()) == episode_index
            assert int(item["frame_index"].item()) == frame_index
            assert int(item["index"].item()) == global_index
            assert int(item["task_index"].item()) == 0 and item["task"] == TASK
            assert item[STATE_KEY].shape == item["action"].shape == (8,)
            assert item[STATE_KEY].dtype == item["action"].dtype == torch.float32
            assert torch.isfinite(item[STATE_KEY]).all().item() and torch.isfinite(item["action"]).all().item()
            np.testing.assert_array_equal(item[STATE_KEY].numpy(), np.asarray(raw["state"], dtype=np.float32))
            action = item["action"].numpy()
            np.testing.assert_array_equal(action, np.asarray(raw["action"], dtype=np.float32))
            np.testing.assert_array_equal(action, np.asarray(post[frame_index]["policy_action"], dtype=np.float32))
            expected_ctrl = np.r_[np.clip(action[:7].astype(np.float64), limits[:7, 0], limits[:7, 1]),
                                     float(np.clip(action[7], 0, 1)) * 255]
            np.testing.assert_allclose(raw["ctrl"], expected_ctrl, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(raw["ctrl"], post[frame_index]["mujoco_actuator_ctrl"])
            # Independent post-step trajectory proves this is pre-action state.
            if frame_index > 0:
                np.testing.assert_array_equal(np.asarray(raw["state"], dtype=np.float32),
                                              np.asarray(post[frame_index - 1]["robot_state"], dtype=np.float32))
            time_s = float(item["timestamp"].item())
            assert abs(time_s - frame_index / FPS) < 1e-4
            assert abs(raw["simulation_time_s"] - frame_index / FPS) < 1e-9
            assert abs(float(item["sim.time"].item()) - frame_index / FPS) < 1e-4
            assert abs(post[frame_index]["time_s"] - (frame_index + 1) / FPS) < 1e-9
            assert int(item["sim.physics_step"].item()) == raw["physics_step"] == frame_index * 10
            assert int(item["sim.control_tick"].item()) == frame_index
            assert 0 <= float(item[STATE_KEY][7]) <= 1 and 0 <= float(action[7]) <= 1
            rgb = tensor_rgb_to_uint8(item[IMAGE_KEY])
            # Exact raw RGB hash proves no lossy encoding, RGB/BGR swap or time shift.
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == raw["rgb_sha256"]
            image_count += 1
            if episode_index == 0 and frame_index in selected:
                keyframes[selected[frame_index]] = (rgb.copy(), time_s, raw["phase"], raw["cube_position_m"][2])
            times.append(time_s)
            action_rows.append(action.copy())
            global_index += 1
        timestamps = np.asarray(times)
        assert np.all(np.diff(timestamps) > 0)
        max_period_error = float(np.max(np.abs(np.diff(timestamps) - 1 / FPS)))
        assert max_period_error < 1e-4
        action_hash = hashlib.sha256(np.asarray(action_rows, dtype=np.float32).tobytes()).hexdigest()
        action_hashes.append(action_hash)
        episode_result = {
            "episode_index": episode_index, "frames": len(audit), "fps": FPS,
            "first_timestamp_s": times[0], "last_timestamp_s": times[-1],
            "physics_duration_s": grasp["final_time_s"],
            "maximum_timestamp_period_error_s": max_period_error,
            "all_rgb_frames_lossless_and_aligned": image_count,
            "all_state_action_rows_aligned": len(audit),
            "grasp_success": grasp["success"], "actual_cube_lift_m": grasp["actual_cube_lift_m"],
            "hold_metrics": grasp["hold_metrics"], "action_sequence_sha256": action_hash,
        }
        checks.append(episode_result)
        print("VERIFIED=" + json.dumps(episode_result), flush=True)
    replay = validation_root / "replay_episode_000"
    replay.mkdir(exist_ok=True)
    montage = Image.new("RGB", (960, 540), "#17202a")
    draw = ImageDraw.Draw(montage)
    for index, (name, (rgb, time_s, phase, cube_z)) in enumerate(sorted(keyframes.items())):
        image = Image.fromarray(rgb)
        image.save(replay / f"{name}.png")
        x, y = index % 3 * 320, index // 3 * 270
        montage.paste(image.resize((320, 240)), (x, y + 30))
        draw.text((x + 5, y + 5), f"{name} t={time_s:.2f}s cube_z={cube_z:.3f}m", fill="white")
    montage.save(replay / "keyframes.png")
    sample = sample_output(dataset[0])
    final_sample = sample_output(dataset[summaries[0]["frames"] - 1])
    write_json(validation_root / "sample-first-frame.json", sample)
    write_json(validation_root / "sample-lifted-frame.json", final_sample)
    files = [path for path in root.rglob("*") if path.is_file()]
    report = {
        "passed": True, "dataset_root": str(root), "format": info["codebase_version"],
        "episode_count": len(checks), "frame_count": len(dataset), "fps": FPS,
        "dataset_file_bytes": sum(path.stat().st_size for path in files), "dataset_file_count": len(files),
        "features": info["features"], "episodes": checks,
        "unique_action_sequences": len(set(action_hashes)),
        "deterministic_repeats_not_generalization_data": True,
        "sample": sample, "lifted_sample": final_sample,
        "image_read_contract": "HWC uint8 RGB at capture; CHW float32 [0,1] from installed image reader",
        "all_frames_pre_step_state_action_alignment_verified": True,
        "all_frame_image_hashes_verified": True,
        "keyframe_directory": str(replay),
    }
    write_json(validation_root / report_name, report)
    print("SAMPLE=" + json.dumps(sample), flush=True)
    print("PASS=" + json.dumps({key: report[key] for key in
                                ("episode_count", "frame_count", "fps", "dataset_file_bytes", "unique_action_sequences")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--validation-root", type=Path, default=VALIDATION_ROOT)
    parser.add_argument("--report-name", default="verification.json")
    args = parser.parse_args()
    verify(args.root, args.validation_root, args.report_name)


if __name__ == "__main__":
    main()
