"""Live WSLg viewer for the existing Panda scene, without an offscreen Renderer.

Run from the repository simulation/ directory with the existing .venv activated:
    env -u MUJOCO_GL -u PYOPENGL_PLATFORM python -m robot_scene.view_panda_scene

The environment overrides apply only to this process. Leave DISPLAY and
WAYLAND_DISPLAY provided by WSLg intact. Use --duration 5 for a short smoke run;
otherwise simulation continues until the viewer window is closed or Ctrl+C.
"""

import argparse
from contextlib import contextmanager
import math
import threading
import time

import mujoco

if __package__:
    from .panda_scene import CAMERA_NAME, PandaTableScene
else:
    from panda_scene import CAMERA_NAME, PandaTableScene


@contextmanager
def managed_passive_viewer(model, data):
    """Close the viewer and wait for its Python display thread before exit.

    In this standalone GUI process, launch_passive starts daemon threads.
    Handle.close only requests exit; interpreter/GLFW teardown must wait until
    those threads have finished. No MuJoCo private APIs are accessed here.
    """
    import mujoco.viewer

    existing_threads = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer_threads = set(threading.enumerate()) - existing_threads
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in viewer_threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("MuJoCo viewer thread did not finish closing")


def run_viewer(duration=None):
    """Reuse one scene's model/data and advance its physics near real time."""
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be a finite, positive number of seconds")

    steps_per_frame = 10  # 50 Hz sync with the scene's 0.002-second physics step.
    total_steps = 0
    with PandaTableScene() as scene:
        with managed_passive_viewer(scene.model, scene.data) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = scene.model.camera(CAMERA_NAME).id
            viewer.sync()
            print(
                f"Panda viewer ready: camera={CAMERA_NAME}; "
                f"nq={scene.model.nq}, nv={scene.model.nv}, nu={scene.model.nu}. "
                "Close the window or press Ctrl+C to exit.",
                flush=True,
            )
            start = time.monotonic()
            while viewer.is_running():
                frame_start = time.monotonic()
                if duration is not None and frame_start - start >= duration:
                    break
                with viewer.lock():
                    # Retain home controls initially; later accept viewer sliders.
                    # Use the core step's range and numerical-stability checks.
                    scene.step(scene.data.ctrl.copy(), n_steps=steps_per_frame)
                    frame_seconds = steps_per_frame * scene.model.opt.timestep
                total_steps += steps_per_frame
                # sync also transfers GUI control and mouse perturbation inputs.
                viewer.sync()
                remaining = frame_seconds - (time.monotonic() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
            simulation_time = float(scene.data.time)
    print(
        f"Viewer closed; resources released. Physics steps={total_steps}, "
        f"simulation time={simulation_time:.3f} s.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=None, metavar="SECONDS",
        help="close automatically after this many wall-clock seconds (default: unlimited)",
    )
    args = parser.parse_args()
    if args.duration is not None and (
        not math.isfinite(args.duration) or args.duration <= 0
    ):
        parser.error("--duration must be a finite, positive number")
    try:
        run_viewer(args.duration)
    except KeyboardInterrupt:
        print("Viewer interrupted; resources released.", flush=True)


if __name__ == "__main__":
    main()
