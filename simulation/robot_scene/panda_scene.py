"""Native MuJoCo Panda tabletop scene; no Gymnasium or policy dependency."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_COMMIT = "f054586a8e90465d49ee5be15335c4a0c7f57caf"
VENDOR = ROOT / "vendor" / "mujoco_menagerie" / MODEL_COMMIT / "franka_emika_panda"
SCENE_PATH = ROOT / "models" / "panda_table_scene.xml"
CAMERA_NAME = "fixed_rgb"
GRASP_SITE_NAME = "grasp_center"
GRASP_CENTER_LOCAL = np.array([0.0, 0.0, 0.1029])
CUBE_POSITION = np.array([0.13, -0.19, 0.782])
TARGET_POSITION = np.array([0.32, 0.20, 0.775])
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")


def build_scene(path=SCENE_PATH):
    """Compose official MJCF assets, retaining its inertias and controllers."""
    spec = mujoco.MjSpec.from_file(str(VENDOR / "panda.xml"))
    spec.modelname = "task4_panda_tabletop"
    spec.meshdir = str(VENDOR / "assets")
    spec.body("link0").pos = [-0.40, 0.0, 0.75]
    # Main fingertip pad centres: finger base z=0.0584 + pad z=0.0445 m.
    # This reference site changes no mass, contacts, joints or actuator parameters.
    spec.body("hand").add_site(
        name=GRASP_SITE_NAME, pos=GRASP_CENTER_LOCAL,
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0],
        rgba=[0.0, 1.0, 0.0, 0.0])
    spec.option.timestep = 0.002
    spec.option.gravity[:] = [0.0, 0.0, -9.81]
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    spec.visual.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient[:] = [0.25, 0.25, 0.25]
    spec.visual.quality.shadowsize = 1024

    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[3, 3, 0.1], rgba=[0.13, 0.17, 0.22, 1])
    world.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0, 0, 0.70], size=[0.72, 0.50, 0.05],
                   rgba=[0.60, 0.43, 0.27, 1], friction=[1, 0.005, 0.0001])
    for i, (x, y) in enumerate([(-0.61, -0.39), (-0.61, 0.39),
                                (0.61, -0.39), (0.61, 0.39)]):
        world.add_geom(name=f"table_leg_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[x, y, 0.34], size=[0.035, 0.035, 0.34],
                       rgba=[0.20, 0.23, 0.27, 1])

    cube = world.add_body(name="red_cube", pos=CUBE_POSITION)
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="red_cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.025, 0.025, 0.025], mass=0.06,
                  rgba=[0.95, 0.08, 0.04, 1], friction=[1, 0.005, 0.0001])

    tray = world.add_body(name="target_tray", pos=[0.32, 0.20, 0.7575])
    tray_color = [0.07, 0.36, 0.70, 1]
    tray.add_geom(name="tray_base", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.12, 0.10, 0.0075], rgba=tray_color)
    for name, pos, size in [
        ("tray_left", [-0.12, 0, 0.0275], [0.007, 0.107, 0.0275]),
        ("tray_right", [0.12, 0, 0.0275], [0.007, 0.107, 0.0275]),
        ("tray_front", [0, -0.10, 0.0275], [0.12, 0.007, 0.0275]),
        ("tray_back", [0, 0.10, 0.0275], [0.12, 0.007, 0.0275]),
    ]:
        tray.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=pos, size=size, rgba=tray_color)
    world.add_site(name="target_center", pos=TARGET_POSITION,
                   type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.003, 0, 0],
                   rgba=[0.1, 0.8, 0.3, 0])

    camera_position = np.array([1.35, -1.75, 2.15])
    camera_target = np.array([-0.04, 0.0, 1.04])
    forward = camera_target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    world.add_camera(name=CAMERA_NAME, pos=camera_position,
                     xyaxes=np.concatenate([right, up]), fovy=45,
                     mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    world.add_light(name="scene_key_light", pos=[0, -0.7, 2.5], dir=[0, 0, -1],
                    diffuse=[0.8, 0.8, 0.8], ambient=[0.15, 0.15, 0.15])
    world.add_light(name="scene_fill_light", pos=[-1, 1, 2.3], dir=[0.4, -0.4, -1],
                    diffuse=[0.45, 0.45, 0.45])

    # Extend the official 9-coordinate home keyframe for the cube's free joint.
    home = spec.key("home")
    home.qpos = list(home.qpos) + list(CUBE_POSITION) + [1, 0, 0, 0]
    spec.compile()
    # Store relative mesh paths so the project remains movable as one directory.
    xml = spec.to_xml()
    xml_root = ET.fromstring(xml)
    xml_root.find("compiler").set(
        "meshdir", str(Path("../vendor/mujoco_menagerie") / MODEL_COMMIT /
                       "franka_emika_panda/assets"))
    xml = ET.tostring(xml_root, encoding="unicode")
    notice = ("<!-- Task 4 scene generated from Google DeepMind Menagerie Panda, "
              f"commit {MODEL_COMMIT}, Apache-2.0. "
              "Added base placement, table, floor, free cube, tray, fixed camera, "
              "lights, grasp reference site and extended home keyframe. "
              "Original vendor files unmodified. -->\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(notice + xml)
    # Validate the serialized file independently, including all mesh paths.
    mujoco.MjModel.from_xml_path(str(path))
    return path


class PandaTableScene:
    """Explicit 8D actuator targets, state access and a fixed-camera renderer.

    control[:7] are absolute joint position targets in radians.
    control[7] is the upstream gripper target: 0 closed, 255 open.
    The fingertip slide coordinates are coupled; aperture is their sum in metres.
    """

    def __init__(self, scene_path=SCENE_PATH):
        self.scene_path = Path(scene_path)
        if not self.scene_path.exists():
            build_scene(self.scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.arm_qpos = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.arm_dofs = np.array([self.model.joint(n).dofadr[0] for n in ARM_JOINTS])
        self.finger_qpos = np.array([self.model.joint(n).qposadr[0] for n in FINGER_JOINTS])
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        self.home_key = self.model.key("home").id
        self._renderer = None
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def step(self, control, n_steps=10):
        control = np.asarray(control, dtype=np.float64)
        if control.shape != (8,) or not np.isfinite(control).all():
            raise ValueError("control must be a finite 8D actuator target vector")
        if (np.any(control < self.ctrl_range[:, 0] - 1e-10) or
                np.any(control > self.ctrl_range[:, 1] + 1e-10)):
            raise ValueError("control is outside the model actuator control ranges")
        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        # Snap only floating-point roundoff at the declared boundaries.
        self.data.ctrl[:] = np.clip(control, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        previous_time = float(self.data.time)
        warning_counts = np.array([w.number for w in self.data.warning])
        mujoco.mj_step(self.model, self.data, nstep=int(n_steps))
        if not all(np.isfinite(a).all() for a in
                   [self.data.qpos, self.data.qvel, self.data.qacc]):
            raise RuntimeError("MuJoCo simulation produced a nonfinite state")
        # Synchronize position-derived observations with the new qpos.
        mujoco.mj_forward(self.model, self.data)
        if not np.isclose(self.data.time, previous_time + n_steps * self.model.opt.timestep,
                          rtol=0, atol=1e-9):
            raise RuntimeError("Unexpected simulation time; possible automatic reset")
        if np.any(np.array([w.number for w in self.data.warning]) > warning_counts):
            raise RuntimeError("MuJoCo reported a new simulation warning")
        return self.observe()

    def observe(self, include_rgb=False):
        obs = {
            "joint_positions": self.data.qpos[self.arm_qpos].copy(),
            "joint_velocities": self.data.qvel[self.arm_dofs].copy(),
            "gripper_width": float(self.data.qpos[self.finger_qpos].sum()),
            "cube_position": self.data.body("red_cube").xpos.copy(),
            "cube_quaternion": self.data.body("red_cube").xquat.copy(),
            "target_position": self.data.site("target_center").xpos.copy(),
            "time": float(self.data.time),
        }
        if include_rgb:
            obs["images"] = {CAMERA_NAME: self.render_rgb()}
        return obs

    def get_world_poses(self):
        """World-frame debug poses, metres and unit quaternions in wxyz order.

        finger_opening_width_m is the nominal jaw aperture derived from the two
        simulated slide coordinates; it is not an actuator command or pad mesh gap.
        """
        def body_pose(name):
            body = self.data.body(name)
            return {"position_m": body.xpos.copy(),
                    "quaternion_wxyz": body.xquat.copy()}

        site = self.data.site(GRASP_SITE_NAME)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, site.xmat)
        return {
            "coordinate_frame": "MuJoCo world",
            "hand": body_pose("hand"),
            "grasp_center": {"position_m": site.xpos.copy(),
                             "quaternion_wxyz": quat,
                             "rotation_matrix": site.xmat.reshape(3, 3).copy()},
            "cube": body_pose("red_cube"),
            "tray_target": {"position_m": self.data.site("target_center").xpos.copy(),
                            "quaternion_wxyz": self.data.body("target_tray").xquat.copy()},
            "finger_opening_width_m": float(self.data.qpos[self.finger_qpos].sum()),
        }

    def _get_renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        return self._renderer

    def render_rgb(self):
        renderer = self._get_renderer()
        renderer.update_scene(self.data, camera=CAMERA_NAME)
        return renderer.render().copy()

    def render_segmentation(self):
        renderer = self._get_renderer()
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(self.data, camera=CAMERA_NAME)
            return renderer.render().copy()
        finally:
            renderer.disable_segmentation_rendering()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
