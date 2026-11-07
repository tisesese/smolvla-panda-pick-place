"""Unchanged expert helper functions extracted from the original controllers."""
import numpy as np
import mujoco

def close_target(r,cfg):
    """Memoryless, continuous piecewise-linear function; normalized aperture."""
    lo=cfg['plateau_aperture_ratio'];hold=cfg['hold_command']
    return hold if r<=lo else hold+(cfg['open_command']-hold)*(r-lo)/(1-lo)

def gate_metrics(scene, orientation, gate):
    poses=scene.get_world_poses()
    ee=np.asarray(poses['grasp_center']['position_m']);cube=np.asarray(poses['cube']['position_m'])
    err=np.empty(3);mujoco.mju_subQuat(err,orientation,poses['grasp_center']['quaternion_wxyz'])
    values={'grasp_cube_distance_m':float(np.linalg.norm(ee-cube)),
        'orientation_error_rad':float(np.linalg.norm(err)),
        'max_arm_qvel_rad_s':float(np.abs(scene.data.qvel[scene.arm_dofs]).max()),
        'finger_width_m':poses['finger_opening_width_m']}
    checks={'position':values['grasp_cube_distance_m']<=gate['grasp_cube_distance_m_max'],
        'orientation':values['orientation_error_rad']<=gate['orientation_error_rad_max'],
        'velocity':values['max_arm_qvel_rad_s']<=gate['max_arm_qvel_rad_s'],
        'open':values['finger_width_m']>=gate['finger_width_m_min']}
    return {**values,'checks':checks,'ready':all(checks.values())}
