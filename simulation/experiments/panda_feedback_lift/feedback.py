"""Spatial lead controller. No elapsed-time progress and no new IK implementation."""
import numpy as np

def lift(scene,ik,orientation,get_action,tick,events,ik_log,cfg,get_frame):
    f=cfg['feedback_lift'];q0=get_action()[:7].copy()
    actual0=scene.get_world_poses()['grasp_center']['position_m'].copy()
    ik.error(q0,actual0,orientation)
    start=ik.data.site(ik.site_id).xpos.copy();goal=start+np.array([0.,0.,.16])
    initial_error=q0-scene.data.qpos[scene.arm_qpos]
    events.append({'event':'consistent_lift_reference','frame':get_frame(),
        'actual_start_xyz_m':actual0,'command_reference_start_xyz_m':start,
        'reference_offset_m':float(np.linalg.norm(start-actual0)),
        'no_live_qpos_writes':True,'feedback_progress':True})
    # Preserve the exact dispatched position at handover; no FK/actual mismatch correction.
    tick('lift',np.r_[q0,cfg['gripper']['hold_command']])
    progress=0.
    for k in range(f['timeout_ticks']):
        previous=get_action()[:7].copy();measured=scene.data.qpos[scene.arm_qpos].copy()
        actual=scene.get_world_poses()['grasp_center']['position_m'].copy()
        observed_progress=float(actual[2]-actual0[2])
        residual=(previous-measured)-initial_error
        tracking_ready=float(np.abs(residual).max())<=f['tracking_residual_rad_max']
        # A measured stop eventually exhausts the spatial lead. The clock cannot
        # advance this reference; max() only enforces a monotone lift path.
        requested=min(.16,max(progress,observed_progress+f['lead_m'])) if tracking_ready else progress
        nxt=min(requested,progress+f['reference_increment_m_max'])
        target=start+np.array([0.,0.,nxt])
        q,m=ik.solve(target,orientation,previous)
        jump=float(abs(q-previous).max())
        if jump>f['joint_target_increment_rad_max']:
            raise RuntimeError('feedback_lift_joint_target_increment_limit')
        ik_log.append({'phase':'lift','frame':get_frame(),**m,'target_position_m':target,
            'actual_xyz_m':actual,'command_q_rad':q,'actual_q_rad':measured,
            'actual_qvel_rad_s':scene.data.qvel[scene.arm_dofs].copy(),
            'tracking_error_rad':q-measured,'tracking_residual_rad':residual,
            'observed_progress_m':observed_progress,'reference_progress_m':nxt,
            'tracking_gate_ready':tracking_ready,'target_delta_rad':jump,
            'elapsed_time_not_used_for_progress':True})
        tick('lift',np.r_[q,cfg['gripper']['hold_command']]);progress=nxt
        if progress>=.16-1e-12:
            events.append({'event':'feedback_lift_reference_complete','frame':get_frame(),
                           'feedback_ticks':k+1,'observed_progress_m':observed_progress})
            return goal
    raise RuntimeError('feedback_lift_timeout; no forced progression')
