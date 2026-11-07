"""Thin native reset/step task; policy observations never contain object truth."""
from collections import deque
import numpy as np
import mujoco
from robot_scene.panda_scene import PandaTableScene
from robot_scene.vla_adapter import ObservationAdapter,ActionAdapter
from robot_scene.reset import ResetPlacement
from .common import SCENE,TASK,OUT,read

class PhysicalStop(RuntimeError):pass

class PandaPickPlaceEnv:
    def __init__(self):
        self.scene=PandaTableScene(SCENE);self.model=self.scene.model;self.data=self.scene.data
        self.observer=ObservationAdapter(self.scene,task=TASK);self.adapter=ActionAdapter(self.model)
        self.cfg=read(OUT/'contract.json');self._image=None;self._image_time=None
        root=self.model.body('link0').id;self.robot=set()
        for b in range(1,self.model.nbody):
            p=b
            while p and p!=root:p=int(self.model.body_parentid[p])
            if p==root and b!=root:self.robot.add(b)
        self.cube_id=self.model.body('red_cube').id;self.cube_joint=self.model.joint('cube_free').id
        self.cube_dof=int(self.model.jnt_dofadr[self.cube_joint]);self.half=self.model.geom('red_cube_geom').size.copy()
        self.base_id=self.model.geom('tray_base').id
        self.obstacles={self.model.geom(x).id for x in ['table_top','floor','tray_base','tray_left','tray_right','tray_front','tray_back']}
        self.limits=self.model.actuator_ctrlrange.copy();self.limits[7]=[0,1]
        for i in range(7):self.limits[i,:]=[max(self.limits[i,0],self.model.jnt_range[self.model.joint(f'joint{i+1}').id,0]),min(self.limits[i,1],self.model.jnt_range[self.model.joint(f'joint{i+1}').id,1])]
    def reset(self,*,seed=810000,options=None,include_rgb=True):
        self.scene.reset();xyz=np.asarray((options or {}).get('cube_xyz_m',[.13,-.19,.782]),float)
        self.placement=ResetPlacement(seed,xyz,0);self.reset_info=self.placement(self.scene)
        self.initial_cube_z=float(xyz[2]);self.initial_ctrl=np.r_[self.data.ctrl[:7],1.]
        self.previous=self.initial_ctrl.copy();self.stable_steps=0;self.stable_positions=deque(maxlen=1001)
        self.ever_lifted=False;self.terminated=False;self.truncated=False;self.reason=None;self.success=False;self.tick_count=0
        self.eq=self.data.eq_active.copy();self.initial_warnings=np.array([w.number for w in self.data.warning]);self._image=None
        self.peak={'arm_speed_rad_s':0.,'finger_speed_m_s':0.,'finger_force_N':0.,'gripper_actuator_force_N':0.,'tracking_error_rad':0.,'joint_target_delta_rad':0.,'robot_obstacle_force_N':0.}
        self.support_seen=False;self.last_info=self.metrics()
        self.contact_window=[];self.require_grasp=False;self.lost_contact_steps=0
        return self.observe(include_rgb),{'reset':self.reset_info,'privileged_task_metrics':self.last_info}
    def image(self):
        if self._image is None or self._image_time!=float(self.data.time):
            self._image=self.scene.render_rgb();self._image_time=float(self.data.time)
        return self._image.copy()
    def observe(self,include_rgb=True):
        o=self.observer.observe(include_rgb=False)
        if include_rgb:o['observation.images.fixed_rgb']=self.image()
        return o
    def metrics(self):
        d=self.data;m=self.model;cube=d.body('red_cube');R=cube.xmat.reshape(3,3);extent=np.abs(R)@self.half
        base=d.geom('tray_base');top=float(base.xpos[2]+m.geom('tray_base').size[2])
        left=float(d.geom('tray_left').xpos[0]+m.geom('tray_left').size[0]);right=float(d.geom('tray_right').xpos[0]-m.geom('tray_right').size[0])
        front=float(d.geom('tray_front').xpos[1]+m.geom('tray_front').size[1]);back=float(d.geom('tray_back').xpos[1]-m.geom('tray_back').size[1])
        margin=self.cfg['success']['interior_margin_m'];inside=bool(cube.xpos[0]-extent[0]>=left+margin and cube.xpos[0]+extent[0]<=right-margin and cube.xpos[1]-extent[1]>=front+margin and cube.xpos[1]+extent[1]<=back-margin)
        support=0.;robot_contacts=[];finger={'left':0.,'right':0.};allcontacts=[];obstacle_force=0.
        for i in range(d.ncon):
            c=d.contact[i];g1,g2=int(c.geom1),int(c.geom2);b1,b2=m.geom_bodyid[[g1,g2]]
            force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);normal=max(0.,float(force[0]));names=[m.geom(g1).name,m.geom(g2).name]
            robot_obstacle=(b1 in self.robot and g2 in self.obstacles) or (b2 in self.robot and g1 in self.obstacles)
            if robot_obstacle:obstacle_force=max(obstacle_force,normal)
            if self.cube_id in (b1,b2):
                other=int(g2 if b1==self.cube_id else g1);body=int(m.geom_bodyid[other]);bodyname=m.body(body).name
                allcontacts.append({'geoms':names,'normal_N':normal,'distance_m':float(c.dist)})
                if other==self.base_id:support+=normal
                if body in self.robot:robot_contacts.append({'body':bodyname,'normal_N':normal,'distance_m':float(c.dist)})
                if bodyname in ['left_finger','right_finger']:finger['left' if bodyname=='left_finger' else 'right']+=normal
            if robot_obstacle and (normal>self.cfg['safety']['obstacle_contact_N_max'] or c.dist<-.001):raise PhysicalStop('robot_obstacle_collision:'+str(names))
            if (b1 in self.robot or b2 in self.robot) and normal>15.:raise PhysicalStop('severe_robot_contact_force')
        qvel=d.qvel[self.cube_dof:self.cube_dof+6];site=d.site('grasp_center').xpos
        bottom=float(cube.xpos[2]-extent[2]);tabletop=float(d.geom('table_top').xpos[2]+m.geom('table_top').size[2])
        return {'cube_xyz_m':cube.xpos.copy(),'cube_quat_wxyz':cube.xquat.copy(),'cube_bottom_z_m':bottom,
            'cube_linear_speed_m_s':float(np.linalg.norm(qvel[:3])),'cube_angular_speed_rad_s':float(np.linalg.norm(qvel[3:])),
            'cube_within_interior':inside,'tray_inner_xy_bounds_m':[left,right,front,back],'tray_top_z_m':top,
            'tray_base_support_force_N':support,'cube_robot_contacts':robot_contacts,'cube_contacts':allcontacts,
            'finger_normal_force_N':finger,'bilateral':min(finger.values())>.05,'cube_grasp_distance_m':float(np.linalg.norm(cube.xpos-site)),
            'grasp_site_xyz_m':site.copy(),'gripper_width_m':self.observer.finger_width_m(),
            'lift_from_initial_m':float(cube.xpos[2]-self.initial_cube_z),'bottom_clearance_above_table_m':bottom-tabletop,
            'robot_obstacle_force_N':obstacle_force}
    def health(self,action):
        d=self.data;s=self.cfg['safety']
        if not all(np.isfinite(x).all() for x in [d.qpos,d.qvel,d.qacc]):raise PhysicalStop('nonfinite_physics')
        if not np.array_equal(d.eq_active,self.eq) or np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise PhysicalStop('external_force_or_equality_change')
        q=d.qpos[self.scene.arm_qpos];v=float(np.abs(d.qvel[self.scene.arm_dofs]).max());fd=[self.model.joint(n).dofadr[0] for n in ['finger_joint1','finger_joint2']]
        fv=float(np.abs(d.qvel[fd]).max());af=float(abs(d.actuator_force[7]));tracking=float(np.abs(q-action[:7]).max())
        for key,value in [('arm_speed_rad_s',v),('finger_speed_m_s',fv),('gripper_actuator_force_N',af),('tracking_error_rad',tracking)]:self.peak[key]=max(self.peak[key],value)
        if v>s['arm_speed_rad_s_max']:raise PhysicalStop('arm_speed_limit')
        if fv>s['finger_speed_m_s_max']:raise PhysicalStop('finger_speed_limit')
        if af>s['gripper_actuator_force_N_max']+1e-6:raise PhysicalStop('gripper_force_limit')
        if tracking>s['tracking_error_rad_max']:raise PhysicalStop('tracking_error_limit')
        if np.any(q<self.limits[:7,0]-1e-4) or np.any(q>self.limits[:7,1]+1e-4):raise PhysicalStop('measured_joint_limit')
        if np.any(np.array([w.number for w in d.warning])>self.initial_warnings):raise PhysicalStop('mujoco_warning')
        info=self.metrics();self.peak['finger_force_N']=max(self.peak['finger_force_N'],max(info['finger_normal_force_N'].values()))
        self.peak['robot_obstacle_force_N']=max(self.peak['robot_obstacle_force_N'],info['robot_obstacle_force_N'])
        if max(info['finger_normal_force_N'].values())>s['finger_contact_force_N_max']:raise PhysicalStop('finger_contact_force_limit')
        if info['cube_xyz_m'][2]<.70:raise PhysicalStop('cube_fallen_off_work_surface')
        return info
    def stable(self,info,action):
        s=self.cfg['success'];supported=info['tray_base_support_force_N']>=s['base_support_force_N_min'] and abs(info['cube_bottom_z_m']-info['tray_top_z_m'])<=s['support_height_tolerance_m']
        self.support_seen=self.support_seen or supported
        self.ever_lifted=self.ever_lifted or (info['lift_from_initial_m']>=.10 and info['bottom_clearance_above_table_m']>=.08 and info['bilateral'])
        good=(self.ever_lifted and supported and info['cube_within_interior'] and not info['cube_robot_contacts']
            and info['cube_linear_speed_m_s']<=s['linear_speed_m_s_max'] and info['cube_angular_speed_rad_s']<=s['angular_speed_rad_s_max']
            and info['cube_grasp_distance_m']>=s['robot_cube_clearance_m_min'] and info['gripper_width_m']>=.075 and action[7]>=.98)
        if good:self.stable_steps+=1;self.stable_positions.append(info['cube_xyz_m'].copy())
        else:self.stable_steps=0;self.stable_positions.clear()
        if self.stable_positions and np.ptp(np.array(self.stable_positions),axis=0).max()>s['position_drift_m_max']:
            self.stable_steps=0;self.stable_positions.clear()
        return self.stable_steps>=1000
    def step(self,action,*,include_rgb=True):
        if self.terminated or self.truncated:raise RuntimeError('step_after_done')
        a=np.asarray(action,dtype=np.float32).astype(float)
        if a.shape!=(8,) or not np.isfinite(a).all() or np.any(a<self.limits[:,0]) or np.any(a>self.limits[:,1]):raise ValueError('invalid_raw_8D_action; no silent clipping')
        delta=float(np.abs(a[:7]-self.previous[:7]).max());self.peak['joint_target_delta_rad']=max(self.peak['joint_target_delta_rad'],delta)
        if delta>self.cfg['safety']['joint_target_increment_rad_max']+4e-7:raise ValueError('raw_joint_target_increment_limit')
        if abs(a[7]-self.previous[7])>.060001:raise ValueError('raw_gripper_target_increment_limit')
        ctrl=self.adapter.to_ctrl(a);np.testing.assert_allclose(ctrl[:7],a[:7],atol=0,rtol=0);self.data.ctrl[:]=ctrl
        start=float(self.data.time)
        self.contact_window=[]
        try:
            for _ in range(10):
                if self.data.time>=30-1e-9:self.truncated=True;self.reason='simulation_timeout_30s';break
                mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
                info=self.health(a);self.last_info=info
                self.contact_window.append(list(info['finger_normal_force_N'].values()))
                if self.require_grasp and info['tray_base_support_force_N']<.1:
                    self.lost_contact_steps=0 if info['bilateral'] else self.lost_contact_steps+1
                    if self.lost_contact_steps>=10:raise PhysicalStop('lost_bilateral_payload_contact')
                    if info['cube_grasp_distance_m']>.04:raise PhysicalStop('payload_outside_gripper')
                else:self.lost_contact_steps=0
                self.success=self.stable(info,a)
            if self.success:self.terminated=True;self.reason='independent_tray_stability_2s'
            elif self.data.time>=30-1e-9:self.truncated=True;self.reason='simulation_timeout_30s'
        except PhysicalStop as e:self.terminated=True;self.reason=str(e);self.success=False
        self.previous=a.copy();self.tick_count+=1;self._image=None
        info={'success':self.success,'reason':self.reason,'sim_time_s':float(self.data.time),'physics_steps':round((float(self.data.time)-start)/.002),
            'stable_seconds':self.stable_steps*.002,'privileged_task_metrics':self.last_info,'peaks':dict(self.peak)}
        if self.terminated or self.truncated:self.hold()
        return self.observe(include_rgb),float(self.success),self.terminated,self.truncated,info
    def hold(self):
        if np.isfinite(self.data.qpos[self.scene.arm_qpos]).all():self.data.ctrl[:]=self.adapter.to_ctrl(np.r_[self.data.qpos[self.scene.arm_qpos],self.previous[7]])
    def close(self):self.scene.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
