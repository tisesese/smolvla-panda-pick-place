"""Native scripted expert composed from the admitted grasp components."""
import json,time
from collections import Counter
import numpy as np
import mujoco
from PIL import Image
from robot_scene.scripted_grasp import NativeJacobianIK
from experiments.panda_descent_feedback.correction import BoundedDescentCorrection
from robot_scene.expert_helpers import close_target
from experiments.panda_feedback_lift.feedback import lift as feedback_lift
from robot_scene.expert_helpers import gate_metrics
from .common import OUT,read,write,convert,log
from .environment import PandaPickPlaceEnv

class TaskEnd(Exception):pass

class Video:
    def __init__(self,path):
        import av
        self.av=av;self.file=av.open(str(path),'w');self.stream=self.file.add_stream('libx264',rate=10)
        self.stream.width=640;self.stream.height=480;self.stream.pix_fmt='yuv420p';self.stream.options={'crf':'21','preset':'veryfast'}
    def add(self,rgb):
        f=self.av.VideoFrame.from_ndarray(rgb,format='rgb24')
        for packet in self.stream.encode(f):self.file.mux(packet)
    def close(self):
        for packet in self.stream.encode():self.file.mux(packet)
        self.file.close()

def run(folder,position,*,recorder=None,video=False):
    folder.mkdir(parents=True,exist_ok=False);(folder/'rgb').mkdir();cfg=read(OUT/'contract.json');events=[];ik_log=[];counts=Counter();failure=None
    movie=Video(folder/'pick-place.mp4') if video else None;started=time.perf_counter();rows=[];current='reset';action=None
    with PandaPickPlaceEnv() as env,(folder/'trajectory.jsonl').open('w') as journal:
        env.reset(seed=position['seed'],options={'cube_xyz_m':position['xyz_m']},include_rgb=False)
        scene=env.scene;ik=NativeJacobianIK(scene);orientation=scene.get_world_poses()['grasp_center']['quaternion_wxyz'].copy()
        action=np.r_[scene.data.ctrl[:7],1.];last_phase=None;correction=None;phase_started={}
        def tick(phase,target):
            nonlocal action,last_phase,current
            current=phase
            if env.terminated or env.truncated:raise TaskEnd()
            if env.data.time>=30-1e-9:raise RuntimeError('30s_timeout_before_command')
            target=np.asarray(target,dtype=np.float32).astype(float);ctrl=env.adapter.to_ctrl(target)
            assert np.array_equal(ctrl[:7],target[:7]) and abs(ctrl[7]-255*target[7])<1e-12
            state=env.observer.state();info=env.last_info;f=env.tick_count
            row={'control_tick':f,'phase':phase,'sim_time_s':float(env.data.time),'state':state.copy(),'action':target.copy(),'ctrl':ctrl.copy(),
                'arm_qvel_rad_s':env.data.qvel[scene.arm_dofs].copy(),'privileged':info}
            if phase not in phase_started:phase_started[phase]=float(env.data.time)
            if recorder is not None:recorder(env,phase,target.copy(),ctrl.copy(),f)
            if phase!=last_phase or (phase in ['release','verify'] and counts[phase]%25==0):
                path=folder/'rgb'/f'{f:04d}-{phase}.png';Image.fromarray(env.image()).save(path);row['rgb_path']=str(path)
            if movie and f%5==0:movie.add(env.image())
            _,_,terminated,truncated,post=env.step(target,include_rgb=False);action=target;counts[phase]+=1;last_phase=phase
            row.update(post_state=env.observer.state(),post_privileged=env.last_info,physics_steps=post['physics_steps'],stable_seconds=post['stable_seconds'])
            journal.write(json.dumps(row,default=convert)+'\n');journal.flush();rows.append(row)
            if terminated or truncated:
                if not env.success:raise RuntimeError(env.reason)
                raise TaskEnd()
        def hold(phase,n):
            for _ in range(n):tick(phase,action)
        def arrived(goal):
            p=scene.get_world_poses()['grasp_center'];e=np.empty(3);mujoco.mju_subQuat(e,orientation,p['quaternion_wxyz'])
            return np.linalg.norm(p['position_m']-goal)<=.015 and np.linalg.norm(e)<=.02 and np.abs(env.data.qvel[scene.arm_dofs]).max()<=.08
        def arrival(phase,goal):
            endpoint_feedback=None
            for _ in range(100):
                if arrived(goal):return
                if phase in ['transport_arrival','place_arrival','retract_arrival'] and np.abs(env.data.qvel[scene.arm_dofs]).max()<=.08:
                    if endpoint_feedback is None:endpoint_feedback=BoundedDescentCorrection(scene,ik.site_id,action[:7])
                    q,detail=endpoint_feedback.propose(scene,goal,orientation,action[:7]);events.append({'event':'bounded_endpoint_feedback','phase':phase,**detail})
                    tick(phase,np.r_[q,action[7]])
                else:tick(phase,action)
            raise RuntimeError(phase+'_arrival_timeout')
        def move(phase,goal,seconds,*,command_start=True,settle=True):
            q=action[:7].copy();actual=scene.get_world_poses()['grasp_center']['position_m'].copy()
            if command_start:ik.error(q,actual,orientation);start=ik.data.site(ik.site_id).xpos.copy()
            else:start=actual
            n=round(seconds/.02);goal=np.asarray(goal,float)
            for k in range(1,n+1):
                a=k/n;blend=10*a**3-15*a**4+6*a**5;target=start+blend*(goal-start)
                q,m=ik.solve(target,orientation,q);ik_log.append({'phase':phase,'frame':env.tick_count,'target':target,**m})
                tick(phase,np.r_[q,action[7]])
            if settle:arrival(phase+'_arrival',goal)
        try:
            hold('home_settle',50)
            cube=scene.get_world_poses()['cube']['position_m'].copy();env.initial_cube_z=float(cube[2])
            move('approach',cube+[0,0,.16],3.,command_start=False)
            cube=scene.get_world_poses()['cube']['position_m'].copy();move('descend',cube,2.5,command_start=False,settle=False)
            for _ in range(cfg['arrival_gate']['timeout_ticks']+1):
                g=gate_metrics(scene,orientation,cfg['arrival_gate'])
                if g['ready']:events.append({'event':'descend_ready','time':float(env.data.time),**g});break
                if correction is None:correction=BoundedDescentCorrection(scene,ik.site_id,action[:7])
                q,detail=correction.propose(scene,cube,orientation,action[:7]);events.append({'event':'bounded_descent_feedback',**detail})
                tick('descend_arrival',np.r_[q,action[7]])
            else:raise RuntimeError('descend_arrival_timeout')
            for _ in range(cfg['lift_gate']['close_timeout_ticks']):
                state=env.observer.state();g=close_target(float(state[7]),cfg['gripper']);tick('grasp',np.r_[action[:7],g])
                metrics=env.last_info;forces=np.min(np.asarray(env.contact_window),axis=0);fd=[scene.model.joint(n).dofadr[0] for n in ['finger_joint1','finger_joint2']]
                ready=(.60<=env.observer.state()[7]<=.64 and metrics['cube_grasp_distance_m']<=.01 and min(forces)>=.25
                    and np.abs(env.data.qvel[fd]).max()<=.008 and np.abs(env.data.qvel[scene.arm_dofs]).max()<=.08)
                if ready:events.append({'event':'bilateral_grasp_ready','time':float(env.data.time),'forces_N':forces});break
            else:raise RuntimeError('grasp_contact_timeout')
            env.require_grasp=True
            goal=feedback_lift(scene,ik,orientation,lambda:action,tick,events,ik_log,cfg,lambda:env.tick_count)
            arrival('lift_arrival',goal)
            metrics=env.last_info
            if not env.ever_lifted:raise RuntimeError('original_lift_clearance_not_met')
            events.append({'event':'lifted','time':float(env.data.time),'metrics':metrics})
            offset=np.asarray(metrics['cube_xyz_m'])-scene.get_world_poses()['grasp_center']['position_m']
            tray=scene.data.geom('tray_base').xpos.copy();base_top=metrics['tray_top_z_m'];safe_cube_z=max(.97,float(metrics['cube_xyz_m'][2]))
            transport_goal=np.r_[tray[:2]-offset[:2],safe_cube_z-offset[2]]
            move('transport',transport_goal,4.5)
            # A low approach still keeps the cube bottom above the rim; center
            # descent and physical support seeking happen only inside the tray.
            preplace=np.r_[tray[:2]-offset[:2],.84-offset[2]];move('place',preplace,1.8)
            ik.error(action[:7],scene.get_world_poses()['grasp_center']['position_m'],orientation);reference=ik.data.site(ik.site_id).xpos.copy();stable_support=0
            lower=base_top+env.half[2]-.015-offset[2]
            for _ in range(220):
                info=env.last_info;supported=info['tray_base_support_force_N']>=.1 and abs(info['cube_bottom_z_m']-base_top)<=.003
                if supported:
                    stable_support+=1;tick('place_support',action)
                    if stable_support>=5:break
                else:
                    stable_support=0;reference[2]=max(lower,reference[2]-.0004)
                    q,m=ik.solve(reference,orientation,action[:7]);ik_log.append({'phase':'place_support','target':reference.copy(),**m})
                    tick('place_support',np.r_[q,action[7]])
            else:raise RuntimeError('tray_support_not_reached')
            events.append({'event':'real_tray_support_before_release','time':float(env.data.time),'metrics':env.last_info})
            env.require_grasp=False
            for _ in range(100):
                g=min(1.,action[7]+.02);tick('release',np.r_[action[:7],g])
                if g>=.999 and env.last_info['gripper_width_m']>=.075 and not env.last_info['cube_robot_contacts']:break
            else:raise RuntimeError('release_not_independent')
            events.append({'event':'released_no_robot_contact','time':float(env.data.time),'metrics':env.last_info})
            actual=scene.get_world_poses()['grasp_center']['position_m'].copy();move('retract',actual+[0,0,.18],1.8)
            while not env.terminated and not env.truncated:tick('verify',action)
        except TaskEnd:pass
        except Exception as e:
            import traceback
            failure={'phase':current,'type':type(e).__name__,'message':str(e),'time_s':float(env.data.time),'traceback':traceback.format_exc()};env.hold()
        finally:
            if movie:movie.close()
            Image.fromarray(env.image()).save(folder/'final.png')
            final=env.last_info;carry=[r for r in rows if r['phase'] in ['lift','lift_arrival','transport']]
            result={'success':bool(env.success and failure is None),'failure':failure,'reason':env.reason,
                'seed':position['seed'],'initial_cube_xyz_m':position['xyz_m'],'sim_seconds_including_reset':float(env.data.time),
                'task_limit_seconds':30,'within30':bool(env.data.time<=30+1e-8),'frames_including_reset':len(rows),'frames_without_reset':sum(r['phase']!='home_settle' for r in rows),
                'phase_seconds':{k:v*.02 for k,v in counts.items()},'phase_start_times':phase_started,'events':events,
                'final':final,'independent_stable_seconds':env.stable_steps*.002,'ever_lifted':env.ever_lifted,
                'carry_bilateral_fraction':float(np.mean([r['post_privileged']['bilateral'] for r in carry])) if carry else None,
                'carry_max_cube_grasp_distance_m':max([r['post_privileged']['cube_grasp_distance_m'] for r in carry] or [0.]),
                'safety_peaks':env.peak,'no_external_forces':not np.any(env.data.qfrc_applied) and not np.any(env.data.xfrc_applied),
                'equality_unchanged':bool(np.array_equal(env.eq,env.data.eq_active)),'reset_only_cube_qpos_writes':1,'execution_cube_qpos_writes':0,
                'wall_seconds':time.perf_counter()-started,'task':cfg['task'],'controller_revision':cfg['revision'],
                'descent_corrections':correction.records if correction else [],'raw_model_predictions':None,'VLA_commands':0}
            write(folder/'result.json',result);write(folder/'ik-log.json',ik_log)
    return result
