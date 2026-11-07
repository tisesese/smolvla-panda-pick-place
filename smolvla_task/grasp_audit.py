"""Replay actual expert/model commands to compare grasp mechanics."""
import json
import time

import mujoco
import numpy as np

from .config import EVIDENCE, OUTPUT
from .data import Demonstrations
from .environment import PandaTask
from .runtime import read, setup, write


def mechanics(env):
    native=env.native
    m=native.last_info
    site=native.data.site("grasp_center")
    rotation=site.xmat.reshape(3,3)
    offset=rotation.T@(np.asarray(m["cube_xyz_m"])-site.xpos)
    contacts=[]
    for i in range(native.data.ncon):
        contact=native.data.contact[i]
        b1,b2=native.model.geom_bodyid[[int(contact.geom1),int(contact.geom2)]]
        if native.cube_id not in [b1,b2]:continue
        other=int(b2 if b1==native.cube_id else b1)
        if native.model.body(other).name not in ["left_finger","right_finger"]:continue
        f=np.zeros(6)
        mujoco.mj_contactForce(native.model,native.data,i,f)
        contacts.append(dict(body=native.model.body(other).name,local_force_N=f[:3],friction=contact.friction[:2].copy()))
    return dict(time_s=float(native.data.time),cube_z_m=float(m["cube_xyz_m"][2]),
                cube_bottom_clearance_m=m["bottom_clearance_above_table_m"],bilateral=m["bilateral"],
                finger_normal_N=m["finger_normal_force_N"],grasp_offset_hand_m=offset,
                hand_z_world=rotation[:,2].copy(),gripper_width_m=m["gripper_width_m"],
                gripper_target=float(native.previous[7]),contacts=contacts)


def run():
    setup()
    out=OUTPUT/"step6/grasp_mechanics_v1"
    assert not out.exists(),"Retain previous mechanics audits"
    out.mkdir(parents=True)
    ds=Demonstrations([0])
    pos=read(EVIDENCE/"plan.json")["positions"][0]
    sources={"expert":ds.numeric[0]["action"]}
    for k in [25,50]:
        p=OUTPUT/f"rollouts/step6_20000_window{k}_pilot_v1/episode000-K{k}/commands.jsonl"
        sources[f"policy_window{k}"]=np.asarray([json.loads(s)["issued"] for s in p.read_text().splitlines()],np.float32)
    summary={}
    for label,commands in sources.items():
        records=[]
        begin=time.perf_counter()
        with PandaTask() as env:
            env.reset(seed=pos["seed"],options={"cube_xyz_m":pos["xyz_m"]})
            for action in commands:
                _,_,done,truncated,info=env.step(action,include_rgb=False)
                records.append(mechanics(env))
                if done or truncated:break
            samples=[]
            clock=np.asarray([r["time_s"] for r in records])
            for t in [7.5,8,9,10,11,12,13]:
                samples.append(records[int(np.argmin(abs(clock-t)))])
            result=dict(source=label,diagnostic_replay_not_new_policy_trial=True,success=info["success"],
                        reason=info["reason"],peak_cube_bottom_clearance_m=max(r["cube_bottom_clearance_m"] for r in records),
                        samples=samples,wall_seconds=time.perf_counter()-begin)
        write(out/(label+".json"),result)
        summary[label]=result
        print(label,dict(success=result["success"],peak_clearance_m=result["peak_cube_bottom_clearance_m"],seconds=result["wall_seconds"]),flush=True)
    write(out/"complete.json",dict(completed=True,diagnostic_only=True,sources=list(summary)))

